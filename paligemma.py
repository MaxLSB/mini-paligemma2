import torch
import math
from torch import nn
from typing import Optional, Tuple
from siglip import SiglipVisionConfig, SiglipVisionModel
from gemma import GemmaConfig, Gemma
from projector import MultiModalProjector


################################### PaliGemma Model ###################################


class PaliGemmaConfig(SiglipVisionConfig):

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_text=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs
    ):
        super().__init__(vision_config, text_config, **kwargs)
        self.ignore_text = ignore_text
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.is_encoder_decoder = False  # Needed to load the HF weights

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)

        self.text_config.num_image_tokens = (
            self.vision_config.image_size // self.vision_config.patch_size
        ) ** 2
        self.vision_config.projection_dim = projection_dim


class PaliGemma(nn.Module):

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionModel(config)
        self.projector = MultiModalProjector(config)
        self.vocab_size = config.vocab_size
        self.language_model = Gemma(config)

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

    # Reusing a parameter of one layer in another layer
    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_embeds_with_image_embeds(
        self,
        image_embeds: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        _, _, embed_dim = image_embeds.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        # (batch_size, sequence_length, hidden_size)
        scaled_image_embeds = image_embeds * math.sqrt(self.config.hidden_size)

        # Combine the embeddings of the text tokens and image tokens and mask all the padding tokens.
        embedding = torch.zeros(
            batch_size,
            sequence_length,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        # (batch_size, sequence_length)
        text_mask = (input_ids != self.pad_token_id) & (input_ids != self.pad_token_id)
        # (batch_size, sequence_length)
        image_mask = input_ids == self.config.image_token_index
        # (batch_size, sequence_length)
        pad_mask = input_ids == self.pad_token_id

        # We expand the masks to the embedding dimension for torch.where
        text_mask = text_mask.unsqueeze(-1).expand_as(-1, -1, embed_dim)
        image_mask = image_mask.unsqueeze(-1).expand_as(-1, -1, embed_dim)
        pad_mask = pad_mask.unsqueeze(-1).expand_as(-1, -1, embed_dim)

        # Add the text embeddings
        embedding = torch.where(text_mask, inputs_embeds, embedding)
        # Add the image embeddings. We can't use torch.where because the sequence length of scaled_image_embeds is different.
        embedding = embedding.masked_scatter(image_mask, scaled_image_embeds)
        embedding = torch.where(pad_mask, torch.zeros_like(embedding), embedding)

        # We create the attention mask
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # If we are not generating tokens, the query can be the entire sequence.
            # Only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # If we are generating tokens, the query must be one token at a time.
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # We don't need to mask anything as we have a single token in the query.
            # Only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # (batch_size, q_len, kv_len) => (batch_size, num_heads_q, q_len, kv_len)
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # we create a position_ids based on the size of the attention mask
            position_ids = (
                (attention_mask.cumsum(-1))
                .masked_fill_((attention_mask == 0), 1)
                .to(device)
            )

        return embedding, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # Extract the input embeddings.
        # (batch_size, sequence_length, hidden_size)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # We now Merge text and images.
        # (batch_size, channels, height, width) => (batch_size, num_patches, embed_dim)
        image_features = self.vision_model(pixel_values.to(inputs_embeds.dtype))

        # Project the image features to the same size as the text embeddings.
        # (batch_size, num_patches, embed_dim) => (batch_size, sequence_length, hidden_size)
        image_embeds = self.projector(image_features)

        # Merge the embeddings of the text tokens and image tokens.
        inputs_embeds, attention_mask, position_ids = (
            self._merge_input_embeds_with_image_embeds(
                image_embeds, inputs_embeds, input_ids, attention_mask, kv_cache
            )
        )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs
