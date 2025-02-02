import torch
from torch import nn
from typing import Optional, Tuple, List
import math

################################### Gemma Model ###################################


class GemmaConfig(nn.Module):
    # The hyperparameters of the Gemma model can be found here: https://huggingface.co/google/paligemma-3b-pt-224/blob/main/config.json
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class KVCache:

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # Shape of the key_cache: (batch_size, num_key_value_heads, sequence_length, head_dim)
            return self.key_cache[0].shape[-2]

    def update(
        self, key: torch.Tensor, value: torch.Tensor, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-cache of this layer, let's create it
            self.key_cache.append(key)
            self.value_cache.append(value)
        else:
            # otherwise we concatenate the new key and value to the existing cache
            # Shape of the key_cache: (batch_size, num_key_value_heads, sequence_length, head_dim)
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value], dim=-2
            )

        # we return all the keys and values of the cache (including the new ones)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # dim is the hidden size of the model
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()) * (1.0 + self.weight.float())
        return output.type_as(x)


class MLP(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # (batch_size, sequence_length, hidden_size) => (batch_size, sequence_length, intermediate_size)
        gated_activation = self.gate_proj(x)
        gated_activation = torch.gelu(gated_activation, approximate="tanh")
        upscaled_input = self.up_proj(x)
        transformed = gated_activation * upscaled_input

        # (batch_size, sequence_length, intermediate_size) => (batch_size, sequence_length, hidden_size)
        output = self.down_proj(transformed)
        return output


class RotaryEmbedding(nn.Module):

    def __init__(self, dim: int, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim  # head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # We calculate the theta according to the formula theta_i = base^(2i/dim) where i = 0, 1, 2, ..., dim // 2
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # (batch_size, num_heads, sequence_length, head_dim)
        self.inv_freq.to(x.device)
        # (batch_size, head_dim//2, 1)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        # position_ids_expanded: (batch_size, 1, sequence_length)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            # We multiply each theta by the position (which is the argument of the sin and cos functions)
            # freqs: (batch_size, head_dim//2, 1) @ (batch_size, 1, sequence_length) => (batch_size, sequence_length, head_dim//2)
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat(
                (freqs, freqs), dim=-1
            )  # HF implementation, which isn't exactly like in the paper.
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor
    x1 = x[..., : x.shape[-1] // 2]  # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :]  # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


# Apply the rotary position embedding to the query and key
def apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)  # add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim)  # add the head dimension
    query = (query * cos) + (rotate_half(query) * cos)
    key = (query * sin) + (rotate_half(key) * sin)
    return query, key


class GroupQueryAttention(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert (
            self.hidden_size % self.num_heads == 0
        ), "Hidden size must be divisible by the number of heads."

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )  # [1024, 8 * 128] = [1024, 1024]
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )  # [1024, 2 * 128] = [1024, 128]
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )  # [1024, 2 * 128] = [1024, 128]
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.size()
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch_size, num_key_value_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(
            batch_size, num_key_value_heads * n_rep, seq_len, head_dim
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.size()
        # (batch_size, sequence_length, hidden_size) => (batch_size, num_heads, sequence_length, head_dim)
        query = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        # (batch_size, sequence_length, hidden_size) => (batch_size, num_key_value_heads, sequence_length, head_dim)
        key = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        # (batch_size, sequence_length, hidden_size) => (batch_size, num_key_value_heads, sequence_length, head_dim)
        value = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        cos, sin = self.rotary_emb(value, position_ids, seq_len=None)
        # (batch_size, num_heads, sequence_length, head_dim), (batch_size, num_key_value_heads, sequence_length, head_dim)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if kv_cache is not None:
            key, value = kv_cache.update(key, value, self.layer_idx)

        # As it is GQA, we need to repeat the key and values to match the number of heads of the query
        key = self.repeat_kv(key, self.num_key_value_groups)
        value = self.repeat_kv(value, self.num_key_value_groups)

        # (batch_size, num_heads, sequence_length, sequence_length)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )

        assert attention_mask is not None, "Attention mask is required for Gemma model."
        # The attention_mask will always be a tensor of zeros. We don't mask anything as we don't have any paddings.
        # We don't pad because we always let the user prompt to also attend futur tokens. Decision from the PaliGemma authors.
        attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value)

        # (batch_size, num_heads, sequence_length, head_dim) => (batch_size, sequence_length, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Concatenate all the heads together: (batch_size, sequence_length, num_heads * head_dim)
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Mix all the heads otherwise each head is independent from the others
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class GemmaLayer(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GroupQueryAttention(config=config, layer_idx=layer_idx)
        self.mlp = MLP(config=config)
        self.input_layer_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output_layer_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:

        residual = hidden_states

        hidden_states = self.input_layer_norm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        # (batch_size, sequence_length, hidden_size)
        hidden_states

        residual = hidden_states
        hidden_states = self.output_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # (batch_size, sequence_length, hidden_size)
        return hidden_states


class GemmaModel(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.padding_idx = config.pad_token_id

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                GemmaLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        # (batch_size, sequence_length, hidden_size)
        normalizer = self.tensor(
            self.config.hidden_size**0.5, dtype=inputs_embeds.dtype
        )
        output = inputs_embeds * normalizer

        for layer in self.layers:
            output = layer(
                output,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # (batch_size, sequence_length, hidden_size)
        output = self.norm(output)

        return output


class Gemma(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.model = GemmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        # inputs_embeds: (batch_size, sequence_length, hidden_size)
        # outputs = (batch_size, sequence_length, hidden_size)
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        logits = self.lm_head(outputs).float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        # We return the logits and the key-value cache if it is used.
        return return_data
