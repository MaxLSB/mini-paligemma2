import torch
import torch.nn as nn


class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_channels,
        image_size,
        patch_size,
        layer_norm_eps,
        attention_dropout,
        num_image_tokens,
    ):
        super.__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid",  # no padding added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values):
        patch_embeds = self.patch_embedding(
            pixel_values
        )  # [batch_size, embed_dim, num_patches_h, num_patches_w] | Also: num_patches_h * num_patches_w = num_patches
        embeddings = patch_embeds.flatten(2)  # [batch_size, embed_dim, num_patches]
        embeddings = embeddings.transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


# Need to implement the encoder part
class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        pass

    def forward(self, x):
        pass


class SiglipVisionTransformer(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embedding = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values):

        x = self.embedding(pixel_values)
        x = self.encoder(x)
        x = self.post_layernorm(x)
        return x


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values):
        return self.vision_model(pixel_values=pixel_values)
