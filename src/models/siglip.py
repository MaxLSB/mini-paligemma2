import torch
import torch.nn as nn
from models.model_config import SiglipConfig


################################### Siglip Vision Encoder ###################################


class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config: SiglipConfig):
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
        )  # (batch_size, embed_dim, num_patches_h, num_patches_w) | Also: num_patches_h * num_patches_w = num_patches
        embeddings = patch_embeds.flatten(2)  # (batch_size, embed_dim, num_patches)
        embeddings = embeddings.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // config.num_attention_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        # (batch_size, num_patches, embed_dim)
        batch_size, num_patches, _ = x.size()
        # (batch_size, num_patches, embed_dim) => (batch_size, num_heads, num_patches, head_dim)
        query = (
            self.q_proj(x)
            .view(batch_size, num_patches, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(x)
            .view(batch_size, num_patches, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        values = (
            self.v_proj(x)
            .view(batch_size, num_patches, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim**0.5

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(
            1, 2
        ).contiguous()  # (batch_size, num_patches, num_heads, head_dim)
        attn_output = attn_output.view(
            batch_size, num_patches, self.embed_dim
        )  # (batch_size, num_patches, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class MLP(nn.Module):
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size

        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.self_attn = SiglipAttention(config)

    def forward(self, x):
        # (batch_size, num_patches, embed_dim)
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attn(x)
        x = residual + x

        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = residual + x
        # (batch_size, num_patches, embed_dim)
        return x


class SiglipEncoder(nn.Module):

    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(self.num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SiglipVisionTransformer(nn.Module):

    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:

        x = self.embeddings(pixel_values)
        x = self.encoder(x)
        x = self.post_layernorm(x)
        return x


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values):
        # pixel_values:(batch_size, num_channels, image_size, image_size) => (batch_size, num_patches, embed_dim)
        return self.vision_model(pixel_values=pixel_values)
