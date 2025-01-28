import torch
from torch import nn
from paligemma import PaliGemmaConfig


################################### Projector ###################################


class MultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        # The projection dimension is the embedding dimension of the Gemma model
        self.linear = nn.Linear(config.vision_config.hidden_size, config.projection_dim)

    def forward(self, image_features):
        # (batch_size, num_patches, hidden_size) -> (batch_size, num_patches, projection_dim)
        return self.linear(image_features)
