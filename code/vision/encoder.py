import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size: tp.Tuple[int, int, int], latent_dim: int, quantize_levels: int):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.quantize_levels = quantize_levels

        self.resnet_encoder = nn.Linear(input_size[0] * input_size[1] * input_size[2], latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.resnet_encoder(x)

        if self.training:
            latent = F.sigmoid(latent)
            gaussian_noise = torch.rand_like(latent) * 0.5 - 0.5
            noise = gaussian_noise * 2 ** -self.quantize_levels
            latent = latent + noise
            latent = torch.log(latent / (1 - latent))

        return latent


__all__ = ["Encoder"]
