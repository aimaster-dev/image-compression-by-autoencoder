import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, resnet_type: str, latent_dim: int, quantize_levels: int):
        super().__init__()
        self.resnet_type = resnet_type
        self.latent_dim = latent_dim
        self.quantize_levels = quantize_levels

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
