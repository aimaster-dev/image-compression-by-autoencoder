import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, resnet_type: str, latent_dim: int):
        """
        Decoder for vision models

        :param resnet_type: (str)
        :param latent_dim: (int)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x).view(-1, *self.output)


__all__ = ["Decoder"]
