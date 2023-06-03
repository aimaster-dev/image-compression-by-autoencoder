import typing as tp

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output: tp.Tuple[int, int, int]):
        """
        Decoder for vision models

        :param latent_dim: (int)
        :param output: (C, H, W)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output = output

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output[0] * output[1] * output[2]),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x).view(-1, *self.output)


__all__ = ["Decoder"]
