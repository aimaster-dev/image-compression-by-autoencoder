import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder


class AutoEncoder(nn.Module):
    def __init__(self, resnet_model_name: str, qb: int):
        super().__init__()
        self.qb = qb
        self.encoder = Encoder(resnet_model_name=resnet_model_name)
        self.decoder = Decoder(resnet_model_name=resnet_model_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)

        x = torch.clamp(x, 0.0, 1.0)
        x = x + (1 / 2 ** self.qb) * (torch.rand_like(x) * 0.5 - 0.5)

        x = self.decoder(x)
        return x


__all__ = ["AutoEncoder"]
