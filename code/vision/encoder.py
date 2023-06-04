from code.vision import utils

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, resnet_model_name: str, quantize_levels: int):
        super().__init__()
        self.resnet_model_name = resnet_model_name
        self.quantize_levels = quantize_levels

        config = utils.resnet_model_config(resnet_model_name)
        resnet = config.model(pretrained=True)

        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        self.reducer = nn.AvgPool2d(kernel_size=2)

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        x = F.sigmoid(x)
        gaussian_noise = torch.rand_like(x) * 0.5 - 0.5
        noise = gaussian_noise * 2 ** -self.quantize_levels
        x = x + noise
        x = torch.log(x / (1 - x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.reducer(x)

        if self.training:
            x = self.add_noise(x)

        return x


__all__ = ["Encoder"]
