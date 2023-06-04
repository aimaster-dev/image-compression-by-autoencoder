from code.vision import utils

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, resnet_model_name: str):
        super().__init__()
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

        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.pool(x)
        return x


__all__ = ["Encoder"]
