import torch
import torch.nn as nn
import torch.nn.functional as F

from code.vision import utils


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, activation=F.relu):
        super().__init__()

        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class ResNetUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks: int, stride: int):
        super().__init__()

        self.upsamples = nn.ModuleList([])
        for i in range(blocks):
            if i == blocks - 1:
                self.upsamples.append(Upsample(in_channels, out_channels, stride))
            else:
                self.upsamples.append(Upsample(in_channels, in_channels, stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.upsamples:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, resnet_model_name: str):
        super().__init__()

        config = utils.resnet_model_config(resnet_model_name)
        self.channels = config.channels

        self.decoder = nn.ModuleList([])
        for i in range(len(self.channels) - 1, 0, -1):
            self.decoder.append(
                ResNetUpsample(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i - 1],
                    blocks=config.blocks[i],
                    stride=config.strides[i]
                )
            )

        self.downsample = nn.Conv2d(
            self.channels[0], 3, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.decoder):
            x = layer(x)

        x = self.downsample(x)
        x = F.sigmoid(x)

        return x


__all__ = ["Decoder"]
