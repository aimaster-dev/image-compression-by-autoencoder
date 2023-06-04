from code.vision import utils

import torch
import torch.nn as nn


class ConvNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_upsample=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.05)

        if use_upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        else:
            self.upsample = nn.Identity()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class ResidualUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvNormRelu(in_channels, in_channels)
        if stride != 1:
            self.conv2 = ConvNormRelu(in_channels, out_channels, use_upsample=True)
            self.residual = ConvNormRelu(in_channels, out_channels, use_upsample=True)
        else:
            self.conv2 = ConvNormRelu(out_channels, out_channels)
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        res = self.residual(x)
        return out + res


class Decoder(nn.Module):
    def __init__(self, resnet_model_name: str):
        super().__init__()
        config = utils.resnet_model_config(resnet_model_name)
        self.inchannel = config.channels[-1]

        self.up = nn.Upsample(scale_factor=2)
        self.layer1 = self.make_layer(out_channels=256, num_blocks=2, stride=2)
        self.layer2 = self.make_layer(out_channels=128, num_blocks=2, stride=2)
        self.layer3 = self.make_layer(out_channels=64, num_blocks=2, stride=2)
        self.layer4 = self.make_layer(out_channels=64, num_blocks=2, stride=1)

        self.resize = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.05),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.sigmoid = nn.Sigmoid()

    def make_layer(self, out_channels: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualUpsampleBlock(self.inchannel, out_channels, stride))
            self.inchannel = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.resize(out)
        out = self.sigmoid(out)
        return out
