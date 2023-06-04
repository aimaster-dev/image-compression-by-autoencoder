import torch
import torch.nn as nn

from code.vision import utils


class TransposedBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int = 1,
            output_padding: int = 1,
            activation=nn.LeakyReLU(0.05)
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


class Decoder(nn.Module):
    def __init__(self, resnet_model_name: str):
        super().__init__()
        self.resnet_model_name = resnet_model_name
        self.channels = utils.resnet_model_config(resnet_model_name).channels

        self.scale = TransposedBlock(self.channels[-1], self.channels[-1], kernel_size=3, stride=2)

        self.decoder = nn.ModuleList([])
        for i in range(len(self.channels) - 1, 0, -1):
            block = TransposedBlock(
                self.channels[i], self.channels[i - 1], kernel_size=3, stride=2
            )
            self.decoder.append(block)

        self.final = nn.Sequential(
            TransposedBlock(self.channels[0], 32, kernel_size=3, stride=2, output_padding=0),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.scale(x)
        for i, layer in enumerate(self.decoder):
            x = layer(x)

        x = self.final(x)

        return x


__all__ = ["Decoder"]