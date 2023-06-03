import torch
import torch.nn as nn
import torchvision.models.resnet as resnet_models

import dataclasses
import typing as tp


def vgg_loss(vgg: nn.Module, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    VGG loss

    :param vgg: (nn.Module)
    :param x: (torch.Tensor) input
    :param x_hat: (torch.Tensor) output
    :return: (torch.Tensor) loss
    """
    x_latent = vgg(x)
    x_hat_latent = vgg(x_hat)
    return torch.mean(torch.abs(x_latent - x_hat_latent))


@dataclasses.dataclass
class ResNetModelConfiguration:
    model: tp.Callable[..., nn.Module]
    channels: tp.List[int]
    blocks: tp.List[int]
    strides: tp.List[int]


RESNET_MODELS = {
    "resnet18": ResNetModelConfiguration(
        model=resnet_models.resnet18,
        channels=[64, 128, 256, 512],
        blocks=[2, 2, 2, 2],
        strides=[1, 2, 2, 2]
    ),
    "resnet34": ResNetModelConfiguration(
        model=resnet_models.resnet34,
        channels=[64, 128, 256, 512],
        blocks=[3, 4, 6, 3],
        strides=[1, 2, 2, 2]
    ),
    "resnet50": ResNetModelConfiguration(
        model=resnet_models.resnet50,
        channels=[256, 512, 1024, 2048],
        blocks=[3, 4, 6, 3],
        strides=[1, 2, 2, 2]
    ),
    "resnet101": ResNetModelConfiguration(
        model=resnet_models.resnet101,
        channels=[256, 512, 1024, 2048],
        blocks=[3, 4, 23, 3],
        strides=[1, 2, 2, 2]
    ),
    "resnet152": ResNetModelConfiguration(
        model=resnet_models.resnet152,
        channels=[256, 512, 1024, 2048],
        blocks=[3, 8, 36, 3],
        strides=[1, 2, 2, 2]
    )
}


def resnet_model_config(model_name: str) -> ResNetModelConfiguration:
    """
    Get resnet model configuration

    :param model_name: (str) model name
    :return: (ResNetModelConfiguration)
    """
    return RESNET_MODELS[model_name]


__all__ = [
    "vgg_loss",
    "resnet_model_config"
]
