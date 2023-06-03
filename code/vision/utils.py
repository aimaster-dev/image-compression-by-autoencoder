import torch
import torch.nn as nn


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


__all__ = [
    "vgg_loss"
]
