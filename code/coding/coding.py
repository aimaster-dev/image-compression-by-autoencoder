import typing as tp

import numpy as np
import torch
import torch.nn.functional as F
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import SimpleAdaptiveModel


def quantize(vector: torch.Tensor, quantize_level: int) -> torch.Tensor:
    return (vector * (2 ** quantize_level) + 0.5).long()


def dequantize(vector: torch.Tensor, quantize_level: int) -> torch.Tensor:
    return vector.float() / (2 ** quantize_level)


def get_coder(quantize_level: int) -> AECompressor:
    keys = [key for key in range(0, 2 ** quantize_level + 1)]
    prob = 1.0 / len(keys)
    model = SimpleAdaptiveModel({k: prob for k in keys})
    coder = AECompressor(model)
    return coder


def compress(vector: torch.Tensor, quantize_level: int):
    """
    Compresses a vector of floats into a bitstring.
    :param vector: The vector to compress.
    :param quantize_level: The number of quantization levels to use.
    :return: The compressed bitstring.
    """

    shape = vector.shape
    vector = vector.flatten()
    vector = F.sigmoid(vector)
    vector = quantize(vector, quantize_level)
    vector = vector.tolist()

    coder = get_coder(quantize_level)

    return coder.compress(vector), list(shape)


def decompress(compressed: tp.List[int], shape: tp.List[int], quantize_levels) -> torch.Tensor:
    length = np.prod(shape)
    coder = get_coder(quantize_levels)
    decompressed = coder.decompress(compressed, length)
    decompressed = np.fromiter(map(int, decompressed), dtype=np.int64)
    decompressed = torch.from_numpy(decompressed).float()
    decompressed = dequantize(decompressed, quantize_levels)
    decompressed = decompressed.view(1, *shape)

    return decompressed


__all__ = ["compress", "decompress"]
