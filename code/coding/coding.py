import typing as tp

import numpy as np
import torch
import torch.nn.functional as F
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import SimpleAdaptiveModel


def quantize(vector: torch.Tensor, qb: int) -> torch.Tensor:
    return (vector * (2 ** qb) + 0.5).to(torch.int64)


def dequantize(vector: torch.Tensor, qb: int) -> torch.Tensor:
    return vector.float() / (2 ** qb)


def get_coder(qb: int) -> AECompressor:
    keys = [key for key in range(0, 2 ** qb + 1)]
    prob = 1.0 / len(keys)
    model = SimpleAdaptiveModel({k: prob for k in keys})
    coder = AECompressor(model)
    return coder


def compress(vector: torch.Tensor, qb: int):
    """
    Compresses a vector of floats into a bitstring.
    :param vector: The vector to compress.
    :param qb: The number of quantization levels to use.
    :return: The compressed bitstring.
    """

    shape = vector.shape
    vector = vector.flatten()
    vector = torch.clamp(vector, 0.0, 1.0)
    vector = quantize(vector, qb)
    vector = vector.tolist()

    coder = get_coder(qb)

    return coder.compress(vector), list(shape)


def decompress(compressed: tp.List[int], shape: tp.List[int], qb: int) -> torch.Tensor:
    length = np.prod(shape)
    coder = get_coder(qb)
    decompressed = coder.decompress(compressed, length)
    decompressed = np.fromiter(map(int, decompressed), dtype=np.int64)
    decompressed = torch.from_numpy(decompressed).float()
    decompressed = dequantize(decompressed, qb)
    decompressed = decompressed.view(1, *shape)

    return decompressed


__all__ = ["compress", "decompress"]
