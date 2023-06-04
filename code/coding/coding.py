import typing as tp
from code.coding.compressed_image import CompressedImage
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import StaticModel


def compress(vector: torch.Tensor, quantize_level: int) -> CompressedImage:
    """
    Compresses a vector of floats into a bitstring.
    :param vector: The vector to compress.
    :param quantize_level: The number of quantization levels to use.
    :return: The compressed bitstring.
    """

    shape = vector.shape
    vector = vector.flatten()
    vector = vector = F.sigmoid(vector)
    vector = vector * 2 ** quantize_level + 0.5
    vector = vector.long()
    vector = vector.tolist()

    counter = Counter(vector)
    probas = {k: v / len(vector) for k, v in counter.items()}
    model = StaticModel(probas)

    coder = AECompressor(model)
    compressed = coder.compress(vector)

    return CompressedImage(
        quantize_level,
        compressed,
        probas,
        len(vector),
        shape
    )


def decompress(image: CompressedImage) -> torch.Tensor:
    model = StaticModel(image.probas)
    coder = AECompressor(model)
    decompressed = coder.decompress(image.latent, image.length)
    decompressed = np.fromiter(map(int, decompressed), dtype=np.int64)
    decompressed = torch.from_numpy(decompressed).float()
    decompressed = decompressed / 2 ** image.quantize_levels
    decompressed = torch.log(decompressed / (1 - decompressed))
    decompressed = decompressed.view(1, *image.shape)

    return decompressed


__all__ = ["compress", "decompress"]
