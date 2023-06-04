import pickle
import typing as tp


class CompressedImage:
    def __init__(
            self,
            quantize_levels: int,
            latent: tp.List[int],
            probas: tp.Dict[int, float],
            length: int,
            shape: tuple
    ):
        self.quantize_levels = quantize_levels
        self.latent = latent
        self.probas = probas
        self.length = length
        self.shape = shape

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)


__all__ = [
    "CompressedImage"
]
