import os
import pickle
import typing as tp


def get_model_name(resnet_model_name: str, quantize_levels: int, is_encoder: bool):
    part_name = "encoder" if is_encoder else "decoder"
    return f"{resnet_model_name}_B={quantize_levels}_{part_name}.pth"


def get_model_path(models_dir: str, resnet_model_name, quantize_levels: int, is_encoder: bool):
    model_name = get_model_name(resnet_model_name, quantize_levels, is_encoder)
    return os.path.join(models_dir, model_name)


def list_to_int(l: tp.List[int]) -> int:
    return int("1" + "".join(map(str, l)), 2)


def shape_to_str(shape: tp.List[int]) -> str:
    return "_".join(map(str, shape))


def save_compressed(compressed: tp.List[int], shape: tp.List[int], file_name: str):
    with open(file_name, "wb") as f:
        pickle.dump(f"{shape_to_str(shape)},{list_to_int(compressed)}", f)


def load_compressed(file_name) -> tp.Tuple[tp.List[int], tp.List[int]]:
    with open(file_name, "rb") as f:
        result = pickle.load(f)
    result = result.split(",")
    shape = list(map(int, result[0].split("_")))
    compressed = list(map(int, bin(int(result[1]))[3:]))
    return compressed, shape


__all__ = [
    "get_model_path"
]
