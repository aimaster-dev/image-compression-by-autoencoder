import os


def get_model_name(resnet_model_name: str, quantize_levels: int, is_encoder: bool):
    part_name = "encoder" if is_encoder else "decoder"
    return f"{resnet_model_name}_B={quantize_levels}_{part_name}.pth"


def get_model_path(models_dir: str, resnet_model_name, quantize_levels: int, is_encoder: bool):
    model_name = get_model_name(resnet_model_name, quantize_levels, is_encoder)
    return os.path.join(models_dir, model_name)


__all__ = [
    "get_model_path"
]
