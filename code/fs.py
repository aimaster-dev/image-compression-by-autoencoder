import os


def get_models_dir(models_dir: str, latent_dim: int, quantize_levels: int):
    return os.path.join(
        models_dir,
        f"latent_dim={latent_dim}",
        f"quantize_levels={quantize_levels}"
    )


def get_model_path(models_dir: str, latent_dim: int, quantize_levels: int, model_name: str):
    return os.path.join(
        get_models_dir(models_dir, latent_dim, quantize_levels),
        f"{model_name}.pth"
    )


__all__ = [
    "get_models_dir",
    "get_model_path"
]
