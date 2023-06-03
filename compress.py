import argparse
import os

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress Image")
    parser.add_argument("--image", type=str, default="data")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--quantize_levels", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    encoder_path = os.path.join(
        args.models_dir,
        f"latent_dim={args.latent_dim}",
        f"quantize_levels={args.quantize_levels}",
        "encoder.pth"
    )

    encoder = torch.load(encoder_path, map_location="cpu").eval().to(args.device)

    image = torch.load(args.image).to(args.device)
