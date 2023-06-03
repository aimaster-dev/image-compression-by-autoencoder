import argparse
import os
from code.vision import Encoder

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress Image")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--quantize_levels", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    encoder = Encoder((3, 32, 32), args.latent_dim, args.quantize_levels)
    encoder_path = os.path.join(
        args.model_dir,
        f"latent_dim={args.latent_dim}",
        f"quantize_levels={args.quantize_levels}",
        "encoder.pth"
    )

    encoder.load_state_dict(torch.load(encoder_path, map_location=args.device)).eval()

    data = torch.load(args.root)
    for x, _ in data:
        x = x.to(args.device)
        latent = encoder(x)
        print(latent.shape)
        break
