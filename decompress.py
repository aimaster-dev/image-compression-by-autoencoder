import argparse
import code.fs as fs
from code.coding import decompress

import numpy as np
import torch
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompress Image")
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--qb", type=int, required=True)
    parser.add_argument("--resnet-model", type=str, default="resnet18")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("Starting decompression with the following parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    print()

    vector, shape = fs.load_compressed(args.file)

    vector = decompress(vector, shape, args.quantize_levels)

    decoder_path = fs.get_model_path(args.models_dir, args.resnet_model, args.qb, is_encoder=False)
    decoder = torch.load(decoder_path, map_location="cpu").to(args.device).eval()

    with torch.no_grad():
        image = decoder(vector)

    image = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    image = image * 255.0
    image = image.astype(np.uint8)

    image = Image.fromarray(image)
    image.save(args.output)
