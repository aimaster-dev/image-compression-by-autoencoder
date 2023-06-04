import argparse
import code.fs as fs
from code.coding import CompressedImage, decompress

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompress Image")
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--quantize_levels", type=int, required=True)
    parser.add_argument("--resnet-model", type=str, default="resnet18")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("Starting decompression with the following parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    print()

    decoder_path = fs.get_model_path(args.models_dir, args.resnet_model, args.quantize_levels, is_encoder=False)
    decoder = torch.load(decoder_path, map_location=args.device).eval()

    image = CompressedImage.load(args.file)
    latent = decompress(image)

    with torch.no_grad():
        image = decoder(latent)

    print(image.shape)
