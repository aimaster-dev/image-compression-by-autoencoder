import argparse
import pickle
from code import fs
from code.coding import compress

import numpy as np
import torch
import torchvision as tv
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress Image")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--resnet-model", type=str, default="resnet18")
    parser.add_argument("--quantize_levels", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("Starting compression with the following parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    print()

    encoder_path = fs.get_model_path(args.models_dir, args.resnet_model, args.quantize_levels, is_encoder=True)
    transform = tv.transforms.ToTensor()

    image = Image.open(args.image)
    image = np.asarray(image, dtype=np.uint8)
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    image = torch.FloatTensor(image).unsqueeze(0).to(args.device)

    encoder = torch.load(encoder_path, map_location=args.device).eval()

    with torch.no_grad():
        image = encoder(image)

    image = image.squeeze(0).cpu()
    compressed, shape = compress(image, args.quantize_levels)
    fs.save_compressed(compressed, shape, args.output)
