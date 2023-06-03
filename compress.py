import argparse
from code import fs
from PIL import Image

import torch
import torchvision as tv

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
    image = transform(image)
    print(image.size)

    encoder = torch.load(encoder_path, map_location=args.device).eval()
    image = image.unsqueeze(0)
    image = encoder(image)
    image = image.squeeze(0)
    image = image.permute(1, 2, 0)
    image = image.detach().numpy()
    image = Image.fromarray(image)
    image.save("compressed.png")
