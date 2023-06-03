import argparse
import os
from code.trainer import AutoEncoderTrainer
from code.vision import Decoder, Encoder
import code.fs as fs

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AutoEncoder")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--resnet-type", type=str, default="resnet34",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vgg_alpha", type=float, default=0.0)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--quantize_levels", type=int, default=4)
    args = parser.parse_args()

    encoder = Encoder(resnet_type=args.resnet_type, latent_dim=args.latent_dim, quantize_levels=args.quantize_levels)
    decoder = Decoder(resnet_type=args.resnet_type, latent_dim=args.latent_dim)
    trainer = AutoEncoderTrainer(
        root=args.root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        encoder=encoder,
        decoder=decoder,
        device=args.device,
        vgg_alpha=args.vgg_alpha,
    )

    trainer.train()

    models_dir = fs.get_models_dir(args.models_dir, args.latent_dim, args.quantize_levels)
    os.makedirs(models_dir, exist_ok=True)
    torch.save(encoder, fs.get_model_path(args.models_dir, args.latent_dim, args.quantize_levels, "encoder.pth"))
    torch.save(decoder, fs.get_model_path(args.models_dir, args.latent_dim, args.quantize_levels, "decoder.pth"))
