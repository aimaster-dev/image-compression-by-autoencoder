import argparse
import os
from code.trainer import AutoEncoderTrainer
from code.vision import Decoder, Encoder

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AutoEncoder")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vgg_alpha", type=float, default=0.0)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--quantize_levels", type=int, default=4)
    args = parser.parse_args()

    encoder = Encoder((3, 32, 32), args.latent_dim, args.quantize_levels)
    decoder = Decoder(args.latent_dim, (3, 32, 32))
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

    save_dir = os.path.join(
        args.save_dir,
        f"latent_dim_{args.latent_dim}",
        f"quantize_levels_{args.quantize_levels}"
    )
    os.makedirs(save_dir, exist_ok=True)
    torch.save(encoder, os.path.join(save_dir, "encoder.pth"))
    torch.save(decoder, os.path.join(save_dir, "decoder.pth"))
