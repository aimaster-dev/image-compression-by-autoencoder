import os
import shutil
from code.vision import Decoder, Encoder
from code.vision.dataset import ImageDataset
from code.vision.utils import vgg_loss
from itertools import chain

import matplotlib.pyplot as plt
import torch
import torch.utils.data as td
import torchvision as tv


class AutoEncoderTrainer:
    def __init__(
            self,
            root: str,
            resnet_model_name: str,
            quantize_levels: int,
            epochs: int,
            batch_size: int,
            lr: float,
            device: torch.device,
            vgg_alpha: float = 0.0,
            save_results_every: int = 10,
    ):
        self.root = root
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.vgg_alpha = vgg_alpha
        self.save_results_every = save_results_every
        self._vgg = None if vgg_alpha == 0.0 else tv.models.vgg16(pretrained=True).eval().to(device)

        self.encoder = Encoder(
            resnet_model_name=resnet_model_name, quantize_levels=quantize_levels
        ).to(device).train()

        self.decoder = Decoder(
            resnet_model_name=resnet_model_name
        ).to(device).train()

    def train(self):
        mse = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=self.lr
        )
        dataset = ImageDataset(
            self.root,
            transform=tv.transforms.ToTensor(),
        )
        loader = td.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Clear train_logs directory
        shutil.rmtree("train_logs", ignore_errors=True)
        os.makedirs("train_logs", exist_ok=True)

        for epoch in range(self.epochs):
            for x in loader:
                x = x.to(self.device)
                latent = self.encoder(x)
                x_hat = self.decoder(latent)

                loss = mse(x_hat, x)
                if self.vgg_alpha > 0:
                    loss += self.vgg_alpha * vgg_loss(self._vgg, x, x_hat)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % self.save_results_every == 0:
                first_batch = next(iter(loader))
                with torch.no_grad():
                    first_batch = first_batch.to(self.device)
                    latent = self.encoder(first_batch)
                    x_hat = self.decoder(latent)

                limit = min(len(first_batch), 8)
                fig, axes = plt.subplots(2, limit, figsize=(limit * 2, 4))
                for i in range(limit):
                    axes[0, i].axis("off")
                    axes[1, i].axis("off")
                    axes[0, i].imshow(first_batch[i].cpu().permute(1, 2, 0))
                    axes[1, i].imshow(x_hat[i].cpu().permute(1, 2, 0))

                plt.savefig(f"train_logs/epoch_{epoch}.png")

            print(f"Epoch: {epoch}, Loss: {loss.item()}")

        return self.encoder, self.decoder


__all__ = [
    "AutoEncoderTrainer"
]
