import code.vision.utils as vision_utils
from code.vision import Decoder, Encoder

from itertools import chain

import torch
import torch.utils.data as td
import torchvision as tv


class AutoEncoderTrainer:
    def __init__(
            self,
            root: str,
            epochs: int,
            batch_size: int,
            lr: float,
            encoder: Encoder,
            decoder: Decoder,
            device: torch.device,
            vgg_alpha: float = 0.0,
    ):
        self.root = root
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device
        self.vgg_alpha = vgg_alpha
        self._vgg = None if vgg_alpha == 0.0 else tv.models.vgg19(pretrained=True).to(device)

    def train(self):
        mse = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=self.lr
        )
        dataset = tv.datasets.DatasetFolder(
            self.root,
            tv.transforms.ToTensor(),
        )
        loader = td.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        for epoch in range(self.epochs):
            for x, _ in loader:
                x = x.to(self.device)
                latent = self.encoder(x)
                x_hat = self.decoder(latent)

                loss = mse(x_hat, x)
                if self.vgg_alpha > 0:
                    loss += self.vgg_alpha * vision_utils.vgg_loss(self._vgg, x, x_hat)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch: {epoch}, Loss: {loss.item()}")


__all__ = [
    "AutoEncoderTrainer"
]
