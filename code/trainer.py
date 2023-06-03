from code.vision import Decoder, Encoder
from code.vision.dataset import ImageDataset
from code.vision.utils import vgg_loss
from itertools import chain

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
        self._vgg = None if vgg_alpha == 0.0 else tv.models.vgg19(pretrained=True).to(device)
        self.encoder = Encoder(resnet_model_name=resnet_model_name, quantize_levels=quantize_levels).to(device)
        self.decoder = Decoder(resnet_model_name=resnet_model_name).to(device)
        self.save_results_every = save_results_every

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

        for epoch in range(self.epochs):
            for x, _ in loader:
                x = x.to(self.device)
                latent = self.encoder(x)
                x_hat = self.decoder(latent)

                loss = mse(x_hat, x)
                if self.vgg_alpha > 0:
                    loss += self.vgg_alpha * vgg_loss(self._vgg, x, x_hat)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

        return self.encoder, self.decoder


__all__ = [
    "AutoEncoderTrainer"
]
