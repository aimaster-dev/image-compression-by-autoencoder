import code.fs as fs
import os
import shutil
from code.vision import AutoEncoder
from code.vision.dataset import ImageDataset
from code.vision.utils import vgg_loss

import matplotlib.pyplot as plt
import torch
import torch.utils.data as td
import torchvision as tv
import tqdm


class AutoEncoderTrainer:
    def __init__(
            self,
            root: str,
            test_root: str,
            resnet_model_name: str,
            qb: int,
            epochs: int,
            batch_size: int,
            lr: float,
            device: torch.device,
            save_results_every: int = 10,
            save_models_dir: str = "models",
            use_checkpoint: bool = False,
    ):
        self.root = root
        self.test_root = test_root
        self.resnet_model_name = resnet_model_name
        self.qb = qb
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.save_results_every = save_results_every
        self.save_models_dir = save_models_dir

        self.model = AutoEncoder(
            resnet_model_name=resnet_model_name,
            qb=qb
        ).to(device).train()

        if use_checkpoint:
            encoder_path = fs.get_model_path(self.save_models_dir, self.resnet_model_name, self.qb, is_encoder=True)
            decoder_path = fs.get_model_path(self.save_models_dir, self.resnet_model_name, self.qb, is_encoder=False)

            self.model.encoder = torch.load(encoder_path).to(device).train()
            self.model.decoder = torch.load(decoder_path).to(device).train()

    def train(self):
        mse = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        dataset = ImageDataset(
            self.root,
            transform=tv.transforms.Compose([
                tv.transforms.Resize((512, 512)),
                tv.transforms.ToTensor(),
            ]))

        test_images = ImageDataset(
            self.test_root,
            transform=tv.transforms.Compose([
                tv.transforms.Resize((512, 512)),
                tv.transforms.ToTensor(),
            ])
        )

        loader = td.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = td.DataLoader(test_images, batch_size=self.batch_size, shuffle=False)

        # Clear train_logs directory
        shutil.rmtree("train_logs", ignore_errors=True)
        os.makedirs("train_logs", exist_ok=True)

        for epoch in range(self.epochs):
            bar = tqdm.tqdm(loader, total=len(loader), desc=f"Epoch {epoch + 1}/{self.epochs}")
            mse_loss = None
            for batch_num, x in enumerate(bar):
                self.model.train()
                x = x.to(self.device)
                x_hat = self.model(x)

                loss = mse(x_hat, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if mse_loss is None:
                    mse_loss = loss.item()
                else:
                    mse_loss = 0.9 * mse_loss + 0.1 * loss.item()

                bar.set_postfix(loss=mse_loss)

                if batch_num % self.save_results_every == 0:
                    self.model.eval()
                    first_batch = next(iter(test_loader))
                    with torch.no_grad():
                        first_batch = first_batch.to(self.device)
                        x_hat = self.model(first_batch)

                    limit = min(len(first_batch), 8)
                    fig, axes = plt.subplots(2, limit, figsize=(limit * 2, 4))
                    for i in range(limit):
                        axes[0, i].axis("off")
                        axes[1, i].axis("off")
                        axes[0, i].imshow(first_batch[i].cpu().permute(1, 2, 0))
                        axes[1, i].imshow(torch.clamp(x_hat[i].cpu().permute(1, 2, 0), 0, 1).numpy())

                    plt.savefig(f"train_logs/epoch={epoch}_batch={batch_num}.png")

            self.save_checkpoint()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    def save_checkpoint(self):
        os.makedirs(self.save_models_dir, exist_ok=True)
        torch.save(
            self.model.encoder,
            fs.get_model_path(self.save_models_dir, self.resnet_model_name, self.qb, is_encoder=True)
        )
        torch.save(
            self.model.decoder,
            fs.get_model_path(self.save_models_dir, self.resnet_model_name, self.qb, is_encoder=False)
        )


__all__ = [
    "AutoEncoderTrainer"
]
