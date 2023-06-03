import os

import torch
import torch.utils.data as td
from PIL import Image
import torchvision as tv


class ImageDataset(td.Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)

        if transform is None:
            transform = tv.transforms.ToTensor()

        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)
        image = self.transform(image)
        return image


__all__ = ["ImageDataset"]
