import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class Data_Loader(Dataset):
    def __init__(self, train_path, hr_transform=None, lr_transform=None):
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform
        self.train_path = train_path

    def __getitem__(self, index):
        image_path = self.train_path[index]
        hr_image = Image.open(image_path).convert("RGB")
        lr_image = Image.open(image_path).convert("RGB")
        if self.hr_transform is not None:
            hr_image = self.hr_transform(hr_image)
        if self.lr_transform is not None:
            lr_image = self.lr_transform(lr_image)
        return {"image":lr_image,"label":hr_image}

    def __len__(self):
        hr_len = len(self.train_path)
        return hr_len