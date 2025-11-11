import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import random

class CelebAInpaintingDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load image
        img_path = os.path.join(self.img_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        # pick a random mask file
        mask_name = random.choice(self.masks)
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            img = self.transform(img)  # [3, H, W] in range [0, 1]
            mask = self.transform(mask)  # [1, H, W] in range [0, 1]

        # Ensure mask is normalized to [0, 1] if it's not already
        if mask.max() > 1.0:
            mask = mask / 255.0

        # masked image = image × (1-mask)
        # mask will broadcast from [1, H, W] to [3, H, W]
        masked_img = img * (1 - mask)

        return masked_img, mask, img   # return everything

