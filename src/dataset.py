import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


COLOR_TO_CLASS = {
    (60, 16, 152): 0,     # Building
    (132, 41, 246): 1,    # Land
    (110, 193, 228): 2,   # Road
    (254, 221, 58): 3,    # Vegetation
    (226, 169, 41): 4,    # Water
    (155, 155, 155): 5,   # Unlabeled
}

IGNORE_INDEX = 255
NUM_CLASSES = 5
CLASS_NAMES = ["Building", "Land", "Road", "Vegetation", "Water"]


def rgb_mask_to_class(mask_rgb, ignore_index=IGNORE_INDEX):
    mask_np = np.array(mask_rgb)
    h, w, _ = mask_np.shape

    class_mask = np.full((h, w), ignore_index, dtype=np.int64)

    for color, class_id in COLOR_TO_CLASS.items():
        if class_id == 5:
            class_id = ignore_index

        matches = np.all(mask_np == color, axis=-1)
        class_mask[matches] = class_id

    return class_mask


class AerialSegmentationDataset(Dataset):
    def __init__(self, csv_path, image_size=(256, 256), augment=False):
        self.csv_path = Path(csv_path)
        self.image_size = image_size
        self.augment = augment
        self.samples = []
        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        self.mask_resize = transforms.Resize(
            self.image_size,
            interpolation=transforms.InterpolationMode.NEAREST,
        )

        with self.csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"]).convert("RGB")

        if self.augment:
            if torch.rand(1).item() < 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            if torch.rand(1).item() < 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)

        image = self.image_transform(image)
        mask = self.mask_resize(mask)

        mask = rgb_mask_to_class(mask)
        mask = torch.from_numpy(mask).long()

        return image, mask
