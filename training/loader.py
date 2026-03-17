import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

def assign_split(center_lat: float, center_lon: float) -> str:
    """
    Based on longitude, assigns to "train", "val", or "test".
    """
    if center_lon < 30:
        return "train"
    elif center_lon < 80:
        return "val"
    else:
        return "test"


class PopulationDataset(Dataset):
    """
    Loads data for my CSE 493G1 final project.
    """

    def __init__(self, dir, split):
        """
        Args:
            dir (str): Directory with all images.
            split (str): 'train', 'val', or 'test'.
        """
        self.data = pd.read_csv(os.path.join(dir, "metadata.csv"))
        
        # Assign split using coordinates
        # Note: ignores the outdated split column in the dataset
        self.data["computed_split"] = self.data.apply(
            lambda r: assign_split(r["center_lat"], r["center_lon"]),
            axis=1
        )
        self.data = self.data[self.data["computed_split"] == split].reset_index(drop=True)

        self.dir = dir

        self.target_col = "log1p_density"

        if split == "train":
            # Prevent overfitting using random data augmentation
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225]
                ),
            ])
        else:
            # Use ImageNet normalization
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225]
                ),
            ])
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.dir, row["image_filename"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        target = torch.tensor(row[self.target_col], dtype=torch.float32)

        return image, target
    