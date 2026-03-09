import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


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
        self.data = self.data[self.data["split"] == split].reset_index(drop=True)
        self.dir = dir
        # Use ImageNet normalization
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.target_col = "log1p_density"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.dir, row["image_filename"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        target = torch.tensor(row[self.target_col], dtype=torch.float32)

        return image, target
    