import pandas as pd
import os

from loader import PopulationDataset

def print_metadata_csv_info(dataset_dir):
    splits = ["train", "val", "test"]

    print("Samples per split in PopulationDataset")
    for split in splits:
        dataset = PopulationDataset(dataset_dir, split)
        print(f"{split}: {len(dataset)}")