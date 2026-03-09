import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

"""
Oops, I downloaded a bunch of imagery from over the ocean, and Google
just returns a white no imagery placeholder image for these.

This script is used to remove those from the dataset.
"""


def is_no_imagery_placeholder(image_path, threshold_mean=220):
    """
    Detects if an image is likely a 'No Imagery' placeholder.
    """
    with Image.open(image_path) as img:
        gray = img.convert("L")
        data = np.array(gray)
        mean_brightness = np.mean(data)
        return mean_brightness > threshold_mean


def cleanup_dataset(dataset_dir):
    csv_path = os.path.join(dataset_dir, "metadata.csv")
    df = pd.read_csv(csv_path)
    initial_count = len(df)

    indices_to_drop = []
    deleted_files = 0

    for index, row in tqdm(df.iterrows(), total=initial_count):
        img_path = os.path.join(dataset_dir, row["image_filename"])

        if os.path.exists(img_path):
            if is_no_imagery_placeholder(img_path):
                # Delete the physical file
                os.remove(img_path)
                indices_to_drop.append(index)
                deleted_files += 1
        else:
            # If the file is already missing, remove it from CSV anyway
            indices_to_drop.append(index)

    # Update the CSV
    df_cleaned = df.drop(indices_to_drop)
    df_cleaned.to_csv(csv_path, index=False)

    print(f"Deleted {deleted_files} 'No Imagery' files.")
    print(f"Remaining records in CSV: {len(df_cleaned)} (out of {initial_count}).")
