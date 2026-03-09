import os
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import global_land_mask

import data_fetch_lib


def generate_dataset(sample_count, output_dir):
    np.random.seed(42)
    os.makedirs(output_dir, exist_ok=True)

    try:
        records = pd.read_csv(os.path.join(output_dir, "metadata.csv")).to_dict(
            orient="records"
        )
        print("Appending to existing metadata.csv")
    except:
        records = []
        print("Will create new metadata.csv")

    accepted = 0

    with tqdm(total=sample_count) as pbar:
        while accepted < sample_count:
            # Choose random location and size
            center_lat = random.uniform(-70, 70)
            center_lon = random.uniform(-180, 180)
            # ln_meter_width = random.uniform(math.log(400), math.log(40000))
            # meter_width = math.exp(ln_meter_width)
            meter_width = 1000

            # lookup population count
            pop_dict = data_fetch_lib.ghsl_get_population(
                center_lat, center_lon, meter_width
            )

            # skip empty oceans
            is_ocean = global_land_mask.is_ocean(center_lat, center_lon)
            if is_ocean and pop_dict["total_population"] == 0:
                continue

            # Higher chance of selecting dense regions,
            # to avoid a dataset full of empty regions.
            probability_of_using = (
                pop_dict["log1p_density"] / math.log1p(100_000) + 0.005
            )
            if np.random.rand() > probability_of_using:
                continue

            img = data_fetch_lib.download_satellite_image(
                center_lat, center_lon, meter_width
            )

            filename = f"{center_lat:.6f}_{center_lon:.6f}_{meter_width}m".replace(
                ".", "-"
            )
            filename = f"{filename}.jpg"
            img.save(os.path.join(output_dir, filename), quality=95)

            pop_dict["image_filename"] = filename
            records.append(pop_dict)

            accepted += 1
            pbar.update(1)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    print("Total samples:", len(df))


if __name__ == "__main__":
    generate_dataset(8000, output_dir="dataset")
