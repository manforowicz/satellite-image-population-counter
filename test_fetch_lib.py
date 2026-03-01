import os
import numpy as np
import pandas as pd
import data_fetch_lib

"""
A few test samples around Anchorage, because
Anchorage is at a non-central longitude and latitude.

First image is ocean, and second image moves east to downtown.

Third image is ocean, and fourth image moves south to suburbs.
"""

OUTPUT_DIR = "test_dataset"

if __name__ == "__main__":
    np.random.seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    records = []

    coords = [
        (61.217, -149.93027),
        (61.217, -149.89819),
        (61.21229, -149.9455),
        (61.19627, -149.9455),
    ]

    for center_lat, center_lon in coords:
        # Choose random location and size
        meter_width = 1000

        pop_dict = data_fetch_lib.ghsl_get_population(
            center_lat, center_lon, meter_width
        )

        img = data_fetch_lib.download_satellite_image(
            center_lat, center_lon, meter_width
        )

        filename = f"{center_lat:.6f}_{center_lon:.6f}_{meter_width}m".replace(".", "-")
        filename = f"{filename}.jpg"
        img.save(os.path.join(OUTPUT_DIR, filename), quality=95)

        pop_dict["image_filename"] = filename
        records.append(pop_dict)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

    print("Total samples:", len(df))
