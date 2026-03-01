import math
from pathlib import Path
from rasterio import windows
import rasterio
import requests
import numpy as np
from rasterio import DatasetReader
from PIL import Image
from io import BytesIO

# Source: GHSL (Global Human Settlement Layer)
# Product: GHS-POP
# Link: https://human-settlement.emergency.copernicus.eu/download.php?ds=pop
# Epoch: 2020
# Resolution: 3 arcsec
# Coordinate system: WGS84
# Extract zip to directory: `ghsl_data`
GHSL_TIF = "ghsl_data/GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0.tif"
OUTPUT_DIR = "dataset"
GOOGLE_STATIC_MAPS_KEY_PATH = "google_static_maps_key.txt"


google_static_maps_key = Path(GOOGLE_STATIC_MAPS_KEY_PATH).read_text().strip()
ghsl_data: DatasetReader = rasterio.open(GHSL_TIF)


def ghsl_get_population(
    center_lat: float, center_lon: float, meter_width: float
) -> dict:
    """
    Returns the population metrics in this square
    """

    def meters_to_latlon_delta(lat: float, meter_length: float) -> tuple[float, float]:
        """
        Converts a meter length to longitude and latitude lengths.
        """
        dlat = meter_length / 111320.0
        dlon = meter_length / (111320.0 * math.cos(math.radians(lat)))
        return dlat, dlon

    def assign_split(center_lat: float, center_lon: float) -> str:
        """
        Based on longitude, assigns to "train", "val", or "test".
        """
        if center_lon < -25:
            return "train"
        elif center_lon < 55:
            return "val"
        else:
            return "test"

    # Get bounds
    dlat, dlon = meters_to_latlon_delta(center_lat, meter_width)
    min_lat = center_lat - dlat / 2
    max_lat = center_lat + dlat / 2
    min_lon = center_lon - dlon / 2
    max_lon = center_lon + dlon / 2

    # Read population density
    window = windows.from_bounds(
        min_lon, min_lat, max_lon, max_lat, transform=ghsl_data.transform
    )
    pop_data = ghsl_data.read(1, window=window)

    area_km2 = meter_width * meter_width / 1e6
    total_population = np.sum(pop_data)
    people_per_km2 = total_population / area_km2
    log1p_density = math.log1p(people_per_km2)
    split = assign_split(center_lat, center_lon)

    return {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "meter_width": meter_width,
        "area_km2": area_km2,
        "total_population": total_population,
        "people_per_km2": people_per_km2,
        "log1p_density": log1p_density,
        "split": split,
    }


def download_satellite_image(
    center_lat: float,
    center_lon: float,
    meter_width: float,
) -> Image.Image:
    """
    Returns a 224 by 224 pixel satellite image with `width_meters`.
    """

    def get_download_meter_width(zoom: int, lat: float, px_width: int) -> float:
        """
        Returns the width in meters that an image downloaded from Google static maps
        with these parameters would have.
        """
        # Formula from: https://groups.google.com/g/google-maps-js-api-v3/c/hDRO4oHVSeM
        meters_per_pixel = 156543.03392 * math.cos(math.radians(lat)) / (2**zoom)
        return meters_per_pixel * px_width

    def center_crop_and_resize(
        img: Image.Image, scale_factor: float, out_px_width: int
    ) -> Image.Image:
        """
        Center crops a square image by scale_factor, and rescales the result so its
        a square with `out_px_width`
        """

        w, h = img.size
        if w != h:
            raise RuntimeError("can only center crop a square image")

        crop_size = int(w * scale_factor)

        left = (w - crop_size) // 2
        right = (w + crop_size) // 2

        cropped = img.crop((left, left, right, right))

        resized = cropped.resize((out_px_width, out_px_width), Image.Resampling.BICUBIC)

        return resized

    DOWNLOAD_PX_WIDTH = 512

    # Keep zooming out until our meter width is greater
    # than our target.
    zoom = 21
    while True:
        gotten_width_meters = get_download_meter_width(
            zoom, center_lat, DOWNLOAD_PX_WIDTH
        )
        if gotten_width_meters > meter_width:
            break

        if zoom == 0:
            raise RuntimeError("Requested image larger than the world")
        zoom -= 1

    # Download image
    url = (
        "https://maps.googleapis.com/maps/api/staticmap"
        f"?center={center_lat},{center_lon}"
        f"&zoom={zoom}"
        f"&size={DOWNLOAD_PX_WIDTH}x{DOWNLOAD_PX_WIDTH}"
        "&maptype=satellite"
        f"&key={google_static_maps_key}"
    )

    r = requests.get(url, timeout=10)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")

    # Crop the image so it matches the requested meter width
    rescale_factor = meter_width / gotten_width_meters
    center_crop = center_crop_and_resize(img, rescale_factor, 224)

    return center_crop
