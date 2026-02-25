import os
import math
import random
import hashlib
import requests
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from pyproj import Geod, Transformer
from shapely.geometry import box
from tqdm import tqdm
from PIL import Image
from io import BytesIO

# -----------------------
# Configuration
# -----------------------

# Source: GHSL (Global Human Settlement Layer)
# Product: GHS-POP
# Link: https://human-settlement.emergency.copernicus.eu/download.php?ds=pop
# Epoch: 2020
# Resolution: 3 arcsec
# Coordinate system: WGS84
# Extract zip to directory: `ghsl_data`
GHSL_TIF = "ghsl_data/GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0.tif"
OUTPUT_DIR = "dataset"

IMAGE_SIZE = 224
SAMPLES = 5000

API_KEY = "YOUR_GOOGLE_STATIC_MAPS_KEY"

MIN_TILE_METERS = 200
MAX_TILE_METERS = 40000

BLOCK_SIZE_KM = 1000
MAX_LATITUDE = 70  # avoid polar distortion

os.makedirs(OUTPUT_DIR, exist_ok=True)

geod = Geod(ellps="WGS84")
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# -----------------------
# Helper Functions
# -----------------------


def meters_to_latlon_delta(lat: float, meters: float):
    dlat = meters / 111320.0
    dlon = meters / (111320.0 * math.cos(math.radians(lat)))
    return dlat, dlon


def compute_area_km2(min_lon, min_lat, max_lon, max_lat):
    poly = box(min_lon, min_lat, max_lon, max_lat)
    lons, lats = poly.exterior.coords.xy
    area, _ = geod.polygon_area_perimeter(lons, lats)
    return abs(area) / 1e6


def estimate_zoom(tile_meters, lat):
    """
    Choose zoom level minimizing error between
    desired meters_per_pixel and actual Google scale.
    """
    target_mpp = tile_meters / IMAGE_SIZE
    best_zoom = 0
    best_error = float("inf")

    for zoom in range(0, 21):
        mpp = 156543.03392 * math.cos(math.radians(lat)) / (2**zoom)
        error = abs(mpp - target_mpp)
        if error < best_error:
            best_error = error
            best_zoom = zoom

    return best_zoom


def download_satellite_image(center_lat, center_lon, zoom):
    url = (
        "https://maps.googleapis.com/maps/api/staticmap"
        f"?center={center_lat},{center_lon}"
        f"&zoom={zoom}"
        f"&size={IMAGE_SIZE}x{IMAGE_SIZE}"
        "&maptype=satellite"
        f"&key={API_KEY}"
    )

    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None

    try:
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def assign_split(lat, lon):
    x, y = transformer.transform(lon, lat)

    block_x = int(x // (BLOCK_SIZE_KM * 1000))
    block_y = int(y // (BLOCK_SIZE_KM * 1000))

    key = f"{block_x}_{block_y}".encode()
    h = int(hashlib.md5(key).hexdigest(), 16) % 100

    if h < 70:
        return "train"
    elif h < 85:
        return "val"
    else:
        return "test"


def deterministic_filename(lat, lon, tile_meters):
    key = f"{lat:.6f}_{lon:.6f}_{tile_meters:.2f}".encode()
    return hashlib.md5(key).hexdigest() + ".jpg"


# -----------------------
# Main Generation
# -----------------------

ghsl_data = rasterio.open(GHSL_TIF)
bounds = ghsl_data.bounds
nodata = ghsl_data.nodata

records = []
accepted = 0

print("Generating samples...")

with tqdm(total=SAMPLES) as pbar:
    while accepted < SAMPLES:

        lon = random.uniform(bounds.left, bounds.right)
        lat = random.uniform(bounds.bottom, bounds.top)

        if abs(lat) > MAX_LATITUDE:
            continue

        tile_meters = random.uniform(MIN_TILE_METERS, MAX_TILE_METERS)
        dlat, dlon = meters_to_latlon_delta(lat, tile_meters)

        min_lat = lat - dlat / 2
        max_lat = lat + dlat / 2
        min_lon = lon - dlon / 2
        max_lon = lon + dlon / 2

        try:
            window = from_bounds(
                min_lon, min_lat, max_lon, max_lat, transform=ghsl_data.transform
            )
            pop_data = ghsl_data.read(1, window=window)
        except Exception:
            continue

        if pop_data.size == 0:
            continue

        if nodata is not None:
            if np.mean(pop_data == nodata) > 0.5:
                continue

        total_population = float(np.sum(pop_data))

        area_km2 = compute_area_km2(min_lon, min_lat, max_lon, max_lat)
        if area_km2 <= 0:
            continue

        density = total_population / area_km2
        log_density = math.log1p(density)

        zoom = estimate_zoom(tile_meters, lat)
        img = download_satellite_image(lat, lon, zoom)

        if img is None:
            continue

        filename = deterministic_filename(lat, lon, tile_meters)
        img.save(os.path.join(OUTPUT_DIR, filename), quality=95)

        split = assign_split(lat, lon)

        records.append(
            {
                "image": filename,
                "center_lat": lat,
                "center_lon": lon,
                "tile_meters": tile_meters,
                "zoom": zoom,
                "area_km2": area_km2,
                "total_population": total_population,
                "avg_density": density,
                "log1p_density": log_density,
                "split": split,
            }
        )

        accepted += 1
        pbar.update(1)

df = pd.DataFrame(records)
df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

print("Total samples:", len(df))
