import os
import math
import random
import requests
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from pyproj import Geod
from shapely.geometry import box
from tqdm import tqdm
from PIL import Image
from io import BytesIO

# -----
# Configuration
# -----

# Source: GHSL (Global Human Settlement Layer)
# Product: GHS-POP
# Link: https://human-settlement.emergency.copernicus.eu/download.php?ds=pop
# Epoch: 2020
# Resolution: 3 arcsec
# Coordinate system: WGS84
# Extract zip to directory: `ghsl_data`
GHSL_TIF = "ghsl_data/GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0.tif"

OUTPUT_DIR = "dataset"

# Each output image will be 224x224 pixels
IMAGE_SIZE = 224

SAMPLES = 5000

API_KEY = "YOUR_GOOGLE_STATIC_MAPS_KEY"

MIN_TILE_METERS = 200
MAX_TILE_METERS = 40000

BLOCK_SIZE_KM = 1000

# -----
# Helpers
# -----

def meters_to_latlon_delta(lat, meters):
    """Convert square size in meters to lat/lon degree deltas."""
    dlat = meters / 111320.0
    dlon = meters / (111320.0 * math.cos(math.radians(lat)))
    return dlat, dlon


def compute_area_m2(bounds):
    """Compute geodesic area of bounding box."""
    minx, miny, maxx, maxy = bounds
    poly = box(minx, miny, maxx, maxy)
    lons, lats = poly.exterior.coords.xy
    area, _ = geod.polygon_area_perimeter(lons, lats)
    return abs(area)


def download_satellite_image(center_lat, center_lon, zoom):
    """Download satellite tile (Google Static Maps example)."""
    url = (
        "https://maps.googleapis.com/maps/api/staticmap"
        f"?center={center_lat},{center_lon}"
        f"&zoom={zoom}"
        f"&size={IMAGE_SIZE}x{IMAGE_SIZE}"
        "&maptype=satellite"
        f"&key={API_KEY}"
    )
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return Image.open(BytesIO(r.content)).convert("RGB")


def estimate_zoom(tile_meters, lat):
    """
    Approximate zoom level so that 224px spans tile_meters.
    Uses Web Mercator scale approximation.
    """
    earth_circumference = 40075016.686
    meters_per_pixel_equator = earth_circumference / 256
    meters_per_pixel = meters_per_pixel_equator * math.cos(math.radians(lat))
    zoom = math.log2((meters_per_pixel * IMAGE_SIZE) / tile_meters)
    zoom = int(max(0, min(20, round(zoom))))
    return zoom


def assign_split(lat, lon):
    """
    Assign train/val/test based on 1000km grid blocks.
    """
    block_lat = int(lat / (BLOCK_SIZE_KM / 111))
    block_lon = int(lon / (BLOCK_SIZE_KM / 111))
    h = hash((block_lat, block_lon)) % 100
    if h < 70:
        return "train"
    elif h < 85:
        return "val"
    else:
        return "test"


# -----
# Main
# -----

os.makedirs(OUTPUT_DIR, exist_ok=True)

geod = Geod(ellps="WGS84")

ghsl_data: rasterio.DatasetReader = rasterio.open(GHSL_TIF)
bounds = ghsl_data.bounds

records = []

print("Generating samples...")

for i in tqdm(range(SAMPLES)):

    # Choose a random tile location
    lon = random.uniform(bounds.left, bounds.right)
    lat = random.uniform(bounds.bottom, bounds.top)

    # Choose a random tile size
    tile_meters = random.uniform(MIN_TILE_METERS, MAX_TILE_METERS)
    dlat, dlon = meters_to_latlon_delta(lat, tile_meters)

    # Calculate tile coordinates
    min_lat = lat - dlat / 2
    max_lat = lat + dlat / 2
    min_lon = lon - dlon / 2
    max_lon = lon + dlon / 2

    try:
        window = from_bounds(
            min_lon, min_lat, max_lon, max_lat,
            transform=ghsl_data.transform
        )
        pop_data = ghsl_data.read(1, window=window)
    except:
        print("TEST")
        continue

    if pop_data.size == 0:
        continue

    total_population = np.nansum(pop_data)

    area_m2 = compute_area_m2((min_lon, min_lat, max_lon, max_lat))
    area_km2 = area_m2 / 1e6

    if area_km2 == 0:
        continue

    density = total_population / area_km2
    log_density = math.log1p(density)

    # Sample proportional to log density
    if random.random() > min(1.0, log_density / 10):
        continue

    zoom = estimate_zoom(tile_meters, lat)
    img = download_satellite_image(lat, lon, zoom)

    if img is None:
        continue

    split = assign_split(lat, lon)

    filename = f"{i}.jpg"
    img.save(os.path.join(OUTPUT_DIR, filename), quality=95)

    records.append({
        "image": filename,
        "center_lat": lat,
        "center_lon": lon,
        "tile_meters": tile_meters,
        "area_km2": area_km2,
        "total_population": total_population,
        "avg_density": density,
        "log1p_density": log_density,
        "split": split
    })

# Save metadata
df = pd.DataFrame(records)
df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

print("Done.")
print("Total samples:", len(df))
