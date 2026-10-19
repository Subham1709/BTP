"""One-off helper: turn Results/susceptibility_zones.tif into a coloured PNG for the web map.

The GeoTIFF is in UTM 43N (metres). Web maps use longitude/latitude, so the grid is
reprojected to EPSG:4326 once here, coloured by zone, and saved with its corner
coordinates. The app then only needs the PNG, so rasterio is not a deploy dependency.

Run from the repo root:  python app/make_overlay.py   (needs rasterio, pillow)
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject
from PIL import Image

SRC = Path("study_area/Results/susceptibility_zones.tif")
OUT = Path("app/assets")

# zone value -> RGBA (0 = outside the study area, fully transparent)
COLOURS = {
    1: (26, 150, 65, 170),    # Very Low
    2: (166, 217, 106, 170),  # Low
    3: (255, 255, 191, 170),  # Moderate
    4: (253, 174, 97, 190),   # High
    5: (215, 25, 28, 200),    # Very High
}

with rasterio.open(SRC) as src:
    transform, width, height = calculate_default_transform(
        src.crs, "EPSG:4326", src.width, src.height, *src.bounds
    )
    zones = np.zeros((height, width), dtype=np.uint8)
    reproject(
        source=rasterio.band(src, 1),
        destination=zones,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.nearest,  # zones are categories, never average them
        dst_nodata=0,
    )

rgba = np.zeros((height, width, 4), dtype=np.uint8)
for zone, colour in COLOURS.items():
    rgba[zones == zone] = colour

OUT.mkdir(parents=True, exist_ok=True)
Image.fromarray(rgba, "RGBA").save(OUT / "susceptibility_zones.png", optimize=True)

west, north = transform * (0, 0)
east, south = transform * (width, height)
(OUT / "susceptibility_zones_bounds.json").write_text(
    json.dumps({"west": west, "south": south, "east": east, "north": north}, indent=2)
)
print(f"saved {width}x{height} overlay, bounds W{west:.4f} S{south:.4f} E{east:.4f} N{north:.4f}")
