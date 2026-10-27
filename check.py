"""Sanity checks for SmallMinesDS before designing splits.

Usage:
    python check_smallminesds.py <path/to/data_splits> <path/to/unzipped/patches>

1. Cross-year leakage: does the same tile ID appear in TRAIN in one year
   and in TEST/VALIDATION in the other? (Same ground location, different date.)
2. Georeferencing: do the patch GeoTIFFs carry a CRS and transform?
   If yes, patch centroids can be used to build spatially blocked splits.
"""
import sys
import glob
import os
import pandas as pd

splits_dir, patches_dir = "data_splits", "Dataset"

# ---- 1. Cross-year leakage --------------------------------------------------
frames = []
for year in (2016, 2022):
    df = pd.read_csv(os.path.join(splits_dir, f"train_test_splits_{year}.csv"))
    df["year"] = year
    df["tile_id"] = df["patch_name"].str.extract(r"_(\d+)_\d{4}")[0]
    frames.append(df)
s = pd.concat(frames)

print("Split counts per year:\n", s.groupby(["year", "split"]).size(), "\n")

wide = s.pivot_table(index="tile_id", columns="year", values="split", aggfunc="first")
straddle = wide[(wide[2016] != wide[2022]) & wide[2016].notna() & wide[2022].notna()]
print(f"Tiles present in both years: {wide.dropna().shape[0]}")
print(f"Tiles whose split differs between years (cross-year leakage): {len(straddle)}")
print(straddle.head(10), "\n")

# ---- 2. Georeferencing ------------------------------------------------------
try:
    import rasterio
except ImportError:
    sys.exit("Install rasterio (pip install rasterio) to run the georeferencing check.")

tifs = sorted(glob.glob(os.path.join(patches_dir, "**", "*.tif"), recursive=True))
print(f"Found {len(tifs)} .tif files")
for path in tifs[:3]:
    with rasterio.open(path) as src:
        print(os.path.basename(path), "| bands:", src.count,
              "| CRS:", src.crs, "| bounds:", tuple(round(b, 1) for b in src.bounds))