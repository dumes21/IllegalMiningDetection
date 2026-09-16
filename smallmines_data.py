"""SmallMinesDS -> TerraMind bi-temporal data pipeline.

Each tile has a 2016 and a 2022 patch (13 x 128 x 128) and a binary mine mask per year.
This module

- maps the 13 bands onto TerraMind's pretraining modalities (S2L2A, S1RTC, DEM),
- standardises them with TerraMind's own pretraining statistics,
- stacks the two years on a time axis so every modality is (C, T=2, H, W), which is what
  TerraTorch's ``TemporalWrapper`` expects,
- encodes the pair of binary masks as one 4-class target so per-year masks and the
  change map fall out of a standard semantic-segmentation head.

This file is also written out from ``terramind.ipynb`` (``%%writefile``) so the notebook
stays self-contained on Colab; keep the two in sync by editing the notebook cell.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

YEARS = ("2016", "2022")
IMAGE_NODATA = -9999.0
MASK_NODATA = 255
IGNORE_INDEX = -1

# Band layout of the 13-band SmallMinesDS GeoTIFF, expressed in TerraMind's band vocabulary.
# The dataset card lists the radar as Sentinel-1 RTC (terrain corrected + speckle filtered),
# which is its own TerraMind modality.
MODALITY_BANDS = {
    "S2L2A": ["BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3",
              "NIR_BROAD", "NIR_NARROW", "SWIR_1", "SWIR_2"],
    "S1RTC": ["VV", "VH"],
    "DEM": ["DEM"],
}
MODALITY_INDICES = {
    "S2L2A": list(range(0, 10)),
    "S1RTC": [10, 11],
    "DEM": [12],
}
ALL_MODALITIES = tuple(MODALITY_BANDS)

# TerraMind v1 pretraining statistics (from IBM/terramind configs). S2L2A is the 12-band
# list; the two bands this dataset lacks (COASTAL_AEROSOL, WATER_VAPOR) are dropped below.
_TM_S2L2A_BANDS = ["COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2",
                   "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "SWIR_1", "SWIR_2"]
_TM_MEAN = {
    "S2L2A": [1390.458, 1503.317, 1718.197, 1853.910, 2199.100, 2779.975,
              2987.011, 3083.234, 3132.220, 3162.988, 2424.884, 1857.648],
    "S1RTC": [-10.93, -17.329],
    "DEM": [670.665],
}
_TM_STD = {
    "S2L2A": [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926,
              1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311],
    "S1RTC": [4.391, 4.459],
    "DEM": [951.272],
}


def _subset(stats, modality):
    if modality == "S2L2A":
        return [stats[_TM_S2L2A_BANDS.index(b)] for b in MODALITY_BANDS["S2L2A"]]
    return stats


TERRAMIND_MEAN = {m: np.array(_subset(_TM_MEAN[m], m), dtype=np.float32) for m in ALL_MODALITIES}
TERRAMIND_STD = {m: np.array(_subset(_TM_STD[m], m), dtype=np.float32) for m in ALL_MODALITIES}

# 4-class target: mutually exclusive, so a standard softmax head predicts both years at once.
CLASS_NAMES = ["no mine", "mine 2016 only", "mine 2022 only (new)", "mine both years"]
NUM_CLASSES = len(CLASS_NAMES)


# --------------------------------------------------------------------------- paths & splits

def find_data_root(start=Path(".")):
    """Locate the folder holding <year>/IMAGE and <year>/MASK.

    Local layout is ``Dataset/2016/...``; the unzipped HF download is
    ``SmallMinesDS/SmallMinesDS/2016/...``.
    """
    start = Path(start)
    for cand in (start / "Dataset", start / "SmallMinesDS" / "SmallMinesDS", start / "SmallMinesDS"):
        if all((cand / y / "IMAGE").is_dir() and (cand / y / "MASK").is_dir() for y in YEARS):
            return cand
    raise FileNotFoundError(f"no <year>/IMAGE folders found under {start.resolve()}")


def find_splits_dir(start=Path(".")):
    start = Path(start)
    for cand in (start / "data_splits", start / "SmallMinesDS" / "data_splits"):
        if (cand / "train_test_splits_2016.csv").is_file():
            return cand
    raise FileNotFoundError(f"no data_splits/ folder found under {start.resolve()}")


def patch_paths(data_root, year, tile):
    data_root = Path(data_root)
    return (data_root / year / "IMAGE" / f"IMG_GH_{tile}_{year}.tif",
            data_root / year / "MASK" / f"MASK_GH_{tile}_{year}.tif")


def make_paired_split(splits_dir, out_csv, seed=42, val_frac=0.15, test_frac=0.15):
    """Build one tile-level split for the paired (2016, 2022) task.

    The published per-year CSVs were stratified independently, so 42% of tiles land in
    different splits in the two years. A model that sees both years of a tile needs a
    single assignment per tile, so we draw a fresh split stratified on the 2016->2022
    change in mine coverage (the quantity a change model has to get right).
    """
    from sklearn.model_selection import train_test_split

    frames = []
    for year in YEARS:
        df = pd.read_csv(Path(splits_dir) / f"train_test_splits_{year}.csv")
        df["tile"] = df["patch_name"].str.extract(r"_(\d{4})_")
        frames.append(df.set_index("tile")[["class_percentage", "split"]]
                        .rename(columns={"class_percentage": f"pct_{year}", "split": f"split_{year}"}))
    paired = frames[0].join(frames[1], how="inner").reset_index()
    paired["delta"] = paired["pct_2022"] - paired["pct_2016"]
    paired["change_bin"] = pd.cut(paired["delta"], bins=[-np.inf, 0, 1, 5, 15, np.inf],
                                  labels=["<=0", "0-1", "1-5", "5-15", ">15"])

    train_val, test = train_test_split(paired, test_size=test_frac, random_state=seed,
                                       stratify=paired["change_bin"])
    train, val = train_test_split(train_val, test_size=val_frac / (1 - test_frac),
                                  random_state=seed, stratify=train_val["change_bin"])
    paired["split"] = "train"
    paired.loc[paired.tile.isin(val.tile), "split"] = "val"
    paired.loc[paired.tile.isin(test.tile), "split"] = "test"
    paired = paired.sort_values("tile").reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out_csv, index=False)
    return paired


# --------------------------------------------------------------------------- dataset

def encode_target(mask_2016, mask_2022):
    """Two binary masks -> one 4-class map; nodata in either year becomes IGNORE_INDEX."""
    invalid = (mask_2016 == MASK_NODATA) | (mask_2022 == MASK_NODATA)
    target = (mask_2016 == 1).astype(np.int64) + 2 * (mask_2022 == 1).astype(np.int64)
    target[invalid] = IGNORE_INDEX
    return target


def decode_target(target):
    """Inverse of encode_target: (mine_2016, mine_2022, new_mining) boolean maps."""
    mine_2016 = (target == 1) | (target == 3)
    mine_2022 = (target == 2) | (target == 3)
    return mine_2016, mine_2022, target == 2


class SmallMinesPairedDataset(Dataset):
    """One item = one tile with both years stacked on a time axis.

    Returns ``{"image": {modality: float tensor (C, 2, H, W)}, "mask": long tensor (H, W),
    "filename": tile}``. The ``filename`` key is the one TerraTorch tasks already ignore.
    """

    def __init__(self, data_root, tiles, modalities=ALL_MODALITIES, augment=False):
        self.data_root = Path(data_root)
        self.tiles = list(tiles)
        self.modalities = list(modalities)
        self.augment = augment

    def __len__(self):
        return len(self.tiles)

    def _read(self, year, tile):
        img_path, mask_path = patch_paths(self.data_root, year, tile)
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        return image, mask

    def __getitem__(self, i):
        tile = self.tiles[i]
        images, masks = zip(*(self._read(year, tile) for year in YEARS))
        target = encode_target(*masks)

        sample = {}
        for mod in self.modalities:
            idx = MODALITY_INDICES[mod]
            x = np.stack([img[idx] for img in images], axis=1)  # (C, T, H, W)
            nodata = x == IMAGE_NODATA
            x = (x - TERRAMIND_MEAN[mod][:, None, None, None]) / TERRAMIND_STD[mod][:, None, None, None]
            x[nodata] = 0.0  # nodata sits at the pretraining mean after standardisation
            sample[mod] = x

        if self.augment:
            sample, target = self._d4(sample, target)

        return {
            "image": {mod: torch.from_numpy(np.ascontiguousarray(x)) for mod, x in sample.items()},
            "mask": torch.from_numpy(np.ascontiguousarray(target)),
            "filename": tile,
        }

    @staticmethod
    def _d4(sample, target):
        """Random dihedral flip/rotation applied identically to every modality and the target."""
        k = np.random.randint(4)
        flip = np.random.rand() < 0.5
        for mod, x in sample.items():
            x = np.rot90(x, k, axes=(-2, -1))
            sample[mod] = x[..., ::-1] if flip else x
        target = np.rot90(target, k, axes=(-2, -1))
        return sample, (target[..., ::-1] if flip else target)


class SmallMinesPairedDataModule(LightningDataModule):
    def __init__(self, data_root, split_csv, modalities=ALL_MODALITIES, batch_size=8,
                 num_workers=2, augment=True, limit=None):
        super().__init__()
        self.data_root = Path(data_root)
        self.split_csv = Path(split_csv)
        self.modalities = list(modalities)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.limit = limit  # cap tiles per split for smoke tests
        self.tiles = {}

    def setup(self, stage=None):
        df = pd.read_csv(self.split_csv, dtype={"tile": str})
        for split in ("train", "val", "test"):
            tiles = df.loc[df.split == split, "tile"].tolist()
            self.tiles[split] = tiles[: self.limit] if self.limit else tiles

    def _loader(self, split, shuffle):
        ds = SmallMinesPairedDataset(self.data_root, self.tiles[split], self.modalities,
                                     augment=self.augment and split == "train")
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers > 0, drop_last=shuffle)

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("val", shuffle=False)

    def test_dataloader(self):
        return self._loader("test", shuffle=False)

    def predict_dataloader(self):
        return self.test_dataloader()
