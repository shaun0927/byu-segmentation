# tests/reconstruct_check.py
import torch, pandas as pd
from data.ds_byu import BYUMotorDataset, simple_collate, LABEL_CSV, load_zarr  # ← 추가
from utils import grid_reconstruct_3d

tid = pd.read_csv(LABEL_CSV)["tomo_id"].iloc[0]
ds  = BYUMotorDataset([tid], mode="val", split="train")
item = ds[0]

imgs, lbls, locs = item["image"], item["label"], item["locs"]
vol_shape = load_zarr(ds.root / f"{tid}.zarr").shape    # (D,H,W)

rec = grid_reconstruct_3d(imgs, locs, vol_shape)
print("reconstructed :", rec.shape, "expected :", vol_shape)
assert rec.shape[1:] == vol_shape, "Shape mismatch!"
