# data/ds_byu.py
"""
BYU-Motor Dataset  (chunked-Zarr version)
────────────────────────────────────────────────────────
• train : 1 tomo → 3 positive + 1 negative patch (128³, z-score)
• val   : full volume → non-overlap GridPatch(128³) + locs
• test  : same as val (label 미포함)

모든 tensor  → (N, 1, 128, 128, 128) float32
CLI
    python -m data.ds_byu --mode train --batch 2
"""
from __future__ import annotations
from pathlib import Path
from typing  import List, Tuple, Dict

import argparse, random, math
import numpy as np, torch, zarr, pandas as pd
from torch.utils.data import Dataset
import monai.transforms as mt

from utils import gaussian_kernel_3d, grid_split_3d

# ─── global hyper-params ───────────────────────────────────────────
ROI: Tuple[int, int, int] = (128, 128, 128)        # (d, h, w)
POS_PER_TOMO, NEG_PER_TOMO = 3, 1
SIGMA_PX   = 4.5                                     # ≃ 45 nm @10 Å / 2.355
CUTOFF     = 0.05
MAX_TRY_NEG = 50
LABEL_CSV   = Path("data/raw/train_labels.csv")      # ← 경로 한 곳만!

# ─── helpers ───────────────────────────────────────────────────────
def load_zarr(path: Path) -> zarr.Array:
    """lazy zarr array 반환(group → 첫 array)."""
    arr = zarr.open(str(path), mode="r")
    if isinstance(arr, zarr.hierarchy.Group):
        arr = next(iter(arr.values()))
    return arr  # (D,H,W) uint8

def norm_patch(p: np.ndarray) -> np.ndarray:
    p = p.astype(np.float32)
    return (p - p.mean()) / (p.std() + 1e-6)

def _to_tensor(a):
    if isinstance(a, np.ndarray):
        t = torch.from_numpy(a)
    else:
        t = a.clone().detach()
    if t.ndim == 3:                       # (D,H,W) → (1,D,H,W)
        t = t.unsqueeze(0)
    return t

# ─── Dataset ───────────────────────────────────────────────────────
class BYUMotorDataset(Dataset):
    """
    mode : "train" | "val" | "test"
    split: "train" / "test"  (data/processed/{split})
    """
    def __init__(
        self,
        tomo_ids: List[str],
        *,                         # ← keyword-only! (안전)
        split: str = "train",
        mode : str = "train",
        root : str = "data/processed",
    ):
        self.ids   = list(tomo_ids)
        self.split = split
        self.mode  = mode
        self.root  = Path(root) / split

        if mode != "test":
            self.df = (
                pd.read_csv(LABEL_CSV)
                  .set_index("tomo_id")
                  .loc[self.ids]          # train/val 공용 label DF
            )

        if mode == "train":
            self.aug = mt.Compose([
                mt.RandZoomd(keys="image", prob=0.7, min_zoom=0.6, max_zoom=1.4),
                mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                mt.RandRotate90d(keys=["image", "label"], prob=0.75, max_k=3),
                mt.RandRotated(
                    keys=["image", "label"],
                    prob=0.2,
                    range_x=math.radians(15),
                    padding_mode="border",
                ),
            ])
        else:
            self.aug = None

    # ----------------------------------------------------------------
    def __len__(self): return len(self.ids)

    # ----------------------------------------------------------------
    def _pos_patch(self, vol: zarr.Array, center: Tuple[int,int,int]):
        d, h, w = ROI
        D, H, W = vol.shape
        zc, yc, xc = center
        jitter = np.random.randint(-d//4, d//4+1, 3)
        z0 = np.clip(zc + jitter[0] - d//2, 0, D-d)
        y0 = np.clip(yc + jitter[1] - h//2, 0, H-h)
        x0 = np.clip(xc + jitter[2] - w//2, 0, W-w)

        img = norm_patch(vol[z0:z0+d, y0:y0+h, x0:x0+w][...])
        lbl = gaussian_kernel_3d(
            ROI, (zc-z0, yc-y0, xc-x0), SIGMA_PX, cutoff=CUTOFF
        )

        assert img.shape == ROI, f"positive size mismatch {img.shape}"
        return _to_tensor(img), _to_tensor(lbl)

    def _neg_patch(self, vol: zarr.Array):
        d, h, w = ROI
        D, H, W = vol.shape
        for _ in range(MAX_TRY_NEG):
            z0 = random.randint(0, D-d)
            y0 = random.randint(0, H-h)
            x0 = random.randint(0, W-w)
            patch = vol[z0:z0+d, y0:y0+h, x0:x0+w][...]
            if patch.max() < 200:                      # heuristic motor-free
                img = norm_patch(patch)
                assert img.shape == ROI
                return _to_tensor(img), _to_tensor(np.zeros(ROI, np.float32))

        # fallback
        img = norm_patch(vol[:d, :h, :w][...])
        assert img.shape == ROI
        return _to_tensor(img), _to_tensor(np.zeros(ROI, np.float32))

    # ----------------------------------------------------------------
    def _grid_patches(self, vol: zarr.Array):
        patches, locs = grid_split_3d(vol, ROI)
        imgs = torch.stack([_to_tensor(norm_patch(p)) for p in patches])
        locs = torch.as_tensor(locs, dtype=torch.long)
        return imgs, locs

    # ----------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tid = self.ids[idx]
        vol = load_zarr(self.root / f"{tid}.zarr")      # lazy (D,H,W)

        # ---------- val / test ----------
        if self.mode != "train":
            imgs, locs = self._grid_patches(vol)
            return {"image": imgs, "locs": locs, "tomo_id": tid}

        # ---------- train ----------
        row = self.df.loc[tid] if not isinstance(self.df.loc[tid], pd.DataFrame) \
              else self.df.loc[tid].iloc[0]

        patches_i, patches_l = [], []

        if row["Number of motors"] > 0:
            spc = row["Voxel spacing"]
            ctr = (int(row["Motor axis 0"]/spc),
                   int(row["Motor axis 1"]/spc),
                   int(row["Motor axis 2"]/spc))
            for _ in range(POS_PER_TOMO):
                im, lb = self._pos_patch(vol, ctr)
                patches_i.append(im); patches_l.append(lb)

        for _ in range(NEG_PER_TOMO):
            im, lb = self._neg_patch(vol)
            patches_i.append(im); patches_l.append(lb)

        # aug (개별 패치단위)
        if self.aug:
            aug_i, aug_l = [], []
            for im, lb in zip(patches_i, patches_l):
                out = self.aug({"image": im, "label": lb})
                aug_i.append(out["image"]); aug_l.append(out["label"])
            patches_i, patches_l = aug_i, aug_l

        # 최종 shape 검증
        for t in patches_i:
            assert tuple(t.shape[-3:]) == ROI, f"patch shape {t.shape}"

        return {
            "image": torch.stack(patches_i),   # (4,1,128,128,128)
            "label": torch.stack(patches_l),
            "tomo_id": tid,
        }

# ─── collate ───────────────────────────────────────────────────────
def simple_collate(batch):
    out: Dict[str, object] = {}
    for k in batch[0]:
        v0 = batch[0][k]
        out[k] = torch.cat([b[k] for b in batch], 0) if isinstance(v0, torch.Tensor) \
                 else [b[k] for b in batch]
    return out

# ─── CLI smoke-test ────────────────────────────────────────────────
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train","val","test"], default="train")
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    tids = pd.read_csv(LABEL_CSV)["tomo_id"].unique()[:8]
    ds   = BYUMotorDataset(tids, mode=args.mode, split="train")
    dl   = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, num_workers=0,
        collate_fn=simple_collate, pin_memory=True
    )
    b = next(iter(dl))
    print("image :", b["image"].shape)
    if args.mode == "train":
        print("label :", b["label"].shape)

if __name__ == "__main__":
    _cli()
