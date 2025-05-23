# data/ds_byu.py (patched)
"""
BYU‑Motor Dataset (chunked‑Zarr version)
────────────────────────────────────────
• train : 1 tomo → 3 positive + 1 negative patch ( **96³** , z‑score)
• val   : full volume → non‑overlap GridPatch(**96³**) + locs
• test  : same as val (label 미포함)

Tensor shape  → (N, 1, 96, 96, 96) float32
CLI
    python -m data.ds_byu --mode train --batch 2
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict

import argparse, random, math
import numpy as np, torch, zarr, pandas as pd
from torch.utils.data import Dataset
import monai.transforms as mt

from utils import grid_split_3d, gaussian_kernel_3d

# ─── global hyper‑params ──────────────────────────────────────────
ROI: Tuple[int, int, int] = (96, 96, 96)          # (d, h, w)
POS_PER_TOMO, NEG_PER_TOMO = 2, 2                 # 기본 양성/음성 패치 수
BALANCE_PATCHES = True                            # 전체 비율 1:1 맞춤용
SIGMA_PX   = 8                                  # spacing‑aware σ 는 ROI 프로토타입 확인 후 조정
CUTOFF     = 0.02
MAX_TRY_NEG = 50
LABEL_CSV   = Path("data/raw/train_labels.csv")  # ← 경로 한 곳만!

# ─── helpers ──────────────────────────────────────────────────────

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

# ─── Dataset ─────────────────────────────────────────────────────

class BYUMotorDataset(Dataset):
    """
    mode  : "train" | "val" | "test"
    split : "train" / "test"  (data/processed/{split})
    """

    def __init__(
        self,
        tomo_ids: List[str],
        *,                         # ← keyword‑only! (안전)
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

        # 패치 비율 조정을 위한 초기 설정 ---------------------------------
        self.pos_per_tomo = POS_PER_TOMO
        if mode == "train" and BALANCE_PATCHES:
            info = self.df["Number of motors"].groupby("tomo_id").first()
            n_pos = int((info > 0).sum())
            n_neg = int((info == 0).sum())
            if n_pos > 0:
                total_neg = (n_pos + n_neg) * NEG_PER_TOMO
                self.pos_per_tomo = max(1, round(total_neg / n_pos))

        if mode == "train":
            # ── augmentation pipeline ────────────────────────────
            self.aug = mt.Compose([
                # Zoom : image+label 함께, 보간 분리
                mt.RandZoomd(
                    keys=["image", "label"],
                    prob=0.7, min_zoom=0.6, max_zoom=1.4,
                    mode=("trilinear", "nearest"),
                ),
                # Flips
                mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                # 90° Rotations
                mt.RandRotate90d(keys=["image", "label"], prob=0.75, max_k=3),
                # small free rotations
                mt.RandRotated(
                    keys=["image", "label"],
                    prob=0.2,
                    range_x=math.radians(15),
                    padding_mode="border",
                ),
                # Intensity scaling (train only)
                mt.RandScaleIntensityd(keys="image", prob=0.5, factors=0.1),
            ])
        else:
            self.aug = None

    # ------------------------------------------------------------
    def __len__(self):
        return len(self.ids)

    # ------------------------------------------------------------
    def _pos_patch(self, vol: zarr.Array, center: Tuple[int, int, int]):
        d, h, w = ROI
        D, H, W = vol.shape
        zc, yc, xc = center
        jitter = np.random.randint(-d // 4, d // 4 + 1, 3)
        z0 = np.clip(zc + jitter[0] - d // 2, 0, D - d)
        y0 = np.clip(yc + jitter[1] - h // 2, 0, H - h)
        x0 = np.clip(xc + jitter[2] - w // 2, 0, W - w)

        img = norm_patch(vol[z0 : z0 + d, y0 : y0 + h, x0 : x0 + w][...])

        dz, dy, dx = (zc - z0, yc - y0, xc - x0)
        lbl = gaussian_kernel_3d(
            ROI, (dz, dy, dx), SIGMA_PX, cutoff=CUTOFF
        ).astype(np.float32)

        return _to_tensor(img), _to_tensor(lbl)

    def _neg_patch(self, vol: zarr.Array):
        d, h, w = ROI
        D, H, W = vol.shape
        for _ in range(MAX_TRY_NEG):
            z0 = random.randint(0, D - d)
            y0 = random.randint(0, H - h)
            x0 = random.randint(0, W - w)
            patch = vol[z0 : z0 + d, y0 : y0 + h, x0 : x0 + w][...]
            if patch.max() < 200:  # heuristic motor‑free
                img = norm_patch(patch)
                return _to_tensor(img), _to_tensor(np.zeros(ROI, np.float32))

        # fallback
        img = norm_patch(vol[:d, :h, :w][...])
        return _to_tensor(img), _to_tensor(np.zeros(ROI, np.float32))

    # ------------------------------------------------------------
    def _grid_patches(self, vol: zarr.Array):
        patches, locs = grid_split_3d(vol, ROI)
        imgs = torch.stack([_to_tensor(norm_patch(p)) for p in patches])
        locs = torch.as_tensor(locs, dtype=torch.long)
        return imgs, locs

    # ------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tid = self.ids[idx]
        vol = load_zarr(self.root / f"{tid}.zarr")  # lazy (D,H,W)

        # ---------- val / test ----------
        if self.mode != "train":
            imgs, locs = self._grid_patches(vol)
            return {"image": imgs, "locs": locs, "tomo_id": tid}

        # ---------- train ----------
        row = (
            self.df.loc[tid]
            if not isinstance(self.df.loc[tid], pd.DataFrame)
            else self.df.loc[tid].iloc[0]
        )

        patches_i, patches_l = [], []

        if row["Number of motors"] > 0:
            # train_labels.csv 에 기록된 모터 좌표는 이미 voxel 단위이므로
            # 별도의 spacing 보정 없이 그대로 사용한다.
            ctr = (
                int(row["Motor axis 0"]),
                int(row["Motor axis 1"]),
                int(row["Motor axis 2"]),
            )
            for _ in range(self.pos_per_tomo):
                im, lb = self._pos_patch(vol, ctr)
                patches_i.append(im)
                patches_l.append(lb)

        for _ in range(NEG_PER_TOMO):
            im, lb = self._neg_patch(vol)
            patches_i.append(im)
            patches_l.append(lb)

        # aug (개별 패치단위)
        if self.aug:
            aug_i, aug_l = [], []
            for im, lb in zip(patches_i, patches_l):
                out = self.aug({"image": im, "label": lb})
                aug_i.append(out["image"])
                aug_l.append(out["label"])
            patches_i, patches_l = aug_i, aug_l

        # shape 검증 (debug)
        for t in patches_i:
            assert tuple(t.shape[-3:]) == ROI, f"patch shape {t.shape}"

        return {
            "image": torch.stack(patches_i),  # (4,1,96,96,96)
            "label": torch.stack(patches_l),
            "tomo_id": tid,
        }

# ─── collate ──────────────────────────────────────────────────────

def simple_collate(batch):
    out: Dict[str, object] = {}
    for k in batch[0]:
        v0 = batch[0][k]
        out[k] = (
            torch.cat([b[k] for b in batch], 0) if isinstance(v0, torch.Tensor) else [b[k] for b in batch]
        )
    return out

# ─── CLI smoke‑test ───────────────────────────────────────────────

def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "val", "test"], default="train")
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    tids = pd.read_csv(LABEL_CSV)["tomo_id"].unique()[:8]
    ds = BYUMotorDataset(tids, mode=args.mode, split="train")
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, num_workers=0, collate_fn=simple_collate, pin_memory=True
    )
    b = next(iter(dl))
    print("image :", b["image"].shape)
    if args.mode == "train":
        print("label :", b["label"].shape)

if __name__ == "__main__":
    _cli()
