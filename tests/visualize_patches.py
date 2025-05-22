#!/usr/bin/env python
"""
visualize_patches.py
────────────────────────────────────────────────────────
· train_labels.csv 에서 'Number of motors ≥ 1' 인 tomo_id 선택
· BYUMotorDataset 에서 label>0 인 **양성 패치**만 시각화
· 각 양성 패치마다
    ├─ 3축 중심 슬라이스 PNG
    └─ 3축 Max-Intensity Projection(MIP) PNG
· 결과 파일은 스크립트와 같은 tests/ 폴더에 저장
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from data.ds_byu import BYUMotorDataset, LABEL_CSV

# ── 0. 시드 고정 (재현성) ─────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 1. 양성 tomogram 선택 ───────────────────────────────────────
labels_df = pd.read_csv(LABEL_CSV)
pos_rows  = labels_df.query("`Number of motors` >= 1")
if pos_rows.empty:
    raise RuntimeError("train_labels.csv에 모터 ≥1 tomogram이 없습니다.")

TID = pos_rows["tomo_id"].iloc[0]
print(f"[INFO] visualizing tomo_id = {TID}")

# ── 2. Dataset & 양성 패치 탐색 ──────────────────────────────────
ds = BYUMotorDataset([TID], mode="train", split="train")
pos_indices = [i for i in range(len(ds)) if ds[i]["label"].sum() > 0]

if not pos_indices:
    raise RuntimeError("선택한 tomogram에서 양성 패치를 찾지 못했습니다.")

# ── 3. 시각화 루프 ───────────────────────────────────────────────
OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

for pidx in pos_indices:
    item = ds[pidx]
    imgs, lbls = item["image"], item["label"]

    for idx, (im, lb) in enumerate(zip(imgs, lbls)):
        if lb.max() == 0:
            continue  # 안전장치: 혹시 음성 패치가 섞여 있으면 건너뜀

        im_np = im[0].numpy()
        lb_np = lb[0].numpy()

        # ── (a) 중심 슬라이스 시각화 ────────────────────────────
        zc, yc, xc = np.unravel_index(lb_np.argmax(), lb_np.shape)
        centers = [zc, yc, xc]

        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        for ax_idx in range(3):
            img_slice = np.take(im_np, centers[ax_idx], axis=ax_idx)
            lbl_slice = np.take(lb_np, centers[ax_idx], axis=ax_idx)

            axes[0, ax_idx].imshow(img_slice, cmap="gray")
            axes[0, ax_idx].set_title(f"image axis {ax_idx}")
            axes[1, ax_idx].imshow(lbl_slice, cmap="gray")
            axes[1, ax_idx].set_title(f"label axis {ax_idx}")

        fig.suptitle(f"positive patch {pidx}_{idx}")
        out_path = OUT_DIR / f"patch_{pidx}_{idx}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print("saved", out_path)
        plt.close(fig)

        # ── (b) Max-Intensity Projection 시각화 ────────────────
        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        for ax_idx in range(3):
            img_mip = im_np.max(axis=ax_idx)
            lbl_mip = lb_np.max(axis=ax_idx)
            axes[0, ax_idx].imshow(img_mip, cmap="gray")
            axes[0, ax_idx].set_title(f"MIP image axis {ax_idx}")
            axes[1, ax_idx].imshow(lbl_mip, cmap="gray")
            axes[1, ax_idx].set_title(f"MIP label axis {ax_idx}")

        fig.suptitle(f"MIP patch {pidx}_{idx}")
        out_path = OUT_DIR / f"patch_{pidx}_{idx}_mip.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print("saved", out_path)
        plt.close(fig)
