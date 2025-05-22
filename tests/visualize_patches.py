import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from data.ds_byu import BYUMotorDataset, LABEL_CSV

# 첫 번째 tomogram만 사용
TID = pd.read_csv(LABEL_CSV)["tomo_id"].iloc[0]

# train 모드에서 양성/음성 패치 추출
Ds = BYUMotorDataset([TID], mode="train", split="train")
item = Ds[0]
imgs, lbls = item["image"], item["label"]

# 시각화 결과 저장 위치
OUT_DIR = Path(__file__).parent

for idx, (im, lb) in enumerate(zip(imgs, lbls)):
    # 양성 패치만 시각화
    if lb.max() == 0:
        continue

    im_np = im[0].numpy()
    lb_np = lb[0].numpy()

    # 히트맵 중심을 찾아 해당 슬라이스 시각화
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

    fig.suptitle(f"positive patch {idx}")
    out_path = OUT_DIR / f"patch_{idx}.png"
    fig.savefig(out_path)
    print("saved", out_path)
    plt.close(fig)

    # ── max projection figure ──────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for ax_idx in range(3):
        img_mip = im_np.max(axis=ax_idx)
        lbl_mip = lb_np.max(axis=ax_idx)
        axes[0, ax_idx].imshow(img_mip, cmap="gray")
        axes[0, ax_idx].set_title(f"MIP image axis {ax_idx}")
        axes[1, ax_idx].imshow(lbl_mip, cmap="gray")
        axes[1, ax_idx].set_title(f"MIP label axis {ax_idx}")

    fig.suptitle(f"MIP patch {idx}")
    out_path = OUT_DIR / f"patch_{idx}_mip.png"
    fig.savefig(out_path)
    print("saved", out_path)
    plt.close(fig)
