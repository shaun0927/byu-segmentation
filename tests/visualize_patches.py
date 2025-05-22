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
    im_np = im[0].numpy()
    lb_np = lb[0].numpy()
    mid = [s // 2 for s in im_np.shape]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for ax_idx in range(3):
        img_slice = np.take(im_np, mid[ax_idx], axis=ax_idx)
        lbl_slice = np.take(lb_np, mid[ax_idx], axis=ax_idx)

        axes[0, ax_idx].imshow(img_slice, cmap="gray")
        axes[0, ax_idx].axhline(mid[(ax_idx + 1) % 3], color="r", linestyle="--", linewidth=0.5)
        axes[0, ax_idx].axvline(mid[(ax_idx + 2) % 3], color="r", linestyle="--", linewidth=0.5)
        axes[0, ax_idx].set_title(f"image axis {ax_idx}")

        axes[1, ax_idx].imshow(lbl_slice, cmap="gray")
        axes[1, ax_idx].set_title(f"label axis {ax_idx}")

    fig.suptitle(f"patch {idx}")
    out_path = OUT_DIR / f"patch_{idx}.png"
    fig.savefig(out_path)
    print("saved", out_path)
    plt.close(fig)
