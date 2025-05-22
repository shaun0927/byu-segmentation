#!/usr/bin/env python
"""
tests/check_patch_logits.py
────────────────────────────────────────────────────────────
· 양성 패치에서 **무작위 가중치**(baseline)와 **학습된 체크포인트**의
  logits.max() 값을 비교해 peak 스케일을 확인한다.
· CLI:  --ckpt  <ckpt1.pt> [ckpt2.pt ...]
        여러 개를 공백으로 나열하면 순차 비교.
"""

from __future__ import annotations
import sys, pathlib, argparse
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import torch, pandas as pd
from data.ds_byu import BYUMotorDataset, LABEL_CSV
from models.net_byu import BYUNet

# ── CLI ────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt", required=True, nargs="+",
        help="불러올 .pt 체크포인트 경로(들). 공백으로 구분해 여러 개 전달 가능"
    )
    return ap.parse_args()

# ── device ────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_positive_patch():
    """첫 번째 '모터≥1' tomogram에서 양성 패치 하나 반환"""
    labels_df = pd.read_csv(LABEL_CSV)
    pos_tid   = labels_df.query("`Number of motors` >= 1")["tomo_id"].iloc[0]
    ds        = BYUMotorDataset([pos_tid], mode="train", split="train")

    for item in ds:
        if item["label"].sum() > 0:
            return item["image"], item["label"]
    raise RuntimeError("양성 패치를 찾지 못했습니다.")

# ── main ──────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    imgs, lbls = load_positive_patch()

    # ➊ 무작위 가중치 모델(baseline)
    net_rand = BYUNet({"backbone": "resnet34"}).to(DEVICE).eval()

    # ➋ 체크포인트별 비교
    for ckpt_path in args.ckpt:
        state = torch.load(ckpt_path, map_location=DEVICE)
        if "model" in state:                 # ← 변경
            state = state["model"]
        net = BYUNet({"backbone": "resnet34"}).to(DEVICE)
        net.load_state_dict(state, strict=False)
        net.eval()

        print(f"\n=== {ckpt_path} ===")
        for idx, (im, lb) in enumerate(zip(imgs, lbls)):
            if lb.max() == 0:
                continue  # 음성 패치 건너뜀

            with torch.no_grad():
                batch = {"image": im.unsqueeze(0).to(DEVICE)}
                rand_mx = net_rand(batch)["logits"].max().item()
                ckpt_mx = net(batch)["logits"].max().item()

            print(
                f"patch {idx:2d}  random={rand_mx:.4f}  "
                f"ckpt={ckpt_mx:.4f}  Δ={ckpt_mx - rand_mx:.4f}"
            )

if __name__ == "__main__":
    main()
