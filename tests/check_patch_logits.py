"""tests/check_patch_logits.py
BYUMotorDataset에서 양성 패치를 골라 무작위 가중치와 학습된 가중치의
logits.max() 값을 비교한다.
"""

import argparse
import torch
import pandas as pd

from data.ds_byu import BYUMotorDataset, LABEL_CSV
from models.net_byu import BYUNet

# ── device 설정 ───────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="불러올 .pt 파일 경로")
    return ap.parse_args()

# ── 1. 인자 파싱 및 양성 tomogram 선택 ────────────────────────
args = parse_args()
labels_df = pd.read_csv(LABEL_CSV)
pos_rows  = labels_df.query("`Number of motors` >= 1")
if pos_rows.empty:
    raise RuntimeError("train_labels.csv에 모터 ≥1 tomogram이 없습니다.")

TID = pos_rows["tomo_id"].iloc[0]
print(f"[INFO] tomo_id = {TID}")

# ── 2. 데이터셋 준비 (train mode) ────────────────────────────
ds = BYUMotorDataset([TID], mode="train", split="train")
item = ds[0]  # 첫 번째 샘플 (양성 패치 포함)
imgs, lbls = item["image"], item["label"]

# ── 3. 모델 로드 ─────────────────────────────────────────────
net_rand = BYUNet({"backbone": "resnet34"}).to(DEVICE)
net_rand.eval()

net = BYUNet({"backbone": "resnet34"}).to(DEVICE)
state = torch.load(args.ckpt, map_location=DEVICE)
if "model" in state:
    state = state["model"]
net.load_state_dict(state, strict=False)
net.eval()

# ── 4. 양성 패치만 추론하여 logits 비교 ────────────────────
for idx, (im, lb) in enumerate(zip(imgs, lbls)):
    if lb.max() == 0:
        continue  # 음성 패치 건너뜀

    with torch.no_grad():
        batch = {"image": im.unsqueeze(0).to(DEVICE)}
        rand_out = net_rand(batch)
        ckpt_out = net(batch)

    rand_mx = rand_out["logits"].max().item()
    ckpt_mx = ckpt_out["logits"].max().item()
    diff    = ckpt_mx - rand_mx
    print(
        f"patch {idx} random={rand_mx:.4f}, ckpt={ckpt_mx:.4f}, Δ={diff:.4f}"
    )
