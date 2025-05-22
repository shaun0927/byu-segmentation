"""tests/check_patch_logits.py
BYUMotorDataset에서 양성 패치만 골라 BYUNet 추론 logits.max()를 출력한다.
"""

import torch
import pandas as pd

from data.ds_byu import BYUMotorDataset, LABEL_CSV
from models.net_byu import BYUNet

# ── device 설정 ───────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. 양성 tomogram 선택 ─────────────────────────────────────
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

# ── 3. 모델 로드 (랜덤 가중치) ───────────────────────────────
net = BYUNet({"backbone": "resnet34"}).to(DEVICE)
net.eval()

# ── 4. 양성 패치만 추론하여 logits.max() 확인 ────────────────
for idx, (im, lb) in enumerate(zip(imgs, lbls)):
    if lb.max() == 0:
        continue  # 음성 패치 건너뜀

    with torch.no_grad():
        batch = {"image": im.unsqueeze(0).to(DEVICE)}
        out   = net(batch)
    mx = out["logits"].max().item()
    print(f"patch {idx} logits.max() = {mx:.4f}")
