"""
1개 train tomogram → 모델 dummy 예측 → 후처리 → 좌표 0/1 출력 확인
"""
import torch, pandas as pd
from data.ds_byu import BYUMotorDataset, LABEL_CSV, simple_collate, ROI
from utils import grid_split_3d, grid_reconstruct_3d
from postprocess.pp_byu import post_process_volume, THRESH
from models.net_byu import BYUNet

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── load 1 tomogram (GT 0 or 1 motor) ─────────────────────────
tid = pd.read_csv(LABEL_CSV).iloc[0]["tomo_id"]
ds  = BYUMotorDataset([tid], mode="val", split="train")
item = ds[0]

full_vol = item["label"][0,0]          # (D,H,W)  float32  (Gaussian or zeros)
spacing  = pd.read_csv(LABEL_CSV).set_index("tomo_id").loc[tid]["Voxel spacing"]

# ── fake logits : bg = 0, motor = +6×heat-map ─────────────────
# (강한 양성 신호이면 모터 검출, 없으면 all − 열어두고 THRESH 로 거름)
motor_logit = torch.tensor(full_vol) * 6.0        # 강도 조절
bg_logit    = torch.zeros_like(motor_logit)
logits = torch.stack([bg_logit, motor_logit])     # (2,D,H,W)

df = post_process_volume(logits, spacing, tomo_id=tid, topk=5,
                         expected_max_dist=1000.0)
print(df)
