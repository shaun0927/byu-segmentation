#!/usr/bin/env python
"""
postprocess/pp_byu.py
────────────────────────────────────────────────────────
BYU-Motor – inference 후처리
  • 한 볼륨에 모터가 0 or 1 개라는 가정
  • softmax → simple 3-D NMS → 최고점 1 개만 채택
  • 최고 확률 < THRESH ⇒ “모터 없음”  → (-1,-1,-1)

사용 예
-------
logits = net(patch_batch)["logits"].softmax(1)[:,1]  # (D,H,W) prob
full_prob = grid_reconstruct_3d(prob_batch, locs, vol_shape)
pred_df   = post_process_volume(full_prob, spacing=10.0, tomo_id="t123")
"""

from __future__ import annotations
from typing import Tuple
import torch, pandas as pd

# ───── 하이퍼 파라미터 ────────────────────────────────
VOX_SPACING_A = 10.0          # 기본 grid_reconstruct 에 맞춘 voxel spacing [Å]
NMS_RADIUS_VX = 5             # ≈ 50 Å / 10 ≃ 5 voxel
THRESH        = 0.30          # 최고 확률 < THRESH → 모터 없음

# ───── 빠른 3-D NMS ───────────────────────────────────
def simple_nms(prob: torch.Tensor, radius: int = NMS_RADIUS_VX) -> torch.Tensor:
    """
    prob : (D,H,W)  – float 0-1
    반환  : 동일 shape, NMS 로 남은 peak 위치만 유지
    """
    mp = torch.nn.functional.max_pool3d(
        prob[None], kernel_size=radius * 2 + 1, stride=1,
        padding=radius)[0]
    keep = prob.eq(mp)
    return torch.where(keep, prob, torch.zeros_like(prob))

# ───── 주 함수 ────────────────────────────────────────
def post_process_volume(
        prob_vol: torch.Tensor | torch.FloatTensor,   # (D,H,W) 확률
        spacing: float = VOX_SPACING_A,
        tomo_id: str = "unknown"
) -> pd.DataFrame:
    """
    확률 볼륨 → 좌표 0 또는 1 개 DataFrame 반환
    """
    if isinstance(prob_vol, torch.Tensor):
        prob = prob_vol.float().detach()
    else:  # numpy → torch
        prob = torch.from_numpy(prob_vol).float()

    # 1) NMS & 최고점
    peaks = simple_nms(prob)                       # (D,H,W)
    conf, flat_idx = peaks.view(-1).max(0)
    conf = conf.item()

    # 2) 모터 없음
    if conf < THRESH:
        return pd.DataFrame(
            [[tomo_id, -1, -1, -1, conf]],
            columns=["tomo_id",
                     "Motor axis 0", "Motor axis 1", "Motor axis 2", "conf"]
        )

    # 3) 모터 있음 → voxel → Å 좌표 변환
    D, H, W = prob.shape
    z = flat_idx // (H * W)
    y = (flat_idx // W) % H
    x = flat_idx % W
    coord_Å = [int(z) * spacing, int(y) * spacing, int(x) * spacing]

    return pd.DataFrame(
        [[tomo_id, *coord_Å, conf]],
        columns=["tomo_id",
                 "Motor axis 0", "Motor axis 1", "Motor axis 2", "conf"]
    )
