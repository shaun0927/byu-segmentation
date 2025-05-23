# postprocess/pp_byu.py
# ---------------------------------------------------------------
"""
BYU-Motor 후처리 (FP16 NMS 버전, down-sampling X)

* 볼륨당 모터 개수 ≤ 1 가정
* softmax-prob (D,H,W)  →  3-D max-pool NMS →  peak 1 개 채택
* 최고 확률 < THRESH 이면 “모터 없음”으로 (-1,-1,-1) 반환
"""

from __future__ import annotations
from typing import Tuple, Union
import torch, pandas as pd, numpy as np

# ───── 하이퍼 파라미터 ───────────────────────────────
VOX_SPACING_A = 10.0          # voxel ↔ Å 변환 값
NMS_RADIUS_VX = 15            # max-pool radius (voxel)
THRESH        = 0.30          # 최고 확률 cutoff
TOPK          = 5             # 후보 peak 개수
DIST_WEIGHT   = 1e-3          # joint score 거리 패널티 가중치
CUDA_OK       = torch.cuda.is_available()

# ───── 3-D NMS (FP16 지원) ───────────────────────────
def simple_nms(prob: torch.Tensor, radius: int = NMS_RADIUS_VX) -> torch.Tensor:
    """
    prob : (D,H,W) fp32 / fp16  –  device (CPU/GPU) 무관
    반환 : 동일 shape, peak 위치만 유지
    """
    k = radius * 2 + 1
    mp = torch.nn.functional.max_pool3d(
        prob[None, None],          # (B=1,C=1,D,H,W)
        kernel_size=k, stride=1, padding=radius
    )[0, 0]
    mask = prob >= mp
    return torch.where(mask, prob, torch.zeros_like(prob))

# ───── 메인 함수 ─────────────────────────────────────
def post_process_volume(
    prob_vol: Union[torch.Tensor, np.ndarray],
    *,
    spacing: float = VOX_SPACING_A,
    tomo_id: str   = "unknown",
    topk: int = TOPK,
    gt_coord: Tuple[float, float, float] | None = None,
    dist_weight: float = DIST_WEIGHT,
) -> pd.DataFrame:
    """
    확률 볼륨 → 1 row DataFrame (없으면 -1,-1,-1)

    Parameters
    ----------
    prob_vol : softmax 확률 볼륨 또는 로짓 텐서
    spacing  : voxel ↔ Å 변환 값
    tomo_id  : 결과 DataFrame 에 기록될 tomogram ID
    topk     : NMS 후 고려할 최고 확률 peak 수
    gt_coord : GT 좌표가 주어지면 거리 페널티를 적용해 최종 좌표 결정
    dist_weight : joint score 계산 시 거리 가중치
    """
    # 0) tensor 로 변환 및 Device / dtype 설정 --------------------
    if isinstance(prob_vol, np.ndarray):
        prob = torch.from_numpy(prob_vol)          # CPU tensor
    else:
        prob = prob_vol.detach()

    # GPU 사용 가능하면 GPU, 그리고 FP16 로 변환
    if CUDA_OK:
        prob = prob.cuda(non_blocking=True).half()  # fp16
    else:
        prob = prob.float()                         # CPU fp32

    # 1) NMS & peak 찾기 -----------------------------------------
    peaks = simple_nms(prob)                        # 같은 device/dtype
    flat = peaks.view(-1).float()
    k = min(topk, flat.numel())
    conf_k, idx_k = flat.topk(k)

    # 2) 후보 좌표 계산 -----------------------------------------
    D, H, W = prob.shape
    best = {"score": -float("inf"), "conf": 0.0, "coord": [-1.0, -1.0, -1.0]}
    for conf, idx in zip(conf_k, idx_k):
        conf_v = float(conf.item())
        ii = int(idx.item())
        z = ii // (H * W)
        y = (ii // W) % H
        x = ii % W
        coord_A = np.array([z * spacing, y * spacing, x * spacing], dtype=float)

        dist = 0.0
        if gt_coord is not None:
            dist = float(np.linalg.norm(coord_A - np.asarray(gt_coord)))
        score = conf_v - dist_weight * dist

        if score > best["score"]:
            best = {"score": score, "conf": conf_v, "coord": coord_A.tolist()}

    conf_val = best["conf"]

    # 2) 모터 없음 -----------------------------------------------
    if conf_val < THRESH:
        return pd.DataFrame(
            [[tomo_id, -1, -1, -1, conf_val]],
            columns=["tomo_id",
                     "Motor axis 0", "Motor axis 1", "Motor axis 2", "conf"]
        )

    # 3) voxel → Å 좌표 변환 -------------------------------------
    coord_Å = best["coord"]

    return pd.DataFrame(
        [[tomo_id, *coord_Å, conf_val]],
        columns=["tomo_id",
                 "Motor axis 0", "Motor axis 1", "Motor axis 2", "conf"]
    )
