"""
metrics/metric_f2.py
────────────────────────────────────────────────────────
BYU-Motor 대회용 F-β (β=2) 계산기
  • 볼륨당 GT 는 0 or 1 motor
  • prediction 은 좌표 3개 또는 (-1,-1,-1)
  • TP / FP / FN 규칙은 대회 규정과 동일
사용 예
-------
from metrics.metric_f2 import f2_score
score = f2_score(gt_df, pred_df, tau=1000.0)   # Å 단위 τ
"""

from __future__ import annotations
import numpy as np, pandas as pd


def _is_no_motor(row) -> bool:
    """(-1,-1,-1) 좌표인지"""
    return (row["Motor axis 0"] < 0) and (row["Motor axis 1"] < 0) and (row["Motor axis 2"] < 0)


def f2_score(
    gt:  pd.DataFrame,
    sub: pd.DataFrame,
    tau: float = 1000.0,     # Å
    beta: float = 2.0,
) -> float:
    """
    Parameters
    ----------
    gt   : columns = ['tomo_id', 'Motor axis 0/1/2']  (GT 0 또는 1 개)
    sub  : columns = ['tomo_id', 'Motor axis 0/1/2']
    tau  : 허용 거리 [Å]
    """
    # index by tomo_id for O(1) lookup
    gt  = gt.set_index("tomo_id")
    sub = sub.set_index("tomo_id")

    TP = FP = FN = 0

    for tid, g in gt.iterrows():
        gt_has = not _is_no_motor(g)

        try:
            s = sub.loc[tid]
        except KeyError:
            # 예측 누락 → (-1,-1,-1) 로 간주
            s = pd.Series({"Motor axis 0": -1, "Motor axis 1": -1, "Motor axis 2": -1})
        pred_has = not _is_no_motor(s)

        if gt_has and pred_has:
            dist = np.linalg.norm(
                np.array([g["Motor axis 0"], g["Motor axis 1"], g["Motor axis 2"]])
                - np.array([s["Motor axis 0"], s["Motor axis 1"], s["Motor axis 2"]])
            )
            if dist <= tau:
                TP += 1
            else:
                FP += 1
                FN += 1
        elif gt_has and (not pred_has):
            FN += 1
        elif (not gt_has) and pred_has:
            FP += 1
        # else : TN → 점수 계산에서 무시

    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    beta2 = beta ** 2
    fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-9)
    return fbeta
