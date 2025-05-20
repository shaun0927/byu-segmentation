"""
GT == pred  →  F2 = 1.0 인지 확인
"""

import pandas as pd
from metrics.metric_f2 import f2_score

gt_df = pd.DataFrame(
    [
        ["tomo_A", 150.0, 300.0, 450.0],   # motor 존재
        ["tomo_B", -1.0,  -1.0,  -1.0],    # motor 없음
    ],
    columns=["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"],
)

# 예측을 GT 와 동일하게 만듦
pred_df = gt_df.copy()

score = f2_score(gt_df, pred_df, tau=1000.0)
print("F2 :", score)
assert abs(score - 1.0) < 1e-6, "GT == pred 인데 F2 가 1 이 아닙니다!"
print("✓  test passed")
