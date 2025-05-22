import pandas as pd
import numpy as np

TH_VOX = 100  # validation threshold in voxels

# train_labels.csv 에서 모터가 있는 첫 번째 tomogram 선택
labels = pd.read_csv("data/raw/train_labels.csv")
row = labels[labels["Number of motors"] > 0].iloc[0]

spacing = row["Voxel spacing"]

tid = row["tomo_id"]
# GT 좌표는 이미 voxel 단위
coord_gt_vox = row[["Motor axis 0", "Motor axis 1", "Motor axis 2"]].values.astype(float)

# 예측 결과를 GT 좌표와 동일하게 만든 후 Å 단위로 변환했다고 가정
coord_pred_A = coord_gt_vox * spacing
pred_df = pd.DataFrame([[tid, *coord_pred_A]],
                       columns=["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"])

# 검증 단계에서 사용되는 voxel 변환 및 거리 계산식
coord_pred_vox = pred_df.iloc[0, 1:4].values / spacing

dist = np.linalg.norm(coord_pred_vox - coord_gt_vox)
print("distance =", dist)

assert dist <= TH_VOX, f"dist {dist} > {TH_VOX}"
print("✓  test passed")
