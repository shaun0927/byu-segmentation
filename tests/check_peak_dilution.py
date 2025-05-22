"""tests/check_peak_dilution.py
Synthetic grid_reconstruct_3d test to confirm peak dilution.
"""

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from torch.nn.functional import sigmoid
from utils import grid_reconstruct_3d

# 작은 패치 2x2x2 를 2x2x2 격자로 배치
roi = (2, 2, 2)
num_patches = 8  # 2x2x2 grid
patches = torch.zeros((num_patches, 1, *roi))

# 첫 번째 패치 한 곳만 강한 신호(1.0)
patches[0, 0, 0, 0, 0] = 1.0

# 각 패치의 위치 (비중복)
locs = []
for z in range(0, 4, 2):
    for y in range(0, 4, 2):
        for x in range(0, 4, 2):
            locs.append((z, y, x))
locs = torch.tensor(locs)

vol_shape = (4, 4, 4)
recon = grid_reconstruct_3d(patches, locs, vol_shape)

print("max value before sigmoid :", recon.max().item())
print("max value after sigmoid  :", sigmoid(recon).max().item())
