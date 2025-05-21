"""
utils.py
────────────────────────────────────────────────────────
공통 유틸리티
  • gaussian_kernel_3d : 3-D Gaussian heat-map 생성
  • natural_key        : slice 파일의 human-friendly sort
  • open_zarr_lazy     : ✨ zarr 배열을 lazy 로 열기 (RAM 점유 X)
  • grid_split_3d      : 3-D 볼륨 → non-overlap 패치 + 위치
  • grid_reconstruct_3d: 패치들을 원본 좌표계로 가중평균 복원
────────────────────────────────────────────────────────
"""

from __future__ import annotations
from typing import List, Any, Tuple

import numpy as np
import torch
import re
import zarr, os


# ──────────────────────────────────────────────────────
def gaussian_kernel_3d(
    shape: Tuple[int, int, int],
    center: Tuple[int, int, int],
    sigma: float,
    cutoff: float = 0.05,
) -> np.ndarray:
    """
    spacing-aware 3-D Gaussian (float32). cutoff 미만 값은 0 으로 희소화.
    """
    zc, yc, xc = center
    D, H, W = shape

    zz, yy, xx = np.ogrid[:D, :H, :W]
    dist2 = (
        (zz - zc) ** 2 +
        (yy - yc) ** 2 +
        (xx - xc) ** 2
    ).astype(np.float32)

    g = np.exp(-0.5 * dist2 / (sigma ** 2)).astype(np.float32, copy=False)
    if cutoff > 0:
        g[g < cutoff] = 0.0
    return g


# ──────────────────────────────────────────────────────
def natural_key(p: str) -> List[Any]:
    """'slice_1', 'slice_10', … 와 같은 문자열을 자연스러운 순서로 정렬하기 위한 key"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p)]


# ──────────────────────────────────────────────────────
#  ✨  Zarr helper -- RAM 에 올리지 않고 shape·슬라이스만 사용
# ──────────────────────────────────────────────────────
def open_zarr_lazy(path: str | "os.PathLike"):
    """
    Parameters
    ----------
    path : str | Path
        '<tomo>.zarr' store 경로

    Returns
    -------
    zarr.Array
        메타 데이터만 로드된 *lazy array* 객체
        (실제 chunk 는 슬라이스할 때까지 디스크에 머무른다)
    """
    return zarr.open(str(path), mode="r")


# ──────────────────────────────────────────────────────
#  Grid  split / reconstruct (non-overlap, reflect-pad)
# ──────────────────────────────────────────────────────
def grid_split_3d(
    arr: np.ndarray,
    roi: Tuple[int, int, int],
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    Parameters
    ----------
    arr : np.ndarray (D,H,W)
    roi : (d,h,w) 패치 크기

    Returns
    -------
    patches : list[np.ndarray]  –  reflect-padding 으로 항상 roi 크기
    locs    : list[tuple[int,int,int]]  –  각 패치의 (z0,y0,x0) 좌상단
    """
    d, h, w = roi
    D, H, W = arr.shape

    patches, locs = [], []
    for z0 in range(0, D, d):
        for y0 in range(0, H, h):
            for x0 in range(0, W, w):
                z1, y1, x1 = min(z0 + d, D), min(y0 + h, H), min(x0 + w, W)

                # 경계 초과분 → reflect-pad
                pad = (
                    (0, d - (z1 - z0)),
                    (0, h - (y1 - y0)),
                    (0, w - (x1 - x0)),
                )
                patches.append(np.pad(arr[z0:z1, y0:y1, x0:x1], pad, mode="reflect"))
                locs.append((z0, y0, x0))
    return patches, locs


def grid_reconstruct_3d(
    patches: torch.Tensor,           # (N, C, d, h, w)
    locs:    torch.Tensor,           # (N, 3)  (z0,y0,x0)
    vol_shape: Tuple[int, int, int], # (D,H,W)  원본 볼륨
) -> torch.Tensor:
    """
    패치 확률(또는 feature)들을 원본 좌표계로 복원한다.
    겹치는 위치는 *가중 평균*.

    Returns
    -------
    torch.Tensor  –  (C, D, H, W)
    """
    C, d, h, w = patches.shape[1:]
    D, H, W    = vol_shape

    out   = torch.zeros((C, D, H, W), dtype=patches.dtype, device=patches.device)
    count = torch.zeros_like(out)

    for patch, (z0, y0, x0) in zip(patches, locs):
        z0, y0, x0 = map(int, (z0, y0, x0))
        dz, dy, dx = min(d, D - z0), min(h, H - y0), min(w, W - x0)

        out[:,  z0:z0+dz, y0:y0+dy, x0:x0+dx] += patch[:, :dz, :dy, :dx]
        count[:, z0:z0+dz, y0:y0+dy, x0:x0+dx] += 1

    return out / count.clamp_min(1)



import time, contextlib, torch


@contextlib.contextmanager
def elapsed(msg: str):
    torch.cuda.synchronize()          # GPU 작업 종료 대기
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"[DBG] {msg:<26} : {dt*1000:.1f} ms")