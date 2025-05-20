# data/build_zarr.py
"""
JPEG stack → chunked Zarr (uint8, default chunks 64×256×256)
------------------------------------------------------------
python -m data.build_zarr \
    --raw_root  data/raw \
    --out_root  data/processed \
    --workers 4 --chunk 64 256 256 --overwrite
"""
from __future__ import annotations
import argparse, os, re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List

import imageio, numpy as np, zarr
from tqdm import tqdm

# silence imageio
imageio.v2.IMAGEIO_NO_WARNINGS = True
read_jpg = imageio.v2.imread

# ─── helpers ──────────────────────────────────────────────────────
def natural_key(s: str) -> List[int | str]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def percentile_norm(x: np.ndarray,
                    pl: float = .5, ph: float = 99.5) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo, hi = np.percentile(x, (pl, ph))
    np.clip(x, lo, hi, out=x)
    x -= lo
    x *= 255.0 / (hi - lo + 1e-7)
    return x.astype(np.uint8, copy=False)

def read_slice(fp: Path) -> np.ndarray:
    im = read_jpg(fp)
    return im[..., 0] if im.ndim == 3 else im   # RGB→gray

def fit_slice(src: np.ndarray, H: int, W: int) -> np.ndarray:
    """zero-pad or center-crop to (H,W)"""
    h, w = src.shape
    if (h, w) == (H, W):
        return src
    canvas = np.zeros((H, W), dtype=np.uint8)
    h0, w0 = max(0, (H - h) // 2), max(0, (W - w) // 2)
    h1, w1 = h0 + min(h, H), w0 + min(w, W)
    canvas[h0:h1, w0:w1] = src[:h1 - h0, :w1 - w0]
    return canvas

# ─── worker function ─────────────────────────────────────────────
def convert_one(jpg_dir: Path, out_fn: Path,
                chunk: Tuple[int, int, int], overwrite: bool) -> str:

    if out_fn.exists() and not overwrite:
        return f"skip – exists {out_fn.name}"

    jpgs = sorted(jpg_dir.glob("*.jpg"), key=lambda p: natural_key(p.name))
    if not jpgs:
        return f"[WARN] no jpg in {jpg_dir}"

    # baseline shape
    first = read_slice(jpgs[0]).astype(np.uint8, copy=False)
    H, W = first.shape
    D = len(jpgs)
    vol = np.empty((D, H, W), dtype=np.uint8)
    vol[0] = first

    for k, fp in enumerate(jpgs[1:], 1):
        sl = fit_slice(read_slice(fp).astype(np.uint8, copy=False), H, W)
        vol[k] = sl

    vol = percentile_norm(vol)
    vol = np.ascontiguousarray(vol, dtype=np.uint8)

    # ---- save with explicit dtype (avoid object_codec bug)
    store = zarr.open(out_fn, mode="w")
    store.create_dataset(
        name="image",
        data=vol,
        chunks=chunk,
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=1),
        dtype="uint8",
        overwrite=True,
    )
    return f"✓ {jpg_dir.name:<14} → {vol.shape} chunks{chunk}"

def _star(args):
    return convert_one(*args)

# ─── CLI / main ──────────────────────────────────────────────────
def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root",  required=True)
    ap.add_argument("--out_root",  required=True)
    ap.add_argument("--workers",   type=int, default=os.cpu_count())
    ap.add_argument("--chunk",     nargs=3, type=int,
                    default=(64,256,256),
                    metavar=("Z","Y","X"))      # ← 괄호 수정
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()

def main() -> None:
    args = parse()
    raw_root  = Path(args.raw_root).resolve()
    out_root  = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[Path, Path, tuple[int,int,int], bool]] = []
    for split in ("train", "test"):
        for tomo in sorted((raw_root / split).glob("*")):
            if tomo.is_dir():
                out_fn = (out_root / split).joinpath(f"{tomo.name}.zarr")
                out_fn.parent.mkdir(exist_ok=True)
                jobs.append((tomo, out_fn, tuple(args.chunk), args.overwrite))

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for msg in tqdm(ex.map(_star, jobs), total=len(jobs)):
            print(msg)
    print("All done.")

if __name__ == "__main__":
    main()
