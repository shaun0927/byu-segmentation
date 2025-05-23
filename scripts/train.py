#!/usr/bin/env python
"""
BYU Motor – Training Script  (val-freq 지원 버전)

Features:
- Dataset ROI = 96³
- BalancedSampler for 6:2 pos:neg tomogram ratio
- BYUNet-internal loss (CEPlus) usage
- Partial validation (--quick-val) with 10 positive + 10 negative samples
- Spacing-aware post-processing
- NEW: --val-freq 로 특정 에폭마다만 validation 실행

Usage:
    python scripts/train.py -C configs.cfg_resnet34 --epochs 10             # 매 에폭마다 검증
    python scripts/train.py -C configs.cfg_resnet34 --epochs 30 --val-freq 3  # 3에폭마다 검증
"""
from __future__ import annotations
import sys, pathlib, argparse, time, math, gc, importlib, random
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch, wandb
from torch.utils.data import DataLoader, Sampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from skimage.util import view_as_blocks

# local imports
from data.ds_byu import BYUMotorDataset, simple_collate, load_zarr, ROI  # ROI = (96,96,96)
from models.net_byu import BYUNet
from postprocess.pp_byu import post_process_volume, THRESH
# ------------------------------------------------------------------------------

VAL_BS    = 16        # validation batch size
TH_VOX    = 100       # TP distance threshold (voxels)
ACC_DTYPE = np.float32
MAX_CHUNK = 8         # gradient split size
PAD_MODE  = "reflect"

# ----------------------------------------------------------------------------
class BalancedSampler(Sampler):
    """
    Finite sampler: 한 에폭에 (kp+kn) * n_batches 만큼만 인덱스 반환
    kp = 양성, kn = 음성 샘플 수
    """
    def __init__(self, pos_idx, neg_idx, *, k_pos=6, k_neg=2, seed=42):
        self.pos, self.neg = list(pos_idx), list(neg_idx)
        self.kp, self.kn   = k_pos, k_neg
        random.seed(seed)

        # 에폭당 배치 수 = 두 클래스 모두 고갈되지 않는 최소치
        self.n_batches = min(len(self.pos)//self.kp, len(self.neg)//self.kn)

        # 에폭당 총 샘플 = n_batches * (kp+kn)
        self._length = self.n_batches * (self.kp + self.kn)

    # ───────── finite iterator ─────────
    def __iter__(self):
        pos_pool = self.pos.copy()
        neg_pool = self.neg.copy()
        random.shuffle(pos_pool); random.shuffle(neg_pool)

        for _ in range(self.n_batches):
            batch = [pos_pool.pop() for _ in range(self.kp)] + \
                    [neg_pool.pop() for _ in range(self.kn)]
            random.shuffle(batch)
            yield from batch

    def __len__(self):
        return self._length          # DataLoader가 정확히 인식

# ----------------------------------------------------------------------------

def center_crop(t: torch.Tensor, size=ROI) -> torch.Tensor:
    """Center-crop last three spatial dims to size"""
    d, h, w = t.shape[-3:]
    cd, ch, cw = size
    zs, ys, xs = (d-cd)//2, (h-ch)//2, (w-cw)//2
    return t[..., zs:zs+cd, ys:ys+ch, xs:xs+cw]


def make_block_view(vol: np.ndarray):
    """Split volume into non-overlap ROI cubes + their top-left coords"""
    D, H, W = vol.shape
    pad = [(ROI[i] - dim % ROI[i]) % ROI[i] for i, dim in enumerate((D, H, W))]
    vol_p = np.pad(vol, ((0,pad[0]),(0,pad[1]),(0,pad[2])), mode=PAD_MODE)

    blocks = view_as_blocks(vol_p, block_shape=ROI)
    nz, ny, nx = blocks.shape[:3]
    patches = blocks.reshape(-1, *ROI)
    coords = [(iz*ROI[0], iy*ROI[1], ix*ROI[2])
              for iz in range(nz) for iy in range(ny) for ix in range(nx)]
    return patches, coords, vol_p.shape, tuple(pad)

# ----------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-C", "--config", required=True,
                    help="Python config module (e.g. configs.cfg_resnet34)")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--quick-val", action="store_true",
                    help="Use partial validation: 10 pos + 10 neg tomograms")
    ap.add_argument("--val-freq", type=int, default=1,
                    help="Run validation every N epochs (≥1).")
    ap.add_argument("--save-ckpt", type=str, default=None,
                    help="Checkpoint prefix; if set, '{prefix}_epN.pt' files are saved")
    ap.add_argument("--dist-weight", type=float, default=None,
                    help="Distance weight for post-processing")
    ap.add_argument("--expected-max-dist", type=float, default=None,
                    help="Override expected max distance in Å")
    ap.add_argument("--score-mode", type=str, default="exp",
                    choices=["linear", "exp"],
                    help="Score combination mode")
    return ap.parse_args()

# ============================================================================

def main():
    args = parse_args()
    # Load configuration
    cfg = importlib.import_module(args.config).cfg
    cfg.epochs = args.epochs

    # Seed and CUDA settings
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    # Initialize wandb
    wb = None
    if not cfg.disable_wandb:
        wb = wandb.init(
            project=cfg.project_name,
            name=cfg.exp_name,
            config=vars(cfg),
            mode="online" if cfg.wandb_api else "disabled"
        )

    # Load labels, split train/val
    df_all = (
        pd.read_csv(cfg.label_csv)
          .query("`Number of motors` <= 1")
          .drop_duplicates("tomo_id")
          .set_index("tomo_id")
    )
    all_ids = df_all.index.tolist()
    cut = math.ceil(0.8 * len(all_ids))
    tr_ids, val_ids = all_ids[:cut], all_ids[cut:]

    # Optional quick validation subset
    if args.quick_val:
        random.seed(cfg.seed)
        pos = [tid for tid in val_ids if df_all.loc[tid, "Number of motors"] > 0]
        neg = [tid for tid in val_ids if df_all.loc[tid, "Number of motors"] == 0]
        sel = random.sample(pos, min(10, len(pos))) + random.sample(neg, min(10, len(neg)))
        random.shuffle(sel)
        val_ids = sel
        print(f"== Quick validation: {len(val_ids)} tomograms ==")

    # Prepare training dataset and sampler (int indices)
    train_ds = BYUMotorDataset(tr_ids, mode="train", root=cfg.data_root)
    pos_idx = [i for i, tid in enumerate(train_ds.ids)
               if df_all.loc[tid, "Number of motors"] > 0]
    neg_idx = [i for i, tid in enumerate(train_ds.ids)
               if df_all.loc[tid, "Number of motors"] == 0]
    sampler = BalancedSampler(pos_idx, neg_idx)

    # Training DataLoader
    tr_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=simple_collate,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        drop_last=True,  # ensure 6+2 per batch
    )

    # Model, optimizer, scheduler, scaler
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = BYUNet(cfg.__dict__).to(dev)
    opt = torch.optim.AdamW(
        net.parameters(), lr=cfg.lr,
        weight_decay=getattr(cfg, "weight_decay", 0.0)
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=len(tr_dl) * cfg.epochs
    )
    scaler = GradScaler()

    # ----------------- Training Loop -----------------
    for ep in range(cfg.epochs):
        net.train()
        ep_loss = 0.0
        for step, batch in enumerate(tqdm(tr_dl, desc=f"Epoch {ep+1}/{cfg.epochs}"), 1):
            # Move tensors to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(dev, non_blocking=True)

            # Safety center-crop if shape mismatch
            if batch["image"].shape[-3:] != ROI:
                batch["image"] = center_crop(batch["image"], ROI)
                batch["label"] = center_crop(batch["label"], ROI)

            # Chunked forward / backward
            loss_chunks: list[torch.Tensor] = []
            B = batch["image"].size(0)
            for s in range(0, B, MAX_CHUNK):
                slice_b = {k: (v[s:s+MAX_CHUNK] if isinstance(v, torch.Tensor) else v)
                           for k, v in batch.items()}
                with autocast():
                    out = net(slice_b)
                    loss_chunks.append(out["loss"])
            loss = torch.stack(loss_chunks).mean()

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ep_loss += loss.item()

            if step % getattr(cfg, "log_every", 42) == 0:
                print(f"Epoch {ep+1} Step {step}/{len(tr_dl)}  loss {loss.item():.4f}")
                if wb:
                    wb.log({"loss/step": loss.item()}, step=ep*len(tr_dl)+step)

        sch.step()
        if wb:
            wb.log({
                "loss/train": ep_loss / len(tr_dl),
                "lr": sch.get_last_lr()[0]
            }, step=ep)
        if args.save_ckpt:
            ckpt_base = pathlib.Path(args.save_ckpt)
            ckpt_base.parent.mkdir(parents=True, exist_ok=True)
            ckpt_fn = ckpt_base.with_name(f"{ckpt_base.stem}_ep{ep+1}.pt")
            torch.save({"epoch": ep + 1, "model": net.state_dict()}, ckpt_fn)
            print(f"[INFO] saved checkpoint to {ckpt_fn}")
        torch.cuda.empty_cache(); gc.collect()

        # ----------------- Validation (conditional) ------------------
        run_val = ((ep + 1) % args.val_freq == 0) or (ep == cfg.epochs - 1)
        if not run_val:
            continue

        net.eval()
        tp = fp = fn = 0
        t0 = time.time()
        desc = "Q-Val" if args.quick_val else "Validation"
        for tid in tqdm(val_ids, desc=desc, leave=False):
            vol_np = load_zarr(pathlib.Path(cfg.data_root)/"train"/f"{tid}.zarr")[:]
            D, H, W = vol_np.shape
            patches, coords, (Dp, Hp, Wp), _ = make_block_view(vol_np)

            acc = np.zeros((Dp, Hp, Wp), dtype=ACC_DTYPE)
            cnt = np.zeros_like(acc)

            for i in range(0, len(patches), VAL_BS):
                sl = slice(i, i+VAL_BS)
                bp = patches[sl].astype(np.float32)
                bp = (bp - bp.mean((1,2,3), keepdims=True)) / (bp.std((1,2,3), keepdims=True) + 1e-6)
                bt = torch.from_numpy(bp[:, None]).to(dev, non_blocking=True)
                with torch.no_grad(), autocast():
                    logits = net({"image": bt})["logits"]        # shape (B,1,D,H,W)
                    prob = torch.sigmoid(logits).squeeze(1)      # shape (B,D,H,W)
                prob = prob.cpu().numpy()
                for p, (z0, y0, x0) in zip(prob, coords[sl]):
                    acc[z0:z0+ROI[0], y0:y0+ROI[1], x0:x0+ROI[2]] += p
                    cnt[z0:z0+ROI[0], y0:y0+ROI[1], x0:x0+ROI[2]] += 1

            full_prob = acc / np.clip(cnt, 1, None)
            full_prob = full_prob[:D, :H, :W]

            spacing = df_all.loc[tid, "Voxel spacing"]
            exp_max = TH_VOX * spacing
            if args.expected_max_dist is not None:
                exp_max = args.expected_max_dist

            row = df_all.loc[tid]
            gt_coord = None
            if row["Number of motors"] > 0:
                gt_coord = (
                    row[["Motor axis 0", "Motor axis 1", "Motor axis 2"]]
                    .astype(float)
                    .values
                    * spacing
                )

            df_pred = post_process_volume(
                full_prob,
                spacing=spacing,
                tomo_id=tid,
                expected_max_dist=exp_max,
                gt_coord=gt_coord,
                dist_weight=args.dist_weight,
                score_mode=args.score_mode,
            )

            conf_val = df_pred["conf"].iloc[0]
            coord_val = df_pred.iloc[0, 1:4].values
            print(
                f"[{tid}] conf={conf_val:.3f}, coord={coord_val.tolist()}, THRESH={THRESH}"
            )

            pred_has = (df_pred.iloc[0, 1:4] >= 0).all()
            gt_has   = row["Number of motors"] > 0
            if gt_has and pred_has:
                # predicted 좌표는 Angstrom 단위이므로 voxel 단위로 맞춰 비교
                dist = np.linalg.norm(
                    df_pred.iloc[0, 1:4].values / spacing -
                    row[["Motor axis 0","Motor axis 1","Motor axis 2"]].values
                )
                print(f"[{tid}] distance to GT = {dist:.1f} vox")
                if dist <= TH_VOX:
                    tp += 1
                else:
                    fp += 1; fn += 1
            elif gt_has:
                fn += 1
            elif pred_has:
                fp += 1

        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f2   = (1 + cfg.beta**2) * prec * rec / (cfg.beta**2 * prec + rec + 1e-9)
        print(f"[{ep+1}/{cfg.epochs}] TP {tp} FP {fp} FN {fn} "
              f"F2 {f2:.3f}  val_time {(time.time()-t0)/60:.1f} min")
        if wb:
            wb.log({
                "TP": tp, "FP": fp, "FN": fn,
                "precision": prec, "recall": rec, "F2/val": f2
            }, step=ep)

    if wb:
        wb.finish()

if __name__ == "__main__":
    main()
