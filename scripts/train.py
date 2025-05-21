#!/usr/bin/env python
"""
scripts/train.py
run:  python scripts/train.py -C configs.cfg_resnet34 --epochs 1
"""
from __future__ import annotations
import sys, pathlib, argparse, time, math, gc, importlib

# ---------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch, wandb
from torch.utils.data import DataLoader
from torch.cuda.amp   import autocast, GradScaler
from tqdm             import tqdm
from skimage.util     import view_as_blocks

# local ----------------------------------------------------------
from configs.common_config import get_cfg
from data.ds_byu          import BYUMotorDataset, simple_collate, load_zarr
from models.net_byu       import BYUNet
from postprocess.pp_byu   import post_process_volume
# ---------------------------------------------------------------

ROI          = (128, 128, 128)
VAL_BS       = 16
TH_VOX       = 100
ACC_DTYPE    = np.float32
MAX_CHUNK    = 8
PAD_MODE     = "reflect"

# -------------------------- helpers ----------------------------
def center_crop(t: torch.Tensor, size=ROI):
    d, h, w = t.shape[-3:]
    cd, ch, cw = size
    zs, ys, xs = (d-cd)//2, (h-ch)//2, (w-cw)//2
    return t[..., zs:zs+cd, ys:ys+ch, xs:xs+cw]

def make_block_view(vol: np.ndarray):
    """pad → stride-view → (N,128,128,128) + 좌표"""
    D, H, W = vol.shape
    pad_d = (ROI[0] - D % ROI[0]) % ROI[0]
    pad_h = (ROI[1] - H % ROI[1]) % ROI[1]
    pad_w = (ROI[2] - W % ROI[2]) % ROI[2]
    vol_p = np.pad(vol,
                   ((0, pad_d), (0, pad_h), (0, pad_w)),
                   mode=PAD_MODE)

    blocks = view_as_blocks(vol_p, block_shape=ROI)          # (nz,ny,nx,128³)
    nz, ny, nx = blocks.shape[:3]
    patches = blocks.reshape(-1, *ROI)                       # (N,128³)

    coords = [(iz*ROI[0], iy*ROI[1], ix*ROI[2])
              for iz in range(nz)
              for iy in range(ny)
              for ix in range(nx)]
    return patches, coords, vol_p.shape, (pad_d, pad_h, pad_w)

# -------------------------- CLI -------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-C", "--config", required=True,
                   help="python module path (e.g. configs.cfg_resnet34)")
    p.add_argument("--epochs", type=int)
    return p.parse_args()

# ========================== main ===============================
def main():
    args = parse_args()
    cfg  = importlib.import_module(args.config).cfg
    if args.epochs:
        cfg.epochs = args.epochs

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    wb = None
    if not cfg.disable_wandb:
        wb = wandb.init(
            project=cfg.project_name,
            name    =cfg.exp_name,
            config  =vars(cfg),
            mode    ="online" if cfg.wandb_api else "disabled"
        )

    # ---------- label & split ----------
    df_all = (
        pd.read_csv(cfg.label_csv)
          .query("`Number of motors` <= 1")
          .drop_duplicates("tomo_id")
          .set_index("tomo_id")
    )
    ids   = df_all.index.tolist()
    cut   = math.ceil(0.8*len(ids))
    tr_ids, val_ids = ids[:cut], ids[cut:]

    # ---------- DataLoader (train) -----
    tr_dl = DataLoader(
        BYUMotorDataset(tr_ids, mode="train", root=cfg.data_root),
        batch_size         = cfg.batch_size,
        shuffle            = True,
        num_workers        = cfg.num_workers,
        collate_fn         = simple_collate,
        pin_memory         = cfg.pin_memory,
        persistent_workers = cfg.persistent_workers,
    )

    # ---------- model / optim ----------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = BYUNet(cfg.__dict__).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr,
                            weight_decay=getattr(cfg,'weight_decay',0.0))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=len(tr_dl)*cfg.epochs)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(getattr(cfg,"class_weights",[1,1]),
                            dtype=torch.float32, device=dev))
    scaler  = GradScaler()

    # ---------------- training loop -----------------
    for ep in range(cfg.epochs):
        net.train()
        ep_loss = 0.0
        for step, batch in enumerate(
                tqdm(tr_dl, desc=f"Epoch {ep+1}/{cfg.epochs}", leave=False), 1):

            imgs = batch["image"].to(dev, non_blocking=True)
            lbls = batch["label"].squeeze(1).long().to(dev, non_blocking=True)
            if imgs.shape[-3:] != ROI:
                imgs = center_crop(imgs); lbls = center_crop(lbls)

            losses = []
            for s in range(0, imgs.size(0), MAX_CHUNK):
                with autocast():
                    out = net({"image": imgs[s:s+MAX_CHUNK]})["logits"]
                    losses.append(loss_fn(out, lbls[s:s+MAX_CHUNK]))
            loss = torch.stack(losses).mean()

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            ep_loss += loss.item()

            if step % getattr(cfg, "log_every", 20) == 0:
                print(f"Ep{ep+1} step{step}/{len(tr_dl)}  loss {loss.item():.4f}")
                if wb:
                    wb.log({"loss/step": loss.item()},
                           step=ep*len(tr_dl)+step)

        sch.step()
        if wb:
            wb.log({"loss/train": ep_loss / len(tr_dl),
                    "lr": sch.get_last_lr()[0]}, step=ep)
        torch.cuda.empty_cache(); gc.collect()

        # ---------------- validation -----------------
        net.eval(); tp = fp = fn = 0
        t0_val = time.time()

        for tid in tqdm(val_ids, desc="Valid-vol", leave=False):
            # 1. load volume (once)
            vol_np = load_zarr(pathlib.Path(cfg.data_root) /
                               "train" / f"{tid}.zarr")[:]
            D, H, W = vol_np.shape

            # 2. block-view
            patches, coords, (Dp, Hp, Wp), pads = make_block_view(vol_np)

            # 3. accumulation buffers (CPU)
            acc = np.zeros((Dp, Hp, Wp), dtype=ACC_DTYPE)
            cnt = np.zeros_like(acc)

            # 4. batched inference
            for i in range(0, len(patches), VAL_BS):
                idx    = slice(i, i+VAL_BS)
                batch  = patches[idx].astype(np.float32)
                batch  = (batch - batch.mean((1,2,3), keepdims=True)) \
                       / (batch.std ((1,2,3), keepdims=True) + 1e-6)
                batch_t = torch.from_numpy(batch[:, None]).to(dev, non_blocking=True)

                with torch.no_grad(), autocast():
                    prob = net({"image": batch_t})["logits"].softmax(1)[:, 1]
                prob = prob.cpu().numpy()

                for p, (z0, y0, x0) in zip(prob, coords[idx]):
                    acc[z0:z0+128, y0:y0+128, x0:x0+128] += p
                    cnt[z0:z0+128, y0:y0+128, x0:x0+128] += 1

            full_prob = acc / np.clip(cnt, 1, None)
            d_pad, h_pad, w_pad = pads
            full_prob = full_prob[:D, :H, :W]          # crop to original size

            # 5. post-process & metric
            df_pred = post_process_volume(full_prob, spacing=10.0, tomo_id=tid)

            row = df_all.loc[tid]
            pred_has = (df_pred.iloc[0][1:4] >= 0).all()
            gt_has   = row["Number of motors"] > 0
            if gt_has and pred_has:
                dist = np.linalg.norm(
                        df_pred.iloc[0][1:4].values -
                        row[["Motor axis 0","Motor axis 1","Motor axis 2"]].values
                ) / 10.0
                if dist < TH_VOX:
                    tp += 1
                else:
                    fp += 1; fn += 1
            elif gt_has:
                fn += 1
            elif pred_has:
                fp += 1

            del acc, cnt, full_prob
            torch.cuda.empty_cache()

        prec = tp / (tp + fp + 1e-6)
        rec  = tp / (tp + fn + 1e-6)
        f2   = (1 + cfg.beta**2) * prec * rec / (cfg.beta**2 * prec + rec + 1e-6)
        dt   = (time.time() - t0_val) / 60.0
        print(f"[{ep+1}/{cfg.epochs}] TP {tp} FP {fp} FN {fn} "
              f"F2 {f2:.3f}  val_time {dt:.1f} min")

        if wb:
            wb.log({"TP": tp, "FP": fp, "FN": fn,
                    "precision": prec, "recall": rec, "F2/val": f2},
                   step=ep)

    if wb:
        wb.finish()

# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
