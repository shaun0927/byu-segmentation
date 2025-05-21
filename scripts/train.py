#!/usr/bin/env python
"""
scripts/train.py
run:  python scripts/train.py -C configs.cfg_resnet34 --epochs 2
"""
from __future__ import annotations
# ── import path ────────────────────────────────────────────────────
import sys, pathlib, argparse, importlib, time, math, gc
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── std / 3rd party ──────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from monai.inferers import sliding_window_inference

# ── local ─────────────────────────────────────────────────────────
from configs.common_config import get_cfg              # noqa: F401
from data.ds_byu import BYUMotorDataset, simple_collate, load_zarr, norm_patch
from models.net_byu import BYUNet
from postprocess.pp_byu import post_process_volume
# ------------------------------------------------------------------

# ───────── helper : on-the-fly patch generator ────────────────────
def gen_grid_patches(vol: np.ndarray, roi: tuple[int,int,int] = (128,128,128)):
    d, h, w = roi
    D, H, W = vol.shape
    for z0 in range(0, D, d):
        for y0 in range(0, H, h):
            for x0 in range(0, W, w):
                z1, y1, x1 = min(z0+d, D), min(y0+h, H), min(x0+w, W)
                pad = ((0, d-(z1-z0)), (0, h-(y1-y0)), (0, w-(x1-x0)))
                yield np.pad(vol[z0:z1, y0:y1, x0:x1], pad, "reflect"), (z0, y0, x0)

# ───────── CLI ────────────────────────────────────────────────────
ROI = (128,128,128)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-C", "--config", required=True,
                   help="python module path (e.g. configs.cfg_resnet34)")
    p.add_argument("--epochs", type=int)
    return p.parse_args()

# ───────── util : safe centre-crop (aug guard) ────────────────────
def center_crop(t: torch.Tensor, size=ROI):
    d,h,w = t.shape[-3:]
    cd,ch,cw = size
    zs,ys,xs = max((d-cd)//2,0), max((h-ch)//2,0), max((w-cw)//2,0)
    return t[..., zs:zs+cd, ys:ys+ch, xs:xs+cw]

# ───────── constants ───────────────────────────────────────────────
EARLY_ABORT_THRESH = 0.12   # early abort threshold

# ==================================================================
def main():
    args = parse_args()
    cfg = importlib.import_module(args.config).cfg
    if args.epochs:
        cfg.epochs = args.epochs

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── W&B --------------------------------------------------------
    wb = None
    if not cfg.disable_wandb:
        wb = wandb.init(
            project=cfg.project_name,
            name=cfg.exp_name,
            config=vars(cfg),
            mode="online" if cfg.wandb_api else "disabled"
        )

    # ── label & split ---------------------------------------------
    df = (
        pd.read_csv(cfg.label_csv)
          .query("`Number of motors` <= 1")
          .drop_duplicates("tomo_id")
          .set_index("tomo_id")
    )
    ids = df.index.tolist()
    cut = math.ceil(0.8 * len(ids))
    tr_ids, val_ids = ids[:cut], ids[cut:]

    # ── DataLoader for training -----------------------------------
    tr_dl = DataLoader(
        BYUMotorDataset(tr_ids, mode="train", root=cfg.data_root),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=simple_collate,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers
    )

    # ── model / optimizer / scheduler -----------------------------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = BYUNet(cfg.__dict__).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(tr_dl)*cfg.epochs)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(getattr(cfg, "class_weights", [1,1]),
                             dtype=torch.float32, device=dev)
    )
    scaler = GradScaler()

    # ── consts (validation) --------------------------------------
    VAL_BS     = getattr(cfg, "val_patch_bs", 16)
    TH_VOX     = getattr(cfg, "match_thresh_vox", 5)
    ACC_DTYPE  = torch.float16 if getattr(cfg, "val_acc_fp16", True) else torch.float32
    FWD_CHUNK  = min(VAL_BS, 16)

    # ==============================================================
    for ep in range(cfg.epochs):
        # ---------- TRAIN ----------------------------------------
        net.train(); ep_loss = 0; t0 = time.time()
        for step, batch in enumerate(
            tqdm(tr_dl, desc=f"Epoch {ep+1}/{cfg.epochs}", leave=False), 1):

            imgs = batch["image"].to(dev, non_blocking=True)
            lbls = batch["label"].squeeze(1).long().to(dev, non_blocking=True)
            if imgs.shape[-3:] != ROI:
                imgs = center_crop(imgs); lbls = center_crop(lbls)

            losses = []
            for s in range(0, imgs.size(0), FWD_CHUNK):
                with autocast(dtype=torch.float16):
                    logit = net({"image": imgs[s:s+FWD_CHUNK]})["logits"]
                    losses.append(loss_fn(logit, lbls[s:s+FWD_CHUNK]))
            loss = torch.stack(losses).mean()

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            ep_loss += loss.item()

            if step % getattr(cfg, "log_every", 20) == 0:
                print(f"Ep{ep+1} step{step}/{len(tr_dl)} loss {loss.item():.4f}")
                if wb:
                    wb.log({"loss/step": loss.item()}, step=ep*len(tr_dl)+step)

        sch.step()
        if wb:
            wb.log({"loss/train": ep_loss/len(tr_dl), "lr": sch.get_last_lr()[0]}, step=ep)
        torch.cuda.empty_cache(); gc.collect()

        # ---------- VALIDATION -----------------------------------
        net.eval(); tp = fp = fn = 0
        for tid in tqdm(val_ids, desc="Valid-vol", leave=False):
            # 1) load full volume into memory once
            zarr_arr = load_zarr(pathlib.Path(cfg.data_root)/"train"/f"{tid}.zarr")
            vol_np = zarr_arr[:]  # (D,H,W)

            # 2) convert to tensor: (1,1,D,H,W)
            vol_tensor = torch.from_numpy(vol_np[None,None].astype(np.float32)).to(dev)

            # 3) sliding window inference
            with torch.no_grad():
                sw_out = sliding_window_inference(
                    inputs=vol_tensor,
                    roi_size=ROI,
                    sw_batch_size=VAL_BS,
                    predictor=lambda x: net({"image": x})["logits"].softmax(1)[:,1:2],
                    overlap=0.0
                )  # (1,1,D,H,W)
            full_prob = sw_out.squeeze().cpu()  # (D,H,W)

            # 4) post-process & evaluate
            df_pred = post_process_volume(full_prob, spacing=10.0, tomo_id=tid)
            row = df.loc[tid]
            pred_has = (df_pred.iloc[0][1:4] >= 0).all()
            gt_has   = row["Number of motors"] > 0
            if gt_has and pred_has:
                dist = np.linalg.norm(
                    df_pred.iloc[0][1:4].values -
                    row[["Motor axis 0","Motor axis 1","Motor axis 2"]].values
                )/10.0
                if dist < TH_VOX: tp += 1
                else: fp += 1; fn += 1
            elif gt_has:
                fn += 1
            elif pred_has:
                fp += 1

        # metrics
        prec = tp/(tp+fp+1e-6); rec = tp/(tp+fn+1e-6)
        f2   = (1+cfg.beta**2)*prec*rec/(cfg.beta**2*prec+rec+1e-6)
        dt   = time.time() - t0
        print(f"[{ep+1}/{cfg.epochs}] loss {ep_loss/len(tr_dl):.4f} "
              f"TP {tp} FP {fp} FN {fn} F2 {f2:.3f}  {dt/60:.1f} min")
        if wb:
            wb.log({"TP":tp,"FP":fp,"FN":fn,
                    "precision":prec,"recall":rec,"F2/val":f2}, step=ep)

    if wb:
        wb.finish()

if __name__ == "__main__":
    main()