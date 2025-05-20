#!/usr/bin/env python
"""
scripts/train.py
────────────────────────────────────────────────────────
예시
    python scripts/train.py -C configs.cfg_resnet34 --epochs 2
"""
from __future__ import annotations

# ── repo root 경로를 PYTHONPATH 에 추가 ─────────────────
import sys, pathlib, argparse, importlib, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── 표준 / 3rd-party ───────────────────────────────────
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb

# ── local ─────────────────────────────────────────────
from configs.common_config import get_cfg
from data.ds_byu          import BYUMotorDataset, simple_collate, load_zarr
from models.net_byu       import BYUNet
from utils                import grid_reconstruct_3d
from postprocess.pp_byu   import post_process_volume
from metrics.metric_f2    import f2_score
# ──────────────────────────────────────────────────────


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-C", "--config", required=True,
                    help="python module path (e.g. configs.cfg_resnet34)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override epochs in config")
    return ap.parse_args()


# ==================================================================== #
#                               main                                   #
# ==================================================================== #
def main():
    args = parse_args()

    # ── config 로드 ---------------------------------------------------
    cfg_mod = importlib.import_module(args.config)
    cfg     = cfg_mod.cfg
    if args.epochs:
        cfg.epochs = args.epochs

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── W&B -----------------------------------------------------------
    wandb_run = None
    if not cfg.disable_wandb:
        wandb_run = wandb.init(
            project = cfg.project_name,
            name    = cfg.exp_name,
            config  = vars(cfg),
            mode    = "online" if cfg.wandb_api else "disabled",
        )

    # ── Dataset -------------------------------------------------------
    tids      = pd.read_csv(cfg.label_csv)["tomo_id"].unique()
    split_pt  = int(0.8 * len(tids))
    train_ds  = BYUMotorDataset(tids[:split_pt],  mode="train", root=cfg.data_root)
    val_ds    = BYUMotorDataset(tids[split_pt:], mode="val",   root=cfg.data_root)

    dl_kw = dict(
        num_workers        = cfg.num_workers,
        collate_fn         = simple_collate,
        pin_memory         = cfg.pin_memory,
        persistent_workers = cfg.persistent_workers,
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,
                          shuffle=True,  **dl_kw)
    val_dl   = DataLoader(val_ds,   batch_size=1,  # 1 tomo/iter
                          shuffle=False, **dl_kw)

    # ── Model / Optim / AMP ------------------------------------------
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    net      = BYUNet(cfg.__dict__).to(device)

    opt   = torch.optim.AdamW(net.parameters(),
                              lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs * len(train_dl))

    ce_loss = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(getattr(cfg, "class_weights", [1, 1]),
                            dtype=torch.float32).to(device)
    )

    scaler = GradScaler()          # ★ AMP scaler

    # ── TRAIN / VAL loop ---------------------------------------------
    for epoch in range(cfg.epochs):
        # ------------------- train -----------------------------------
        net.train()
        tr_loss = 0.0
        t0 = time.time()
        for batch in train_dl:
            imgs = batch["image"].to(device)                  # (B,1,128³)
            lbls = batch["label"].squeeze(1).long().to(device)

            opt.zero_grad(set_to_none=True)
            with autocast(dtype=torch.float16):
                logits = net({"image": imgs})["logits"]       # (B,2,…)
                loss   = ce_loss(logits, lbls)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss += loss.item()
        sched.step()

        if wandb_run:
            wandb.log({"loss/train": tr_loss/len(train_dl),
                       "lr": sched.get_last_lr()[0]}, step=epoch)

        # ------------------- quick val -------------------------------
        net.eval()
        pred_dfs, gt_dfs = [], []
        with torch.no_grad():
            for batch in val_dl:                               # 1 tomo
                tid   = batch["tomo_id"][0]
                imgs  = batch["image"].to(device)              # (N,1,128³)
                locs  = batch["locs"]                          # (N,3) cpu

                with autocast(dtype=torch.float16):
                    prob_pch = net({"image": imgs})["logits"].softmax(1)[:, 1]

                vol_shape = load_zarr(val_ds.root / f"{tid}.zarr").shape
                full_prob = grid_reconstruct_3d(prob_pch.unsqueeze(1),
                                                 locs, vol_shape).squeeze(0)

                df_pred = post_process_volume(full_prob, spacing=10.0, tomo_id=tid)

                row = val_ds.df.loc[tid]
                if row["Number of motors"] > 0:
                    df_gt = pd.DataFrame([{
                        "tomo_id": tid,
                        "Motor axis 0": row["Motor axis 0"],
                        "Motor axis 1": row["Motor axis 1"],
                        "Motor axis 2": row["Motor axis 2"],
                    }])
                else:
                    df_gt = pd.DataFrame([{
                        "tomo_id": tid,
                        "Motor axis 0": -1, "Motor axis 1": -1, "Motor axis 2": -1,
                    }])

                pred_dfs.append(df_pred)
                gt_dfs.append(df_gt)

        f2 = f2_score(pd.concat(gt_dfs,   ignore_index=True),
                      pd.concat(pred_dfs, ignore_index=True),
                      beta=cfg.beta)

        if wandb_run:
            wandb.log({"F2/val": f2}, step=epoch)

        dt = time.time() - t0
        print(f"[{epoch+1:02d}/{cfg.epochs}] "
              f"loss={tr_loss/len(train_dl):.4f}   "
              f"F2={f2:.3f}   ({dt/60:.1f} min)")

    if wandb_run:
        wandb.finish()


if __name__ == "__main__":
    main()
