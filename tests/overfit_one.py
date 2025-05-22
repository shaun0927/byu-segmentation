# tests/overfit_one.py
from __future__ import annotations
import numpy as np
import pandas as pd
import torch, tqdm
from data.ds_byu import BYUMotorDataset, simple_collate, LABEL_CSV
from models.net_byu import BYUNet

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── tomo_id & spacing 가져오기 ────────────────────────────────
labels_df = pd.read_csv(LABEL_CSV)
row = labels_df.query("`Number of motors` >= 1").iloc[0]
tomo_id: str   = row["tomo_id"]
spacing_arr = None
if "Voxel spacing" in row:
    sp_raw = str(row["Voxel spacing"]).replace("[", "").replace("]", "").replace(",", " ")
    spacing_arr = torch.tensor([float(s) for s in sp_raw.split()][:3],
                               dtype=torch.float32, device=device)  # (Δz,Δy,Δx)

# ── dataset / loader ------------------------------------------------
ds_all = BYUMotorDataset([tomo_id], mode="train", split="train")
pos_idx = next(i for i in range(len(ds_all)) if ds_all[i]["label"].sum() > 0)
ds = torch.utils.data.Subset(ds_all, [pos_idx])
dl = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=simple_collate)

# ── model & optim ---------------------------------------------------
net = BYUNet({"backbone": "resnet34", "pos_weight": 24.}).to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

# ── training loop ---------------------------------------------------
for step in tqdm.trange(200):
    batch = next(iter(dl))
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    opt.zero_grad()
    out = net(batch)
    out["loss"].backward()
    opt.step()

# ── evaluation ------------------------------------------------------
net.eval()
batch = next(iter(dl))
batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
with torch.no_grad():
    out = net(batch)
loss_val = out["loss"].item()

prob_map = torch.sigmoid(out["logits"][0, 0])
pred_idx = prob_map.view(-1).argmax().item()
D, H, W  = prob_map.shape
pred_vox = torch.tensor([pred_idx // (H * W),
                         (pred_idx % (H * W)) // W,
                         pred_idx % W],
                        dtype=torch.float32, device=device)

gt_mask   = batch["label"][0, 0]
gt_center = torch.nonzero(gt_mask > 0.05, as_tuple=False).float().mean(0).to(device)

# spacing 결정
if "spacing" in batch:
    spacing = batch["spacing"][0].to(device)
elif spacing_arr is not None:
    spacing = spacing_arr
else:
    spacing = torch.tensor([10., 10., 10.], device=device)   # fallback

dist_A = torch.norm((pred_vox - gt_center) * spacing).item()

print(f"\nfinal loss {loss_val:.4f}, dist {dist_A:.1f} Å")
print("✓ overfit success" if (loss_val <= 0.1 and dist_A <= 1000) else "✗ overfit failed")
