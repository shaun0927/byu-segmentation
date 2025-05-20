# tests/overfit_one.py
import torch, pandas as pd, tqdm
from data.ds_byu import BYUMotorDataset, simple_collate, LABEL_CSV
from models.net_byu import BYUNet

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. tiny dataset : 한 tomogram  → 4 patch ───────────────
tid = pd.read_csv(LABEL_CSV)["tomo_id"].iloc[0]
ds  = BYUMotorDataset([tid], mode="train", split="train")
dl  = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=simple_collate)

# ── 2. model + optim ───────────────────────────────────────
net = BYUNet({"backbone": "resnet34", "pos_weight": 128.}).to(device)
opt = torch.optim.Adam(net.parameters(), 1e-3)

# ── 3. loop 50 steps ───────────────────────────────────────
pbar = tqdm.trange(50)
for step in pbar:
    batch = next(iter(dl))
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}

    opt.zero_grad()
    out = net(batch)
    out["loss"].backward()
    opt.step()

    pbar.set_description(f"step {step:02d}  loss {out['loss'].item():.4f}")
