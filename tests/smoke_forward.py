# tests/smoke_forward.py
import torch, pandas as pd
from data.ds_byu import BYUMotorDataset, simple_collate, LABEL_CSV
from models.net_byu import BYUNet

# ── 1. DataLoader : 2 tomograms → 8 patches ───────────────────
ids = pd.read_csv(LABEL_CSV)["tomo_id"].unique()[:2]    # small subset
ds  = BYUMotorDataset(list(ids), mode="train", split="train")
dl  = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=simple_collate)

batch = next(iter(dl))
print("batch image :", batch["image"].shape)            # (8,1,128,128,128)
print("batch label :", batch["label"].shape)

# ── 2. Model  ────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
net = BYUNet({"backbone": "resnet34"}).to(device)

# tensors → device
batch = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}

# forward
out = net(batch)
print("logits  :", out["logits"].shape)                 # (8,2,64,64,64)
print("loss    :", out["loss"].item())

# ── 3. one backward to make sure grads flow ────────────────
out["loss"].backward()
# grad 확인 : 첫 번째 학습가능 파라미터 검색
for p in net.parameters():
    if p.grad is not None:
        print("✓ backward pass OK  (grad norm)", p.grad.norm().item())
        break
