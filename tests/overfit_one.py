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

# ── 4. loss 와 예측 좌표 확인 ──────────────────────────────────
net.eval()
batch = next(iter(dl))
batch = {k: v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
with torch.no_grad():
    out = net(batch)
loss_val = out["loss"].item()

# 첫 번째 패치 (양성) 기준으로 좌표 계산
prob = torch.sigmoid(out["logits"][0,0])
pred_idx = prob.view(-1).argmax().item()
D, H, W = prob.shape
pred_coord = torch.tensor([
    pred_idx // (H * W),
    (pred_idx % (H * W)) // W,
    pred_idx % W,
], dtype=torch.float32, device=prob.device)

gt_mask = batch["label"][0,0]
gt_center = torch.nonzero(gt_mask > 0.5, as_tuple=False).float().mean(0)

dist_vox = torch.norm(pred_coord - gt_center).item()
dist_A   = dist_vox * 10.0

print(f"final loss {loss_val:.4f}, dist {dist_A:.1f} Å")
if (loss_val <= 0.1) and (dist_A <= 1000.0):
    print("✓ overfit success")
else:
    print("✗ overfit failed")
