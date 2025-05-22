# models/net_byu.py
from __future__ import annotations
import torch, torch.nn.functional as F
from torch import nn
from models.mdl_flexunet import FlexibleUNet


# ──────────────────────────────────────────────────────
class CEPlus(nn.Module):
    """
    Weighted Cross-Entropy + optional Label-Smoothing + optional Focal-γ.
    Args
    ----
    pos_w : 양성/배경 가중치비 (>=1)
    smooth: label-smoothing ε  (0~0.1 정도)
    gamma : focal loss γ       (0 ⇒ 일반 CE)
    """
    def __init__(self, pos_w: float = 128.0, smooth: float = 0.0, gamma: float = 0.0):
        super().__init__()
        self.register_buffer("w", torch.tensor([1.0, pos_w]))
        self.smooth = smooth
        self.gamma  = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits : (N,2,*)  target(one-hot) : (N,2,*)
        logp = F.log_softmax(logits, dim=1)

        if self.smooth > 0.0:
            K = logits.size(1)
            target = (1 - self.smooth) * target + self.smooth / K

        ce = -(target * logp)                       # (N,2,*)

        if self.gamma > 0.0:                        # focal scaling
            p = torch.exp(-ce)
            ce = (1 - p) ** self.gamma * ce

        w_ce = self.w[None, :, None, None, None] * ce
        return w_ce.mean()


# ──────────────────────────────────────────────────────
class BYUNet(nn.Module):
    """
    BYU Motor detector
    cfg dict keys (with defaults) :
        backbone          : "resnet34"
        pretrained        : False
        pos_weight        : 256.0
        label_smooth      : 0.0
        focal_gamma       : 0.0
        deep_supervise    : True   # ← prefer this
        deep_supervision  : alias  # ← fallback for old configs
    """
    def __init__(self, cfg: dict):
        super().__init__()

        # ── deep-supervision 키 통합 ─────────────────────────────
        deep = cfg.get("deep_supervise", cfg.get("deep_supervision", True))

        self.model = FlexibleUNet(
            in_channels=1,
            out_channels=2,               # background / motor
            backbone       = cfg.get("backbone", "resnet34"),
            pretrained     = cfg.get("pretrained", False),
            deep_supervision = deep,      # ← 단일 변수로 전달
        )

        self.loss_fn = CEPlus(
            pos_w  = cfg.get("pos_weight", 256.0),
            smooth = cfg.get("label_smooth", 0.0),
            gamma  = cfg.get("focal_gamma", 0.0),
        )

    # --------------------------------------------------
    def forward(self, batch: dict):
        x   = batch["image"]                 # (B,1,128³)
        ygt = batch.get("label")             # (B,1,128³) or None

        outs   = self.model(x)               # list(seg-maps) or [seg]
        logits = outs[-1]                    # main map

        out = {"logits": logits}
        if ygt is not None:                  # ---- training ----
            # 1->2 channel one-hot
            y_bg = 1.0 - ygt.clamp(max=1.0)
            y2   = torch.cat([y_bg, ygt], 1)

            losses = []
            for p in outs:                   # deep-supervise
                gtd = F.adaptive_max_pool3d(y2, p.shape[-3:])
                losses.append(self.loss_fn(p, gtd))
            out["loss"] = torch.stack(losses).mean()
        return out
