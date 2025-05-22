# models/net_byu.py
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from models.mdl_flexunet import FlexibleUNet

class BYUNet(nn.Module):
    """
    BYU Motor detector with binary heatmap regression using BCEWithLogitsLoss

    cfg dict keys (with defaults) :
        backbone          : "resnet34"
        pretrained        : False
        pos_weight        : 24.0
        label_smooth      : 0.0    # unused in BCE
        focal_gamma       : 0.0    # unused in BCE
        deep_supervise    : True
        deep_supervision  : alias
    """
    def __init__(self, cfg: dict):
        super().__init__()
        # integrate deep supervision key
        deep = cfg.get("deep_supervise", cfg.get("deep_supervision", True))

        # segmentation network
        self.model = FlexibleUNet(
            in_channels=1,
            out_channels=1,
            backbone=cfg.get("backbone", "resnet34"),
            pretrained=cfg.get("pretrained", False),
            deep_supervision=deep,
        )

        # binary cross-entropy with logits, apply pos_weight
        pw = torch.tensor(cfg.get("pos_weight", 1.0))
        self.register_buffer("pos_weight", pw)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, batch: dict):
        x = batch["image"]            # (B,1,D,H,W)
        ygt = batch.get("label")      # (B,1,D,H,W) or None

        # forward once
        outs = self.model(x)           # list of heatmaps, each (B,1,d',h',w')
        logits = outs[-1]              # main output

        out = {"logits": logits}
        if ygt is not None:
            losses = []
            # deep supervision: downsample ground truth to each resolution
            for pred in outs:
                # adaptive pool to match pred size
                y_ds = F.adaptive_max_pool3d(ygt, pred.shape[-3:])
                losses.append(self.loss_fn(pred, y_ds))
            out["loss"] = torch.stack(losses).mean()
        return out
