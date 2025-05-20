from __future__ import annotations
import torch
from torch import nn
from monai.networks.nets.flexible_unet import (
    FLEXUNET_BACKBONE, SegmentationHead, UNetDecoder
)

# ──────────────────────────────────────────────────────
class PatchedUNetDecoder(UNetDecoder):
    """Return decoder feature maps *after* each decoder block (MONAI default)."""

    def forward(self, feats: list[torch.Tensor], skip_connect: int = 4):
        skips = feats[:-1][::-1]
        feats = feats[1:][::-1]       # top-most encoder feature not used directly

        outs, x = [], feats[0]        # x = first decoder input
        for i, blk in enumerate(self.blocks):
            x = blk(x, skips[i] if i < skip_connect else None)
            outs.append(x)            # ONLY append after block
        return outs                   # len == decoder_levels


class FlexibleUNet(nn.Module):
    """
    3-D FlexibleUNet wrapper with dynamic decoder_channels
    • encoder backbone: resnet{10…200}, efficientnet-b{0…8,l2}
    • deep-supervision supported (list of seg-maps)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        backbone: str = "resnet34",
        pretrained: bool = False,
        spatial_dims: int = 3,
        deep_supervision: bool = True,
        min_decoder_ch: int = 16,            # floor value
    ):
        super().__init__()
        if backbone not in FLEXUNET_BACKBONE.register_dict:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ---------- Encoder ----------
        enc_cfg      = FLEXUNET_BACKBONE.register_dict[backbone]
        enc_channels = (in_channels, *enc_cfg["feature_channel"])  # e.g. (1,64,128,256,512)

        enc_params = enc_cfg["parameter"]
        enc_params.update(
            {"spatial_dims": spatial_dims,
             "in_channels":  in_channels,
             "pretrained":   pretrained}
        )
        self.encoder = enc_cfg["type"](**enc_params)
        n_levels     = len(enc_channels) - 1      # decoder depth

        # ---------- Dynamic decoder_channels ----------
        enc_last  = enc_channels[-1]           # 512 for resnet34
        dec_ch    = []
        ch        = enc_last // 2             # first decoder out = 256
        for _ in range(n_levels):
            dec_ch.append(max(ch, min_decoder_ch))
            ch //= 2
        dec_ch = tuple(dec_ch)                # (256,128,64,32)

        self.skip_level = n_levels - 1            # for UNet skip-connection count

        # ---------- Decoder ----------
        self.decoder = PatchedUNetDecoder(
            spatial_dims=spatial_dims,
            encoder_channels=enc_channels,
            decoder_channels=dec_ch,
            act=("relu", {"inplace": True}),
            norm=("batch", {"eps": 1e-3, "momentum": 0.1}),
            dropout=0.0,
            bias=False,
            upsample="nontrainable",
            pre_conv="default",
            interp_mode="nearest",
            align_corners=None,
            is_pad=True,
        )

        # ---------- Segmentation heads ----------
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.seg_heads = nn.ModuleList(
                SegmentationHead(
                    spatial_dims=spatial_dims,
                    in_channels=c,
                    out_channels=out_channels,
                    kernel_size=3,
                    act=None,
                )
                for c in dec_ch[:-1]
            )
        else:
            self.seg_heads = nn.ModuleList()

        self.final_head = SegmentationHead(
            spatial_dims=spatial_dims,
            in_channels=dec_ch[-1],
            out_channels=out_channels,
            kernel_size=3,
            act=None,
        )

    # --------------------------------------------------
    def forward(self, x: torch.Tensor):
        feats = self.encoder(x)                       # encoder feature list
        decs  = self.decoder(feats, self.skip_level)  # decoder feature list

        outs: list[torch.Tensor] = []
        if self.deep_supervision:
            for head, feat in zip(self.seg_heads, decs[:-1]):
                outs.append(head(feat))
        outs.append(self.final_head(decs[-1]))        # highest-res seg-map
        return outs                                   # list of tensors
