"""
EfficientNet-B3 Flexible-UNet 실험 설정
"""

from configs.common_config import get_cfg

cfg = get_cfg(
    exp_name   = "effb3_flexunet",
    backbone   = "efficientnet-b3",
    pretrained = True,
    decoder_channels = (192, 112, 40, 24, 16),
    deep_supervision = True,

    epochs     = 60,
    lr         = 1e-4,
    optimizer  = "Adam",
    batch_size = 6,
)
