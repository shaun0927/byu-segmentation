"""
ResNet-34 Flexible-UNet 실험 설정
────────────────────────────────────────────────────────
기본값은 configs.common_config._DEFAULTS 를 상속하고,
여기서 모델·학습 하이퍼파라미터만 덮어쓴다.
"""

from configs.common_config import get_cfg

cfg = get_cfg(
    # ───────────────────────────────────────────
    # 식별용
    exp_name      = "resnet34_flexunet",

    # 모델
    backbone      = "resnet34",
    pretrained    = False,
    decoder_channels = (256, 128, 64, 32, 16),
    deep_supervision = False,

    # 학습
    epochs        = 40,
    lr            = 2e-4,
    weight_decay  = 1e-5,
    optimizer     = "AdamW",
    batch_size    = 8,
    num_workers   = 8,

    # 손실/metric
    beta          = 2,
    pos_weight    = 2.0,          # [motor, background]

    # 로그
    disable_wandb = False,           # wandb 사용
)


