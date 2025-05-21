"""
configs/common_config.py
────────────────────────────────────────────────────────
모델·실험 공통 설정을 SimpleNamespace 로 관리.

사용 예
-------
from configs.common_config import get_cfg
cfg = get_cfg(wandb_api="xxxxxxxx", epochs=30)
print(cfg.beta)        # 2
print(cfg.classes)     # ['motor']
"""

from types import SimpleNamespace
import copy, os
from dotenv import load_dotenv

# ─── 기본값 정의 ───────────────────────────────────────
_DEFAULTS = dict(
    # project & logging
    project_name   = "byu-motor-2025",
    wandb_api    = os.getenv("WANDB_API_KEY"),   # ← .env 값 자동 사용
    disable_wandb  = False,

    # data
    data_root      = "data/processed",
    label_csv      = "data/raw/train_labels.csv",
    roi_size       = (128, 128, 128),

    # training
    epochs         = 10,
    lr             = 1e-3,
    batch_size     = 8,
    num_workers    = 8,
    seed           = 42,
    pin_memory     = True,          # ① GPU 전송 가속
    persistent_workers = True,      # ② 워커 프로세스 재사용

    # model / metric
    beta           = 2,             # F-β, β=2
    classes        = ["motor"],     # single-class segmentation
    in_channels    = 1,

    # logging
    val_patch_bs = 2,
    log_every    = 20, 
)

# ─── factory 메서드 ───────────────────────────────────
def get_cfg(**overrides):
    """
    kwargs 로 전달된 값만 덮어쓰기한 새 SimpleNamespace 반환
    (원본 _DEFAULTS 는 건드리지 않음)
    """
    cfg_dict = copy.deepcopy(_DEFAULTS)
    cfg_dict.update(overrides)
    cfg = SimpleNamespace(**cfg_dict)

    # 파생 값
    cfg.n_classes = len(cfg.classes)

    return cfg
