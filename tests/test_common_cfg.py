"""
Unit–test for configs.common_config

✔  (A) .env / environment variable 가 자동 적용되는지 확인
✔  (B) get_cfg(**overrides) 로 값이 덮어써지는지 확인
"""

import os, importlib, sys, textwrap
from types import SimpleNamespace
from pathlib import Path

# ------------------------------------------------------------------
# (A) 환경변수   (테스트 격리를 위해 임시로 주입했다가 나중에 복원)
# ------------------------------------------------------------------
_ORIG_ENV = os.environ.get("WANDB_API_KEY")
TEST_KEY  = "TEST_TOKEN_1234567890"

os.environ["WANDB_API_KEY"] = TEST_KEY

# ①  모듈 재로드 → load_dotenv() 가 새 env 읽도록
if "configs.common_config" in sys.modules:
    del sys.modules["configs.common_config"]
from configs.common_config import get_cfg

cfg_env = get_cfg()
assert cfg_env.wandb_api == TEST_KEY, "env variable not picked up"
assert cfg_env.beta == 2
assert cfg_env.classes == ["motor"]
print("env-auto-load OK")

# ------------------------------------------------------------------
# (B) override
# ------------------------------------------------------------------
cfg_over = get_cfg(wandb_api="OVERRIDE_KEY", epochs=30)
assert cfg_over.wandb_api == "OVERRIDE_KEY"
assert cfg_over.epochs == 30
print("override OK")

# ------------------------------------------------------------------
# tidy-up
# ------------------------------------------------------------------
if _ORIG_ENV is None:
    os.environ.pop("WANDB_API_KEY")
else:
    os.environ["WANDB_API_KEY"] = _ORIG_ENV
print("✓  all tests passed")
