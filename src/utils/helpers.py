"""项目通用的实用函数。"""

import os
from pathlib import Path


def project_root() -> Path:
    """返回项目根目录。"""
    return Path(__file__).resolve().parent.parent.parent


def load_yaml(path: str) -> dict:
    """加载 YAML 文件并返回字典。"""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """如果目录不存在则创建并返回 Path。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def seeded_rng(seed: int):
    """从种子创建 numpy 随机生成器。"""
    import numpy as np
    return np.random.default_rng(seed)
