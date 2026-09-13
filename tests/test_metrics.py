"""HV/IGD 多目标评价指标封装测试。

调研C D8: 统一参考点 + 目标归一化; IGD 前沿用跨 seed 非支配并集近似。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import (
    normalize_objectives,
    compute_hypervolume,
    compute_igd,
    compute_metrics,
    DEFAULT_REFERENCE_POINT,
)


def test_normalize_objectives_basic():
    """归一化: 除以初始解目标值 (ScPO 式 z = f/f(x0))。"""
    points = [(100.0, 200.0), (120.0, 180.0)]
    initial = (100.0, 200.0)
    norm = normalize_objectives(points, initial)
    assert norm[0] == pytest.approx((1.0, 1.0))
    assert norm[1][0] == pytest.approx(1.2)
    assert norm[1][1] == pytest.approx(0.9)


def test_normalize_rejects_zero_initial():
    """初始解目标为 0 → 报错。"""
    with pytest.raises(ValueError):
        normalize_objectives([(1.0, 2.0)], (0.0, 2.0))


def test_compute_hypervolume_basic():
    """HV: 单点 (1.5, 1.5) 相对参考点 (2, 2) = 0.25。"""
    points = [(1.5, 1.5)]
    hv = compute_hypervolume(points, (2.0, 2.0))
    assert hv == pytest.approx(0.25)


def test_compute_hypervolume_two_points():
    """两点非支配: HV = 面积并集。"""
    # (1.2, 1.8) 和 (1.8, 1.2) 相对 (2,2):
    # (2-1.2)(2-1.8) + (2-1.8)(2-1.2) - 重叠(2-1.8)(2-1.8)
    a = (2.0 - 1.2) * (2.0 - 1.8)
    b = (2.0 - 1.8) * (2.0 - 1.2)
    overlap = (2.0 - 1.8) * (2.0 - 1.8)
    expected = a + b - overlap
    hv = compute_hypervolume([(1.2, 1.8), (1.8, 1.2)], (2.0, 2.0))
    assert hv == pytest.approx(expected)


def test_compute_igd_basic():
    """IGD: candidate 到 reference 前沿的平均距离。

    candidate 命中 (1,2) 但 (2,1) 距离 sqrt(2)≈1.414, 平均 = 0.707。
    """
    reference = [(1.0, 2.0), (2.0, 1.0)]
    candidate = [(1.0, 2.0)]  # 命中一个参考点
    igd = compute_igd(candidate, reference)
    assert igd == pytest.approx(0.7071067811865476)


def test_compute_igd_dominated_candidate():
    """候选被支配: IGD > 0。"""
    reference = [(1.0, 2.0), (2.0, 1.0)]
    candidate = [(3.0, 3.0)]
    igd = compute_igd(candidate, reference)
    assert igd > 0


def test_compute_metrics_full_pipeline():
    """完整指标管线: 归一化 → HV/IGD。"""
    points = [(110.0, 220.0), (130.0, 180.0)]
    initial = (100.0, 250.0)
    reference_front = [(105.0, 210.0), (125.0, 175.0)]
    metrics = compute_metrics(points, initial, reference_front)
    assert metrics["hv"] > 0
    assert metrics["igd"] >= 0
    assert metrics["n_points"] == 2
    assert "reference_point" in metrics
