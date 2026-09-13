"""多目标评价指标封装: HV/IGD + 归一化。

设计决策 (调研C D8):
- 参考点固定 (2, 2) 归一化 (ScPO 式 z = f/f(x0))
- IGD 参考前沿: 调用方提供 (跨 seed 非支配并集)
- 复用 src.core.pareto 的 hypervolume_2d/igd 原语
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.core.pareto import hypervolume_2d, igd

# 归一化后固定参考点 (ScPO: 初始解 2 倍目标值)
DEFAULT_REFERENCE_POINT = (2.0, 2.0)


def normalize_objectives(
    points: Sequence[tuple[float, float]],
    initial_objective: tuple[float, float],
) -> list[tuple[float, float]]:
    """归一化: z = f / f(x0) (逐目标)。

    Args:
        points: 原始目标点列表 [(cost, emission), ...]
        initial_objective: 初始解目标 (归一化基准)

    Returns:
        归一化点列表

    Raises:
        ValueError: 初始目标任一分量 <= 0
    """
    c0, e0 = initial_objective
    if c0 <= 0 or e0 <= 0:
        raise ValueError(f"初始目标必须为正, 得到 ({c0}, {e0})")
    return [(c / c0, e / e0) for c, e in points]


def compute_hypervolume(
    points: Sequence[tuple[float, float]],
    reference_point: tuple[float, float] = DEFAULT_REFERENCE_POINT,
) -> float:
    """计算 Pareto 前沿的 hypervolume (二维)。

    Args:
        points: 归一化目标点
        reference_point: 参考点 (默认 (2,2))
    """
    if not points:
        return 0.0
    return float(hypervolume_2d(list(points), reference_point))


def compute_igd(
    candidate: Sequence[tuple[float, float]],
    reference_front: Sequence[tuple[float, float]],
) -> float:
    """计算 IGD (Inverted Generational Distance)。

    Args:
        candidate: 候选解集 (归一化)
        reference_front: 参考前沿 (归一化, 跨 seed 非支配并集)
    """
    if not reference_front:
        return float("inf")
    return float(igd(list(candidate), list(reference_front)))


def compute_metrics(
    points: Sequence[tuple[float, float]],
    initial_objective: tuple[float, float],
    reference_front: Sequence[tuple[float, float]],
    reference_point: tuple[float, float] = DEFAULT_REFERENCE_POINT,
) -> dict:
    """完整指标管线: 归一化 → HV + IGD。

    Returns:
        {"hv": float, "igd": float, "n_points": int, "reference_point": tuple}
    """
    norm = normalize_objectives(points, initial_objective)
    norm_ref = normalize_objectives(reference_front, initial_objective)
    return {
        "hv": compute_hypervolume(norm, reference_point),
        "igd": compute_igd(norm, norm_ref),
        "n_points": len(points),
        "reference_point": reference_point,
    }
