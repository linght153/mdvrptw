"""多目标 ALNS 的父代解选择策略。"""

import numpy as np

from src.core.pareto import Objective, crowding_distance


def select_crowding(
    archive: list[tuple[Objective, dict]], rng: np.random.Generator
) -> dict:
    """选择拥挤度最大的父代解。促进 Pareto 前沿稀疏区域的探索。"""
    if not archive:
        return None
    if len(archive) == 1:
        return archive[0]

    objs = [a[0] for a in archive]
    indices = list(range(len(archive)))
    cd = crowding_distance(objs, indices)

    # 选择拥挤度最大的解
    best_idx = max(cd, key=lambda i: cd[i])
    return archive[best_idx]


def select_random(
    archive: list[tuple[Objective, dict]], rng: np.random.Generator
) -> dict:
    """从档案中均匀随机选择。"""
    if not archive:
        return None
    idx = int(rng.integers(0, len(archive)))
    return archive[idx]


def select_roulette(
    archive: list[tuple[Objective, dict]], rng: np.random.Generator
) -> dict:
    """
    基于拥挤度的轮盘赌选择。
    平衡随机性与多样性促进。
    """
    if not archive:
        return None
    if len(archive) == 1:
        return archive[0]

    objs = [a[0] for a in archive]
    indices = list(range(len(archive)))
    cd = crowding_distance(objs, indices)

    # 处理 inf 值（边界解）
    weights = []
    for i in indices:
        v = cd.get(i, 0)
        if v == float("inf"):
            weights.append(1000.0)
        else:
            weights.append(max(v, 0.001))

    total = sum(weights)
    probs = [w / total for w in weights]
    idx = int(rng.choice(len(archive), p=probs))
    return archive[idx]


# ============================================================
# 注册表
# ============================================================

PARENT_SELECTION_STRATEGIES = {
    "crowding": select_crowding,
    "random": select_random,
    "roulette": select_roulette,
}
