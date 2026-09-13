"""Pareto 支配、非支配排序、拥挤度计算和质量指标。"""

import math
from dataclasses import dataclass


Objective = tuple[float, float]  # (cost, emission)


def dominates(a: Objective, b: Objective) -> bool:
    """如果解 a Pareto 支配解 b，返回 True。"""
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def non_dominated_sort(population: list[Objective]) -> list[list[int]]:
    """
    NSGA-II 风格非支配排序。

    Args:
        population: 目标值元组列表

    Returns:
        前沿列表，每个前沿是指向 population 的索引列表
    """
    n = len(population)
    domination_count = [0] * n
    dominated_set = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(population[i], population[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif dominates(population[j], population[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1

        if domination_count[i] == 0:
            fronts[0].append(i)

    current = 0
    while current < len(fronts) and fronts[current]:
        next_front = []
        for i in fronts[current]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current += 1
        if next_front:
            fronts.append(next_front)

    return fronts


def crowding_distance(population: list[Objective], front: list[int]) -> list[float]:
    """
    计算一个前沿中解的拥挤度。

    Args:
        population: 完整目标值元组列表
        front: 索引列表（population 索引的子集）

    Returns:
        索引 → 拥挤度的字典
    """
    dist = {idx: 0.0 for idx in front}
    if len(front) <= 2:
        for idx in front:
            dist[idx] = float("inf")
        return dist

    obj_range = [0.0, 0.0]
    for m in range(2):
        vals = [population[idx][m] for idx in front]
        obj_range[m] = max(vals) - min(vals)
        if obj_range[m] == 0:
            obj_range[m] = 1.0  # avoid division by zero

    for m in range(2):
        sorted_front = sorted(front, key=lambda idx: population[idx][m])
        dist[sorted_front[0]] = float("inf")
        dist[sorted_front[-1]] = float("inf")
        for i in range(1, len(sorted_front) - 1):
            dist[sorted_front[i]] += (
                population[sorted_front[i + 1]][m]
                - population[sorted_front[i - 1]][m]
            ) / obj_range[m]

    return dist


def merge_archives(
    archive_a: list[tuple[Objective, dict]],
    archive_b: list[tuple[Objective, dict]],
    capacity: int,
) -> list[tuple[Objective, dict]]:
    """
    合并两个档案并裁剪至容量上限。

    每个档案条目为 (目标值, 额外信息)。
    """
    combined = archive_a + archive_b
    if not combined:
        return []

    # 首先过滤被支配的解
    non_dominated = []
    for i, (obj_i, info_i) in enumerate(combined):
        is_dominated = False
        for j, (obj_j, _) in enumerate(combined):
            if i != j and dominates(obj_j, obj_i):
                is_dominated = True
                break
        if not is_dominated:
            non_dominated.append((obj_i, info_i))

    # 去除近似重复
    unique = {}
    for obj, info in non_dominated:
        key = (round(obj[0], 4), round(obj[1], 4))
        if key not in unique:
            unique[key] = info

    result = [(key, unique[key]) for key in unique]

    # 如果超过容量，按拥挤度裁剪
    if len(result) > capacity:
        objs = [r[0] for r in result]
        indices = list(range(len(result)))
        cd = crowding_distance(objs, indices)
        sorted_idx = sorted(indices, key=lambda i: cd.get(i, 0), reverse=True)
        result = [result[i] for i in sorted_idx[:capacity]]

    return result


def build_reference_front(
    all_archives: list[list[tuple[Objective, dict]]],
) -> list[Objective]:
    """
    从所有方法和运行的档案中构建参考 Pareto 前沿。
    用于计算 IGD。
    """
    combined = []
    for archive in all_archives:
        for obj, _ in archive:
            combined.append(obj)

    if not combined:
        return []

    # 查找非支配解
    non_dominated = []
    for i, a in enumerate(combined):
        is_dominated = False
        for j, b in enumerate(combined):
            if i != j and dominates(b, a):
                is_dominated = True
                break
        if not is_dominated:
            non_dominated.append(a)

    # 去重
    unique = []
    seen = set()
    for o in non_dominated:
        key = (round(o[0], 4), round(o[1], 4))
        if key not in seen:
            seen.add(key)
            unique.append(o)

    return unique


def hypervolume_2d(points: list[Objective], ref_point: Objective) -> float:
    """
    计算精确的 2D 超体积。

    Args:
        points: (成本, 排放) 点列表
        ref_point: 参考点（必须支配或等于所有点）

    Returns:
        超体积值
    """
    if not points:
        return 0.0

    # 过滤被参考点支配的点
    valid = [p for p in points if p[0] <= ref_point[0] and p[1] <= ref_point[1]]
    if not valid:
        return 0.0

    # 按第一目标值排序
    valid.sort(key=lambda p: p[0])

    # 计算超体积
    hv = 0.0
    prev_y = ref_point[1]
    for p in valid:
        if p[1] < prev_y:
            hv += (ref_point[0] - p[0]) * (prev_y - p[1])
            prev_y = p[1]

    return hv


def igd(candidate: list[Objective], reference: list[Objective]) -> float:
    """
    反向世代距离：每个参考点到其最近候选点的平均距离。

    Returns:
        IGD 值（越小越好），候选为空时返回 inf
    """
    if not candidate:
        return float("inf")
    if not reference:
        return 0.0

    total = 0.0
    for ref in reference:
        min_dist = min(
            math.sqrt((ref[0] - c[0]) ** 2 + (ref[1] - c[1]) ** 2)
            for c in candidate
        )
        total += min_dist

    return total / len(reference)


def unary_epsilon(candidate: list[Objective], reference: list[Objective]) -> float:
    """
    一元 epsilon 指标：使候选 epsilon 支配参考点的最小 epsilon。

    Returns:
        epsilon 值（越小越好）
    """
    if not candidate:
        return float("inf")
    if not reference:
        return 0.0

    max_eps = 0.0
    for ref in reference:
        min_eps = min(
            max(c[0] / ref[0] - 1 if ref[0] > 0 else 0, c[1] / ref[1] - 1 if ref[1] > 0 else 0)
            for c in candidate
        )
        max_eps = max(max_eps, min_eps)

    return max_eps
