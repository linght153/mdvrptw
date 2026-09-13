"""ALNS 破坏算子 — 从解中移除 k 个客户。"""

import numpy as np

from src.core.solution import Solution
from src.core.objectives import removal_cost_delta


def random_removal(
    solution: Solution, k: int, rng: np.random.Generator
) -> tuple[Solution, list[int]]:
    """随机移除 k 个客户。"""
    partial = solution.copy()
    customers = []
    all_customers = list(partial.all_served_customers())
    indices = rng.choice(len(all_customers), size=min(k, len(all_customers)), replace=False)

    for idx in indices:
        ci = all_customers[idx]
        customers.append(ci)
        _remove_customer(partial, ci)

    return partial, customers


def worst_cost_removal(
    solution: Solution, k: int, rng: np.random.Generator
) -> tuple[Solution, list[int]]:
    """移除移除后成本节省最大的 k 个客户。"""
    partial = solution.copy()
    all_customers = list(partial.all_served_customers())
    deltas = []
    for ci in all_customers:
        cost_saving, _ = removal_cost_delta(partial, ci)
        deltas.append((ci, cost_saving))

    deltas.sort(key=lambda x: -x[1])
    # 添加噪声以避免确定性行为
    noise = [rng.random() * 0.1 * max(abs(d[1]), 1e-6) for d in deltas]
    deltas_noisy = [(d[0], d[1] + n) for d, n in zip(deltas, noise)]
    deltas_noisy.sort(key=lambda x: -x[1])

    customers = []
    for ci, _ in deltas_noisy[:k]:
        customers.append(ci)
        _remove_customer(partial, ci)

    return partial, customers


def worst_emission_removal(
    solution: Solution, k: int, rng: np.random.Generator
) -> tuple[Solution, list[int]]:
    """移除移除后排放节省最大的 k 个客户。"""
    partial = solution.copy()
    all_customers = list(partial.all_served_customers())
    deltas = []
    for ci in all_customers:
        _, emission_saving = removal_cost_delta(partial, ci)
        deltas.append((ci, emission_saving))

    deltas.sort(key=lambda x: -x[1])
    noise = [rng.random() * 0.1 * max(abs(d[1]), 1e-6) for d in deltas]
    deltas_noisy = [(d[0], d[1] + n) for d, n in zip(deltas, noise)]
    deltas_noisy.sort(key=lambda x: -x[1])

    customers = []
    for ci, _ in deltas_noisy[:k]:
        customers.append(ci)
        _remove_customer(partial, ci)

    return partial, customers


def worst_combined_removal(
    solution: Solution, k: int, rng: np.random.Generator
) -> tuple[Solution, list[int]]:
    """
    按组合（成本 + 排放）改进移除 k 个客户。
    每次迭代随机分配成本与排放权重。
    """
    partial = solution.copy()
    all_customers = list(partial.all_served_customers())
    w = rng.random()  # weight for cost
    deltas = []
    for ci in all_customers:
        cs, es = removal_cost_delta(partial, ci)
        combined = w * cs + (1 - w) * es
        deltas.append((ci, combined))

    deltas.sort(key=lambda x: -x[1])
    customers = []
    for ci, _ in deltas[:k]:
        customers.append(ci)
        _remove_customer(partial, ci)

    return partial, customers


def shaw_removal(
    solution: Solution, k: int, rng: np.random.Generator, sigma: float = 10.0
) -> tuple[Solution, list[int]]:
    """
    基于 Shaw 相似度度量移除相关客户。
    相似度考虑距离、需求和车场分配。
    """
    partial = solution.copy()
    all_customers = list(partial.all_served_customers())
    inst = partial.instance

    # 随机选择种子客户
    seed = all_customers[rng.integers(0, len(all_customers))]

    customers = [seed]
    _remove_customer(partial, seed)

    # 一次性计算所有客户相对种子的相似度并排序 (种子在循环中不变 —
    # 旧实现每轮对全部剩余客户重算 + _shaw_distance 每次重扫距离矩阵
    # O(n²), pr05 实测单次 shaw ~7.5s, destroy 占引擎时间 88%)
    sims = []
    for ci in all_customers:
        if ci == seed:
            continue
        similarity = (_shaw_distance(inst, seed, ci) + _shaw_demand(inst, seed, ci)
                      + _shaw_depot(solution, seed, ci))
        sims.append((ci, similarity))
    sims.sort(key=lambda x: x[1])

    # 每轮在剩余客户上按 rank 偏置概率选择 (P(i) ~ (1/rank)^σ),
    # rng 消费与原实现一致 (每轮一次 rng.choice, 概率按当前剩余数归一化)
    remaining = sims
    while len(customers) < k and remaining:
        m = len(remaining)
        ranks = np.arange(1, m + 1)
        probs = 1.0 / (ranks ** sigma * 0.1)
        probs /= probs.sum()
        pick_idx = int(rng.choice(m, p=probs))
        picked, _ = remaining.pop(pick_idx)
        customers.append(picked)
        _remove_customer(partial, picked)

    return partial, customers


def depot_focused_removal(
    solution: Solution, k: int, rng: np.random.Generator
) -> tuple[Solution, list[int]]:
    """优先从单个车场的路由中移除 k 个客户。"""
    partial = solution.copy()
    inst = partial.instance

    # 随机选择一个车场
    depot_idx = int(rng.choice(inst.num_depots))
    depot_customers = []
    for route in solution.routes.get(depot_idx, []):
        depot_customers.extend(route)

    if not depot_customers:
        # 回退：随机移除
        return random_removal(solution, k, rng)

    to_remove = min(k, len(depot_customers))
    indices = rng.choice(len(depot_customers), size=to_remove, replace=False)
    customers = [depot_customers[i] for i in indices]

    for ci in customers:
        _remove_customer(partial, ci)

    return partial, customers


def cluster_removal(
    solution: Solution, k: int, rng: np.random.Generator
) -> tuple[Solution, list[int]]:
    """移除空间聚集的 k 个客户。"""
    partial = solution.copy()
    inst = partial.instance

    # 随机选择一个客户作为聚类中心
    all_customers = list(partial.all_served_customers())
    center = all_customers[rng.integers(0, len(all_customers))]
    cx = inst.customers[center - inst.num_depots].x
    cy = inst.customers[center - inst.num_depots].y

    # 计算到中心的距离
    dists = []
    for ci in all_customers:
        c = inst.customers[ci - inst.num_depots]
        d = (c.x - cx) ** 2 + (c.y - cy) ** 2
        dists.append((ci, d))

    dists.sort(key=lambda x: x[1])
    customers = [d[0] for d in dists[:k]]

    for ci in customers:
        _remove_customer(partial, ci)

    return partial, customers


def route_removal(
    solution: Solution, k: int, rng: np.random.Generator
) -> tuple[Solution, list[int]]:
    """从随机选择的路由中移除所有客户。"""
    partial = solution.copy()

    all_routes = []
    for depot_idx, routes in solution.routes.items():
        for route in routes:
            if route:
                all_routes.append(route)

    if not all_routes:
        return partial, []

    route_idx = int(rng.integers(0, len(all_routes)))
    customers = list(all_routes[route_idx])

    for ci in customers:
        _remove_customer(partial, ci)

    return partial, customers


def mismatched_removal(
    solution: Solution, k: int, rng: np.random.Generator
) -> tuple[Solution, list[int]]:
    """错配优先移除 (方向①, 2026-09-06): 优先移除"实际车场较远"的客户。

    错配度 = d(实际路由车场, 客户) − d(最近车场, 客户) (全局号索引
    distance_matrix; 最近车场 = argmin 纯距离)。按错配度降序移除, 直到移除
    k 个或错配客户耗尽; 不足 k 用 random_removal 语义补齐。

    只做移除 (不查容量/车辆数/TW) — 移除后 repair 走现有全车场 greedy
    重插, 客户自然获得跨车场重插机会。
    """
    partial = solution.copy()
    inst = partial.instance
    dm = inst.distance_matrix
    nd = inst.num_depots

    # 所有已服务客户中实际车场 ≠ 最近车场的, 按错配度降序
    scored = []
    for di, routes in solution.routes.items():
        for route in routes:
            for ci in route:
                nearest = min(range(nd), key=lambda d: dm[d][ci])
                mismatch = dm[di][ci] - dm[nearest][ci]
                if mismatch > 1e-9:
                    scored.append((ci, mismatch))
    scored.sort(key=lambda t: -t[1])

    customers = []
    for ci, _ in scored:
        if len(customers) >= k:
            break
        customers.append(ci)
        _remove_customer(partial, ci)

    # 不足 k 用 random_removal 语义补齐 (只补仍服务客户数)
    if len(customers) < k:
        served = list(partial.all_served_customers())
        need = min(k - len(customers), len(served))
        if need > 0:
            partial, extra = random_removal(partial, need, rng)
            customers.extend(extra)

    return partial, customers


# ============================================================
# 内部辅助函数
# ============================================================

def _remove_customer(solution: Solution, customer_idx: int):
    """从路由中移除客户；空路由则删除。"""
    for depot_idx, routes in solution.routes.items():
        for route in routes:
            if customer_idx in route:
                route.remove(customer_idx)
                if not route:
                    routes.remove(route)
                return


def _shaw_distance(inst, seed: int, other: int) -> float:
    """基于距离的相似度（归一化; max_d 走 Instance 惰性缓存, 曾每调用 O(n²)）。"""
    d = inst.distance_matrix[seed][other]
    max_d = inst.distance_max
    return d / max_d if max_d > 0 else 0


def _shaw_demand(inst, seed: int, other: int) -> float:
    """需求量相似度 (max_dem 走 Instance 惰性缓存)。"""
    s = inst.customers[seed - inst.num_depots].demand
    o = inst.customers[other - inst.num_depots].demand
    max_dem = inst.demand_max
    return abs(s - o) / max_dem if max_dem > 0 else 0


def _shaw_depot(sol: Solution, seed: int, other: int) -> float:
    """同车场相似度。"""
    for di, routes in sol.routes.items():
        s_in = any(seed in r for r in routes)
        o_in = any(other in r for r in routes)
        if s_in and o_in:
            return 0.0
        if s_in != o_in:
            return 1.0
    return 0.5


# ============================================================
# 注册表
# ============================================================

DESTROY_OPERATORS = {
    "random": random_removal,
    "worst_cost": worst_cost_removal,
    "worst_emission": worst_emission_removal,
    "worst_combined": worst_combined_removal,
    "shaw": shaw_removal,
    "depot_focused": depot_focused_removal,
    "cluster": cluster_removal,
    "route": route_removal,
}
