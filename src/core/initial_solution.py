"""Green MDVRP 初始解构造策略。"""

import numpy as np

from src.core.instance import Instance
from src.core.solution import Solution


def nearest_neighbor(instance: Instance, rng: np.random.Generator) -> Solution:
    """
    最近邻启发式，结合车场就近分配（车辆数上限约束）。

    阶段 1: 每个客户分派最近车场; 逐车场用最近邻构建路由 (只取本车场
    分派客户), 直到容量满或本车场车辆用完。
    阶段 2: 车辆满而仍有未服务客户 → 重分派到最近的未满车场。
    """
    routes: dict[int, list[list[int]]] = {d.index: [] for d in instance.depots}
    depot_assignments: dict[int, int] = {}

    # 每个客户分配到最近车场
    for c in instance.customers:
        best_dist = float("inf")
        best_depot = 0
        for d in instance.depots:
            dist = instance.distance_matrix[d.index][c.index]
            if dist < best_dist:
                best_dist = dist
                best_depot = d.index
        depot_assignments[c.index] = best_depot
        c.assigned_depot_index = best_depot

    cap = instance.max_vehicle_capacity
    remaining = set(instance.customer_indices)
    tw_aware = instance.has_time_windows

    # 阶段 1: 逐车场构建 (只取本车场分派客户)
    for depot_idx in instance.depot_indices:
        veh_limit = instance.depots[depot_idx].vehicles_available
        while remaining and len(routes[depot_idx]) < veh_limit:
            candidates = [
                ci for ci in remaining if depot_assignments.get(ci) == depot_idx
            ]
            if not candidates:
                break  # 本车场区域客户已服务完
            start = min(
                candidates,
                key=lambda ci: instance.distance_matrix[depot_idx][ci],
            )
            route = [start]
            remaining.remove(start)
            current_load = instance.customers[start - instance.num_depots].demand
            current = start

            # 最近邻扩展 (只在本车场分派客户内; TW/duration 实例只接受追加后
            # 仍可行的客户 — 否则长路由传播违规, 事后 enforce 车满时无法修复:
            # pr14/17/18/20 实测 120s 无可行解 (TW); pr16 实测初始超 D 328
            # (duration, 2026-09-11))
            route_idx = len(routes[depot_idx])
            dur_limited = instance.route_duration_limit is not None
            if tw_aware or dur_limited:
                from src.operators.repair import _route_tw_excess
                from src.core.solution import route_return_duration
            while remaining:
                best_next = None
                best_dist = float("inf")
                for ci in remaining:
                    if depot_assignments.get(ci) != depot_idx:
                        continue
                    c_demand = instance.customers[ci - instance.num_depots].demand
                    if current_load + c_demand <= cap:
                        if tw_aware or dur_limited:
                            trial = route + [ci]
                            if tw_aware and _route_tw_excess(
                                    instance, depot_idx, trial,
                                    route_idx) > 1e-9:
                                continue
                            if dur_limited and route_return_duration(
                                    instance, depot_idx, trial,
                                    route_idx)[1] > instance.route_duration_limit + 1e-9:
                                continue
                        d = instance.distance_matrix[current][ci]
                        if d < best_dist:
                            best_dist = d
                            best_next = ci
                if best_next is None:
                    break
                route.append(best_next)
                current_load += instance.customers[best_next - instance.num_depots].demand
                remaining.remove(best_next)
                current = best_next

            routes[depot_idx].append(route)
            for ci in route:
                instance.customers[ci - instance.num_depots].assigned_depot_index = depot_idx

    # 阶段 2: 车辆满而仍有客户 → 复用跨车场插入 (优先已有路由容量空位,
    # 不增车; 无空位才在未满车场开新路由; 全满兜底负载最轻路由)。
    # TW 实例用 TW-aware 版 (纯距离插入会制造传播违规)。
    # 旧实现: 只找"未满车场开新路由" → 已有路由空位被浪费, 且全满时
    # 无条件 append 超车辆上限 (pr17 每场 1 辆: depot1 出现 3 条路由;
    # pr18 depot4 9 条 vs 上限 2 — 2026-09-03 修复)
    from src.operators.repair import (
        _insert_cheapest_cost, _insert_cheapest_cost_tw,
    )

    insert_fn = _insert_cheapest_cost_tw if tw_aware else _insert_cheapest_cost
    for ci in sorted(remaining):
        insert_fn(Solution(instance, routes), ci)

    return Solution(instance, routes)


def savings_method(instance: Instance, rng: np.random.Generator) -> Solution:
    """
    Clarke-Wright 节约法，扩展到 MDVRP。

    从每个客户单独一条路由开始（分配到最近车场）。
    按节约值从高到低合并路由，满足容量约束。
    """
    routes: dict[int, list[list[int]]] = {d.index: [] for d in instance.depots}
    depot_assignments: dict[int, int] = {}

    # 分配到最近车场
    for c in instance.customers:
        best_dist = float("inf")
        best_depot = 0
        for d in instance.depots:
            dist = instance.distance_matrix[d.index][c.index]
            if dist < best_dist:
                best_dist = dist
                best_depot = d.index
        depot_assignments[c.index] = best_depot

    # 初始：每个客户单独一条路由
    depot_routes: dict[int, dict[int, list[int]]] = {d.index: {} for d in instance.depots}
    for c in instance.customers:
        di = depot_assignments[c.index]
        depot_routes[di][c.index] = [c.index]

    # 计算同一车场内每对客户的节约值
    savings_list = []
    for di, cust_map in depot_routes.items():
        custs = list(cust_map.keys())
        for i_idx in range(len(custs)):
            for j_idx in range(i_idx + 1, len(custs)):
                ci = custs[i_idx]
                cj = custs[j_idx]
                s = (
                    instance.distance_matrix[di][ci]
                    + instance.distance_matrix[di][cj]
                    - instance.distance_matrix[ci][cj]
                )
                savings_list.append((s, di, ci, cj))

    savings_list.sort(key=lambda x: -x[0])
    cap = instance.max_vehicle_capacity

    # 合并
    for _, di, ci, cj in savings_list:
        if ci not in depot_routes[di] or cj not in depot_routes[di]:
            continue
        ri = depot_routes[di][ci]
        rj = depot_routes[di][cj]
        combined = ri + rj
        combined_demand = sum(
            instance.customers[x - instance.num_depots].demand for x in combined
        )
        if combined_demand <= cap:
            depot_routes[di][ci] = combined
            del depot_routes[di][cj]

    # 展平
    for di, cust_map in depot_routes.items():
        routes[di] = list(cust_map.values())

    sol = Solution(instance, routes)
    return _trim_to_vehicle_limit(sol)


def random_insertion(instance: Instance, rng: np.random.Generator) -> Solution:
    """
    随机顺序插入：打乱客户，将每个客户插入其分配车场的
    任意路由中的随机可行位置。
    """
    routes: dict[int, list[list[int]]] = {d.index: [] for d in instance.depots}
    cap = instance.max_vehicle_capacity

    # 分配到最近车场
    depot_assignments: dict[int, int] = {}
    for c in instance.customers:
        best_dist = float("inf")
        best_depot = 0
        for d in instance.depots:
            dist = instance.distance_matrix[d.index][c.index]
            if dist < best_dist:
                best_dist = dist
                best_depot = d.index
        depot_assignments[c.index] = best_depot

    # 随机顺序
    customer_order = rng.permutation([c.index for c in instance.customers])

    for ci in customer_order:
        di = depot_assignments[ci]
        c_demand = instance.customers[ci - instance.num_depots].demand
        placed = False

        # 尝试插入到该车场的已有路由中
        for route in routes[di]:
            route_demand = sum(
                instance.customers[x - instance.num_depots].demand for x in route
            )
            if route_demand + c_demand <= cap:
                pos = rng.integers(0, len(route) + 1)
                route.insert(pos, ci)
                placed = True
                break

        if not placed and len(routes[di]) < instance.depots[di].vehicles_available:
            routes[di].append([ci])
        elif not placed:
            # 该车场车辆满: 留给后处理跨车场重插
            routes[di].append([ci])

    sol = Solution(instance, routes)
    return _trim_to_vehicle_limit(sol)


def _trim_to_vehicle_limit(sol: Solution) -> Solution:
    """车辆数约束后处理: 超限车场移除最短路由, 客户用跨车场贪心重插。

    初始解构造 (savings/random) 可能产生每车场路由数 > vehicles_available
    的解 (如 savings 初始每客户一路由), 此函数收敛到合法车辆数。
    兜底 (所有车场满) 时 _insert_cheapest_cost 就近开新车 (数据异常场景)。
    """
    from src.operators.repair import _insert_cheapest_cost

    inst = sol.instance
    for di in list(sol.routes.keys()):
        limit = inst.depots[di].vehicles_available
        while len(sol.routes[di]) > limit:
            # 移除最短 (客户最少) 路由
            idx = min(range(len(sol.routes[di])),
                      key=lambda i: len(sol.routes[di][i]))
            removed = sol.routes[di].pop(idx)
            for ci in removed:
                _insert_cheapest_cost(sol, ci)
    return sol


def build_initial_solution(
    instance: Instance, rng: np.random.Generator, method: str = "nearest"
) -> Solution:
    """初始解构造工厂函数。"""
    if method == "nearest":
        return nearest_neighbor(instance, rng)
    elif method == "savings":
        return savings_method(instance, rng)
    elif method == "random":
        return random_insertion(instance, rng)
    else:
        raise ValueError(f"Unknown initial solution method: {method}")
