"""Green MDVRP 解的表示。"""

import copy

from src.core.instance import Instance


def _route_start(inst: Instance, depot_idx: int, route_index: int):
    """该路由对应车辆的 (起始时刻, 起始节点) (静态 = 0/车场节点)。"""
    depot_node = inst.depots[depot_idx].index
    if inst.vehicle_start_times is None:
        return 0.0, depot_node
    st = inst.vehicle_start_times[depot_idx]
    sn = (inst.vehicle_start_nodes or [None] * len(inst.depots))[depot_idx]
    t0 = st[route_index] if route_index < len(st) else 0.0
    n0 = sn[route_index] if sn and route_index < len(sn) else depot_node
    return float(t0), int(n0)


def route_return_duration(inst: Instance, depot_idx: int, route: list[int],
                          route_index: int = 0) -> tuple[float, float]:
    """单条路线的 (回场时刻, 路线时长) — duration 唯一事实源。

    时长口径 (docs/durfix-spec-20260911.md §1): 按"车场 0 时刻出发"前向排程,
    duration = return_time − wait_{c1} (首客等待不计入, 其余等待计入);
    等价 Σ弧行驶 + Σ服务 + Σ_{j≥2} wait_{c_j}; 空路由 → 0。
    """
    if not route:
        return 0.0, 0.0
    depot_node = inst.depots[depot_idx].index
    t0, n0 = _route_start(inst, depot_idx, route_index)
    t = t0
    prev = n0
    first_wait = None
    for c_idx in route:
        cust = inst.customers[c_idx - inst.num_depots]
        t += inst.distance_matrix[prev][c_idx]
        wait = max(0.0, cust.ready_time - t)
        if first_wait is None:
            first_wait = wait
        t += wait + cust.service_time
        prev = c_idx
    return_time = t + inst.distance_matrix[prev][depot_node]
    duration = return_time - (first_wait if first_wait is not None else 0.0)
    return float(return_time), float(duration)


def duration_stats(sol) -> dict:
    """runner 输出共享辅助: 最大路线时长超出量 + 最大回场时刻。

    max_route_duration_excess: 0 = 全合规 (实例无 D 恒 0);
    max_return_time: 所有路由最大回场时刻 (空解 = 0)。
    """
    if sol is None:
        return {"max_route_duration_excess": None, "max_return_time": None}
    inst = sol.instance
    D = inst.route_duration_limit
    max_excess = 0.0
    max_return = 0.0
    for di, rl in sol.routes.items():
        for ri, route in enumerate(rl):
            ret, dur = route_return_duration(inst, di, route, ri)
            if ret > max_return:
                max_return = ret
            if D is not None and dur - D > max_excess:
                max_excess = dur - D
    return {
        "max_route_duration_excess": round(max_excess, 4),
        "max_return_time": round(max_return, 4),
    }


class Solution:
    """
    MDVRP 解的表示。

    routes[depot_index] = 路由列表
        每条路由 = 客户全局索引列表

    2 车场示例：
        routes = {
            0: [[5, 8, 12], [3, 7]],   # depot 0 有 2 条路由
            1: [[6, 9, 10, 11]],        # depot 1 有 1 条路由
        }
    """

    def __init__(self, instance: Instance, routes: dict[int, list[list[int]]] | None = None):
        self.instance = instance
        self.routes: dict[int, list[list[int]]] = routes or {}
        for d_idx in range(instance.num_depots):
            if d_idx not in self.routes:
                self.routes[d_idx] = []

    @property
    def num_depots(self) -> int:
        """快捷属性：车场数量（LLM 代码经常直接访问此属性）。"""
        return self.instance.num_depots

    def copy(self) -> "Solution":
        """深拷贝当前解。"""
        new_sol = Solution(self.instance)
        new_sol.routes = {
            d: [list(route) for route in routes]
            for d, routes in self.routes.items()
        }
        return new_sol

    def all_served_customers(self) -> set[int]:
        """返回所有已被服务的客户全局索引集合。"""
        served = set()
        for routes in self.routes.values():
            for route in routes:
                served.update(route)
        return served

    def unassigned_customers(self) -> list[int]:
        """返回尚未被服务的客户全局索引列表。"""
        served = self.all_served_customers()
        return [
            c.index
            for c in self.instance.customers
            if c.index not in served
        ]

    def num_vehicles(self, depot_index: int) -> int:
        """某车场使用的车辆数量。"""
        return len(self.routes.get(depot_index, []))

    def total_vehicles(self) -> int:
        """所有车场的车辆总数。"""
        return sum(len(routes) for routes in self.routes.values())

    def route_load_fast(self, route: list[int]) -> float:
        """一条路由的总需求量（快速直接查找）。"""
        return sum(self.instance.customers[i - self.instance.num_depots].demand for i in route)

    def _route_start(self, depot_index: int, route_index: int):
        """该路由对应车辆的 (起始时刻, 起始节点)。

        动态重优化时分车起始 (2026-09-01); 静态实例 (无
        vehicle_start_times) 返回 (0.0, 车场节点)。
        """
        inst = self.instance
        depot_node = inst.depots[depot_index].index
        if inst.vehicle_start_times is None:
            return 0.0, depot_node
        st = inst.vehicle_start_times[depot_index]
        sn = (inst.vehicle_start_nodes or [None] * len(inst.depots))[depot_index]
        t0 = st[route_index] if route_index < len(st) else 0.0
        n0 = sn[route_index] if sn and route_index < len(sn) else depot_node
        return float(t0), int(n0)

    def route_schedule(self, depot_index: int, route: list[int],
                       route_index: int = 0) -> list[dict]:
        """计算一条路由的时间表（服务开始时间, 含等待）。

        返回: [{"customer": idx, "arrival": t, "start": t', "wait": w}, ...]
        - arrival: 到达时间
        - start: 服务开始时间 = max(arrival, ready_time)
        - wait: 等待时间 = max(0, ready_time - arrival)
        时间基线 = 该车起始时刻 (默认 0 = 车场无 TW 假设 ready=0);
        起点节点 = 该车当前位置 (默认车场)。
        """
        inst = self.instance
        depot = inst.depots[depot_index]
        t0, n0 = self._route_start(depot_index, route_index)
        schedule = []
        t = t0
        prev = n0
        for c_idx in route:
            cust = inst.customers[c_idx - inst.num_depots]
            travel = inst.distance_matrix[prev][c_idx]
            arrival = t + travel
            start = max(arrival, cust.ready_time)
            wait = start - arrival
            schedule.append({
                "customer": c_idx,
                "arrival": float(arrival),
                "start": float(start),
                "wait": float(wait),
            })
            t = start + cust.service_time
            prev = c_idx
        return schedule

    def tw_violations(self) -> list[dict]:
        """返回所有时间窗违规: [{"customer": idx, "arrival": t, "due": due, "excess": e}]。

        仅当实例有 TW 时检查; 无 TW 返回空列表。
        """
        inst = self.instance
        if not inst.has_time_windows:
            return []
        violations = []
        for depot_idx, routes in self.routes.items():
            for r_idx, route in enumerate(routes):
                schedule = self.route_schedule(depot_idx, route, r_idx)
                for entry in schedule:
                    cust = inst.customers[entry["customer"] - inst.num_depots]
                    excess = entry["arrival"] - cust.due_time
                    if excess > 1e-9:
                        violations.append({
                            "customer": entry["customer"],
                            "arrival": entry["arrival"],
                            "due": cust.due_time,
                            "excess": excess,
                        })
        return violations

    def route_duration(self, depot_index: int, route: list[int],
                       route_index: int = 0) -> float:
        """单条路线时长 (口径 = return_time − wait_{c1}); 空路由 → 0。"""
        return route_return_duration(self.instance, depot_index, route,
                                     route_index)[1]

    def duration_violations(self) -> list[dict]:
        """路线时长违规列表: [{depot, route_index, duration, limit, excess}]。

        实例无 D 或无超长路线 → 空列表。
        """
        inst = self.instance
        if inst.route_duration_limit is None:
            return []
        limit = inst.route_duration_limit
        violations = []
        for depot_idx, routes in self.routes.items():
            for r_idx, route in enumerate(routes):
                _ret, dur = route_return_duration(inst, depot_idx, route, r_idx)
                if dur > limit + 1e-9:
                    violations.append({
                        "depot": depot_idx,
                        "route_index": r_idx,
                        "duration": dur,
                        "limit": limit,
                        "excess": dur - limit,
                    })
        return violations

    def is_feasible(self) -> bool:
        """
        检查所有可行性约束：
        1. 所有客户恰好被服务一次
        2. 任何路由不超车辆容量
        3. 时间窗硬约束（实例含 TW 时，TW 违规 → 不可行）
        """
        inst = self.instance

        # 所有客户必须被服务
        served = self.all_served_customers()
        expected = set(c.index for c in inst.customers)
        if served != expected:
            return False

        # 检查每条路由的容量
        cap = inst.max_vehicle_capacity
        for depot_idx, routes in self.routes.items():
            for route in routes:
                if self.route_load_fast(route) > cap:
                    return False

        # 车辆数硬约束: 每车场路由数 ≤ 可用车辆数
        # (曾缺失 — ALNS 产出车辆数违规解, 2026-09-01 P0 探针 SCP 抓到)
        for depot_idx, routes in self.routes.items():
            if len(routes) > inst.depots[depot_idx].vehicles_available:
                return False

        # 时间窗硬约束 (实例含 TW 时)
        if inst.has_time_windows and self.tw_violations():
            return False

        # 最大路线时长硬约束 (实例定义 D 时)
        if inst.route_duration_limit is not None and self.duration_violations():
            return False

        # 车场最迟返回硬约束 (depot.tw_late 非 None 时)
        for depot_idx, routes in self.routes.items():
            depot = inst.depots[depot_idx]
            if depot.tw_late is None:
                continue
            for r_idx, route in enumerate(routes):
                return_time, _ = route_return_duration(
                    inst, depot_idx, route, r_idx)
                if return_time > depot.tw_late + 1e-9:
                    return False

        return True

    def apply_destroy(self, destroy_fn, k, rng) -> tuple["Solution", list[int]]:
        """
        应用破坏算子。

        Returns:
            (部分解, 被移除的客户索引列表)
        """
        return destroy_fn(self, k, rng)

    def apply_repair(self, repair_fn, removed: list[int], rng) -> "Solution":
        """对部分解应用修复算子。"""
        return repair_fn(self, removed, rng)

    def __repr__(self):
        total_routes = self.total_vehicles()
        return f"Solution(depot_routes={total_routes}, feasible={self.is_feasible()})"
