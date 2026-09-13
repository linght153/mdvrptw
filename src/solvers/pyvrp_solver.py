"""pyvrp 0.13.4 求解器封装 — SOTA 基线。

将项目 Instance (含 TW) 转换为 pyvrp.ProblemData 并求解。
注意 pyvrp 0.13 API: Client(delivery=[...]), 距离/时间须为 int, 用 round_func 取整。
"""

from pathlib import Path

import numpy as np

from src.core.instance import Instance

try:
    from pyvrp import Client, Depot, ProblemData, VehicleType, read, solve
    from pyvrp.stop import MaxRuntime
except ImportError:  # pragma: no cover
    Client = Depot = ProblemData = VehicleType = read = solve = None
    MaxRuntime = None


def instance_to_pyvrp(inst: Instance, round_func="round") -> ProblemData:
    """将项目 Instance 转换为 pyvrp ProblemData。

    Args:
        inst: 项目实例 (可含 TW)
        round_func: 距离取整函数 ('round' 或可调用对象); pyvrp 距离矩阵需要整数

    Returns:
        pyvrp.ProblemData
    """
    # 车场 (索引 0..D-1 在前)
    depots = [
        Depot(
            x=d.x,
            y=d.y,
            tw_early=0,
            tw_late=int(1_000_000),
            service_duration=0,
        )
        for d in inst.depots
    ]
    # 客户 (索引 D..D+N-1)
    clients = [
        Client(
            x=c.x,
            y=c.y,
            delivery=[int(c.demand)],
            tw_early=int(c.ready_time),
            tw_late=int(c.due_time) if c.due_time < float("inf") else int(1_000_000),
            service_duration=int(c.service_time),
        )
        for c in inst.customers
    ]
    # 车辆类型: 每个车场 vehicles_available 辆车, 绑定出发/返回车场
    # (曾缺 start_depot → 全部默认 0, pyvrp 把车辆全部分配给车场 0,
    # 转换后解车辆数违规 — 2026-09-01 修复)
    per_depot = inst.depots[0].vehicles_available if inst.depots else 1
    # 最大路线时长 D: 实例定义时 shift_duration = int(D); 无 D → 1_000_000
    # (行为不变)。⚠️ pyvrp duration 语义 (延迟出发口径) 与官方口径不同, 见
    # docs/durfix-spec-20260911.md §5。
    shift_dur = (int(inst.route_duration_limit)
                 if inst.route_duration_limit is not None else int(1_000_000))
    vehicle_types = [
        VehicleType(
            num_available=per_depot,
            capacity=[int(inst.vehicle_capacity)],
            tw_early=0,
            tw_late=int(1_000_000),
            shift_duration=shift_dur,
            start_depot=d,
            end_depot=d,
        )
        for d in range(inst.num_depots)
    ]

    # 距离矩阵 → int (pyvrp 需要整数; round 与 Cordeau 基准一致)
    dist = np.asarray(inst.distance_matrix, dtype=float)
    dist_int = np.rint(dist).astype(int)

    return ProblemData(
        clients=clients,
        depots=depots,
        vehicle_types=vehicle_types,
        distance_matrices=[dist_int],
        duration_matrices=[dist_int],
    )


def solve_with_pyvrp(
    inst: Instance,
    max_runtime_seconds: float = 30.0,
    seed: int = 42,
    initial_solution=None,
):
    """用 pyvrp 求解实例。

    Args:
        inst: 项目实例
        max_runtime_seconds: 求解时间预算
        seed: 随机种子
        initial_solution: 可选的热启动解 (pyvrp Solution)

    Returns:
        PyVrpResult 包装: 内含 .best (pyvrp Solution), .cost(), .is_feasible(),
        .routes(), .num_routes
    """
    data = instance_to_pyvrp(inst)
    stop = MaxRuntime(max_runtime_seconds)
    result = solve(data, stop=stop, seed=seed, display=False, initial_solution=initial_solution)
    return PyVrpResult(result)


class PyVrpResult:
    """pyvrp Result 的轻量包装, 暴露求解测试需要的属性。"""

    def __init__(self, result):
        self._result = result
        self.best = result.best

    def is_feasible(self) -> bool:
        return self.best.is_feasible()

    def cost(self) -> float:
        return float(self.best.distance_cost())

    @property
    def num_routes(self) -> int:
        return self.best.num_routes()

    def routes(self):
        return self.best.routes()


def pyvrp_routes_to_solution(result, inst: Instance):
    """将 pyvrp 结果路由转换为项目 Solution。

    pyvrp 0.13 的 route.visits() 返回全局节点编号 (已含车场偏移,
    范围 = [num_depots, total_nodes)), 与项目 Solution 的客户索引约定一致,
    无需再加 offset。
    """
    from src.core.solution import Solution

    routes: dict[int, list[list[int]]] = {d: [] for d in range(inst.num_depots)}
    for route in result.routes():
        visits = list(route.visits())
        if not visits:
            continue
        depot_idx = int(route.start_depot())
        routes[depot_idx].append([int(v) for v in visits])
    return Solution(inst, routes)
