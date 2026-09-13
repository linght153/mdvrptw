"""TW-aware repair 算子测试。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import Customer, Depot, EmissionModel, Instance
from src.core.solution import Solution
from src.operators.repair import greedy_cost_insertion, greedy_cost_tw_insertion


def _make_tw_instance():
    """紧 TW 实例, 制造"插入位置影响 TW 违规"的场景。

    depot(0,0); A(10,0) due=24 service=0; C(15,0) due=15.5 service=10。
    - 先插入 A (到达 10 < 24, 可行)
    - 再插入 C:
        C 在 A 前: C 到达 15 可行(start=15), 服务 10 后 t=25, A 到达 30 > 24 → 违规
        C 在 A 后: A 到达 10 可行, C 到达 15 < 15.5 → 全部可行
    """
    import math

    depots = [Depot(index=0, id=0, x=0.0, y=0.0, vehicles_available=2, capacity=100.0)]
    customers = [
        Customer(index=1, id=1, x=10.0, y=0.0, demand=1.0, assigned_depot_index=0,
                 ready_time=0.0, due_time=24.0, service_time=0.0),
        Customer(index=2, id=2, x=20.0, y=0.0, demand=1.0, assigned_depot_index=0,
                 ready_time=0.0, due_time=100.0, service_time=0.0),
        Customer(index=3, id=3, x=15.0, y=0.0, demand=1.0, assigned_depot_index=0,
                 ready_time=0.0, due_time=15.5, service_time=10.0),
    ]
    nodes = [(d.x, d.y) for d in depots] + [(c.x, c.y) for c in customers]
    n = len(nodes)
    dist = [[math.dist(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]
    return Instance(
        name="tiny_tw", source="test", num_depots=1, num_customers=3,
        vehicle_capacity=100.0, depots=depots, customers=customers,
        distance_matrix=dist, emission_model=EmissionModel(),
        tw_penalty_weight=10.0,
    )


def test_tw_repair_prefers_tw_feasible_route():
    """TW-aware repair 应把 C 放在 A 之后 (避免 A 因 C 的长服务而违规)。"""
    inst = _make_tw_instance()
    sol = Solution(inst, {0: [[1]]})  # 已插入 A
    repaired = greedy_cost_tw_insertion(sol.copy(), [3], np.random.default_rng(0))
    # C 应插入在 A 之后 → 无违规
    assert repaired.tw_violations() == []
    assert repaired.routes[0][0] == [1, 3]


def test_tw_repair_versus_plain_greedy():
    """同样场景下, plain greedy 距离相等选 pos=0 (违规), TW-aware 选 pos=1 (可行)。"""
    inst = _make_tw_instance()
    sol = Solution(inst, {0: [[1]]})

    plain = greedy_cost_insertion(sol.copy(), [3], np.random.default_rng(0))
    tw = greedy_cost_tw_insertion(sol.copy(), [3], np.random.default_rng(0))

    assert len(tw.tw_violations()) <= len(plain.tw_violations())
    # plain 在距离相同下选第一个位置 (C 在 A 前 → 违规)
    assert plain.routes[0][0] == [3, 1]
    assert len(plain.tw_violations()) == 1


def test_tw_repair_no_tw_instance_unchanged():
    """无 TW 实例: TW-aware 与 plain greedy 结果一致。"""
    from src.core.instance import load_instance

    inst = load_instance("p01")
    from src.core.initial_solution import nearest_neighbor

    sol = nearest_neighbor(inst, np.random.default_rng(1))
    # 从现有解中移除一个客户 (从第一个非空路由)
    served = sorted(sol.all_served_customers())
    removed = [served[0]]
    partial = sol.copy()
    for routes in partial.routes.values():
        for route in routes:
            if removed[0] in route:
                route.remove(removed[0])
                break

    a = greedy_cost_tw_insertion(partial.copy(), list(removed), np.random.default_rng(5))
    b = greedy_cost_insertion(partial.copy(), list(removed), np.random.default_rng(5))
    assert a.routes == b.routes
