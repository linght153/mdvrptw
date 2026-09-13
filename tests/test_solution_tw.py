"""Solution 时间窗可行性校验测试。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from src.core.solution import Solution
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw


@pytest.fixture(scope="module")
def pr01_instance():
    text = (Path(__file__).resolve().parent.parent / "data/mdvrptw_raw/pr01.txt").read_text(encoding="utf-8")
    return instance_from_parsed(parse_mdvrptw(text), name="pr01")


def _solution_with_routes(inst, routes):
    """构造解并手动把客户放进指定车场路由。"""
    return Solution(inst, routes)


def test_no_tw_instance_always_tw_feasible():
    """无 TW 实例: tw_violations 为空 (legacy 行为不受影响)"""
    from src.core.instance import load_instance

    inst = load_instance("p01")
    # 用最近邻初始解
    from src.core.initial_solution import nearest_neighbor

    sol = nearest_neighbor(inst, __import__("numpy").random.default_rng(42))
    assert sol.tw_violations() == []


def _make_tw_instance():
    """构造一个人为的紧 TW 实例: 1 车场 + 2 客户, 距离 10, TW 紧。

    depot0 在 (0,0); 客户 A (10,0) ready=0 due=5 (违反: 到达即 10 > 5);
    客户 B (20,0) ready=0 due=100 (宽松)。
    """
    from src.core.instance import Customer, Depot, EmissionModel, Instance

    depots = [Depot(index=0, id=0, x=0.0, y=0.0, vehicles_available=2, capacity=100.0)]
    customers = [
        Customer(index=1, id=1, x=10.0, y=0.0, demand=1.0, assigned_depot_index=0,
                 ready_time=0.0, due_time=5.0, service_time=0.0),
        Customer(index=2, id=2, x=20.0, y=0.0, demand=1.0, assigned_depot_index=0,
                 ready_time=0.0, due_time=100.0, service_time=0.0),
    ]
    nodes = [(d.x, d.y) for d in depots] + [(c.x, c.y) for c in customers]
    n = len(nodes)
    import math
    dist = [[math.dist(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]
    return Instance(
        name="tiny_tw", source="test", num_depots=1, num_customers=2,
        vehicle_capacity=100.0, depots=depots, customers=customers,
        distance_matrix=dist, emission_model=EmissionModel(),
    )


def test_tw_violation_after_due():
    """紧 TW: 直接车场→A 距离 10 > due 5 → 违规检出"""
    inst = _make_tw_instance()
    sol = Solution(inst, {0: [[1]]})  # 只服务客户 A
    violations = sol.tw_violations()
    assert len(violations) == 1
    assert violations[0]["customer"] == 1
    assert violations[0]["excess"] == pytest.approx(5.0)  # 10 - 5


def test_tw_feasible_route_passes():
    """宽松客户 B (due=100) 不违规"""
    inst = _make_tw_instance()
    sol = Solution(inst, {0: [[2]]})  # 只服务客户 B
    assert sol.tw_violations() == []


def test_route_arrival_time_computation(pr01_instance):
    """route_schedule: 计算每个客户的服务开始时间 (含等待)"""
    inst = pr01_instance
    c0, c1 = inst.customers[0], inst.customers[1]
    depot = inst.depots[0]
    route = [c0.index, c1.index]
    sol = Solution(inst, {0: [route]})

    schedule = sol.route_schedule(0, route)
    assert len(schedule) == len(route)
    # 第一个客户: arrival = dist(depot, c0), service_start = max(arrival, ready)
    d_depot_c0 = inst.distance_matrix[depot.index][c0.index]
    assert schedule[0]["arrival"] == pytest.approx(d_depot_c0)
    assert schedule[0]["start"] == pytest.approx(max(d_depot_c0, c0.ready_time))
    # 时间单调不减
    starts = [s["start"] for s in schedule]
    assert starts == sorted(starts)
