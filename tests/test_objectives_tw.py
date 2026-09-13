"""目标函数 TW 软惩罚测试。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import Customer, Depot, EmissionModel, Instance
from src.core.objectives import calculate_objectives, calculate_objectives_tw
from src.core.solution import Solution


def _make_tw_instance():
    """紧 TW 实例: depot(0,0); A(10,0) due=5(违规); B(20,0) due=100(宽松)。"""
    import math

    depots = [Depot(index=0, id=0, x=0.0, y=0.0, vehicles_available=2, capacity=100.0)]
    customers = [
        Customer(index=1, id=1, x=10.0, y=0.0, demand=1.0, assigned_depot_index=0,
                 ready_time=0.0, due_time=5.0, service_time=0.0),
        Customer(index=2, id=2, x=20.0, y=0.0, demand=1.0, assigned_depot_index=0,
                 ready_time=0.0, due_time=100.0, service_time=0.0),
    ]
    nodes = [(d.x, d.y) for d in depots] + [(c.x, c.y) for c in customers]
    n = len(nodes)
    dist = [[math.dist(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]
    return Instance(
        name="tiny_tw", source="test", num_depots=1, num_customers=2,
        vehicle_capacity=100.0, depots=depots, customers=customers,
        distance_matrix=dist, emission_model=EmissionModel(),
    )


def test_calculate_objectives_unchanged_for_no_tw():
    """无 TW: calculate_objectives 行为不变 (返回二元组)"""
    from src.core.instance import load_instance
    from src.core.initial_solution import nearest_neighbor
    import numpy as np

    inst = load_instance("p01")
    sol = nearest_neighbor(inst, np.random.default_rng(42))
    result = calculate_objectives(sol)
    assert isinstance(result, tuple) and len(result) == 2


def test_calculate_objectives_tw_returns_triple():
    """有 TW: calculate_objectives_tw 返回 (cost, emission, tw_penalty)"""
    inst = _make_tw_instance()
    sol = Solution(inst, {0: [[1]]})  # 服务违规客户 A
    cost, emission, penalty = calculate_objectives_tw(sol, tw_penalty_weight=10.0)
    # 违规 excess = 10 - 5 = 5, penalty = 10 * 5 = 50
    assert penalty == pytest.approx(50.0)
    assert cost > 0 and emission > 0


def test_tw_penalty_zero_when_no_violation():
    """无违规: tw_penalty = 0"""
    inst = _make_tw_instance()
    sol = Solution(inst, {0: [[2]]})  # 宽松客户 B
    _, _, penalty = calculate_objectives_tw(sol, tw_penalty_weight=10.0)
    assert penalty == 0.0


def test_tw_penalty_weight_zero_disables():
    """tw_penalty_weight=0 (默认): 惩罚为 0, 与 calculate_objectives 兼容"""
    inst = _make_tw_instance()
    sol = Solution(inst, {0: [[1]]})
    _, _, penalty = calculate_objectives_tw(sol)  # 默认 weight=0
    assert penalty == 0.0
