"""pyvrp 求解器封装测试。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from src.solvers.pyvrp_solver import (
    instance_to_pyvrp,
    solve_with_pyvrp,
)
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw


@pytest.fixture(scope="module")
def pr01_instance():
    text = (Path(__file__).resolve().parent.parent / "data/mdvrptw_raw/pr01.txt").read_text(encoding="utf-8")
    return instance_from_parsed(parse_mdvrptw(text), name="pr01")


def test_instance_to_pyvrp_basic(pr01_instance):
    """转换后 ProblemData 规模正确。"""
    data = instance_to_pyvrp(pr01_instance)
    assert data.num_clients == pr01_instance.num_customers
    assert data.num_depots == pr01_instance.num_depots
    assert data.num_vehicles == (
        pr01_instance.num_depots * pr01_instance.depots[0].vehicles_available
    )


def test_solve_with_pyvrp_returns_feasible(pr01_instance):
    """pyvrp 求解返回可行解 (成本 > 0 且有限)。"""
    result = solve_with_pyvrp(
        pr01_instance,
        max_runtime_seconds=10,
        seed=42,
    )
    assert result.is_feasible()
    cost = result.cost()
    assert cost > 0
    assert np.isfinite(cost)
    assert result.num_routes > 0


def test_solve_deterministic_same_seed(pr01_instance):
    """同 seed 结果可复现。"""
    a = solve_with_pyvrp(pr01_instance, max_runtime_seconds=5, seed=7)
    b = solve_with_pyvrp(pr01_instance, max_runtime_seconds=5, seed=7)
    assert a.cost() == pytest.approx(b.cost(), rel=1e-6)


def test_solve_returns_route_structure(pr01_instance):
    """结果可转回我们的 Solution 结构。"""
    result = solve_with_pyvrp(pr01_instance, max_runtime_seconds=5, seed=42)
    routes = result.routes()
    assert len(routes) >= 1
    # 所有客户恰好被访问一次
    visited = [c for route in routes for c in route]
    assert len(visited) == len(set(visited)) == pr01_instance.num_customers
