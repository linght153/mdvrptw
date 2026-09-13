"""pyvrp 结果 → 项目 Solution 转换回归测试。

背景: 曾因误以为 visits() 返回相对客户编号而重复加 offset 导致 IndexError。
实测 pyvrp 0.13 的 visits() 返回全局节点编号 (已含车场偏移)。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from src.core.objectives import calculate_cost
from src.solvers.pyvrp_solver import pyvrp_routes_to_solution, solve_with_pyvrp
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw


@pytest.fixture(scope="module")
def pr01_instance():
    text = (Path(__file__).resolve().parent.parent / "data/mdvrptw_raw/pr01.txt").read_text(encoding="utf-8")
    return instance_from_parsed(parse_mdvrptw(text), name="pr01")


def test_conversion_covers_all_customers_once(pr01_instance):
    """转换后所有客户恰好被服务一次 (无越界/重复)。"""
    result = solve_with_pyvrp(pr01_instance, max_runtime_seconds=5, seed=42)
    sol = pyvrp_routes_to_solution(result, pr01_instance)

    served = sol.all_served_customers()
    expected = set(c.index for c in pr01_instance.customers)
    assert served == expected
    # 客户索引都在合法范围 [num_depots, total_nodes)
    assert all(
        pr01_instance.num_depots <= v < pr01_instance.total_nodes for v in served
    )


def test_converted_solution_is_feasible(pr01_instance):
    """转换后的解满足容量 + 时间窗 + 覆盖 + 车辆数约束。

    注: 不含最大路线时长 (官方口径) — pyvrp shift_duration 用延迟出发口径,
    官方口径 duration 可能泄漏超时 (docs/durfix-spec-20260911.md §5,
    处置待泄漏判定文档, 不在此断言)。
    """
    result = solve_with_pyvrp(pr01_instance, max_runtime_seconds=5, seed=42)
    sol = pyvrp_routes_to_solution(result, pr01_instance)
    inst = pr01_instance
    assert sol.tw_violations() == []
    served = sol.all_served_customers()
    assert served == set(c.index for c in inst.customers)
    assert all(sol.route_load_fast(r) <= inst.vehicle_capacity
               for rl in sol.routes.values() for r in rl)
    assert all(len(rl) <= inst.depots[di].vehicles_available
               for di, rl in sol.routes.items())


def test_converted_cost_close_to_pyvrp(pr01_instance):
    """转换后成本与 pyvrp 报值一致 (容差: 整数取整误差)。"""
    result = solve_with_pyvrp(pr01_instance, max_runtime_seconds=5, seed=42)
    sol = pyvrp_routes_to_solution(result, pr01_instance)
    our_cost = calculate_cost(sol)
    pyvrp_cost = result.cost()
    # 相对差 < 1% (整数距离 vs 浮点欧氏)
    assert abs(our_cost - pyvrp_cost) / pyvrp_cost < 0.01
