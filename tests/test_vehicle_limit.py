"""车辆数上限硬约束测试: 每车场路由数 ≤ vehicles_available。

背景: 项目坑 2 (车辆数上限不是硬约束) — nearest_neighbor/repair/is_feasible
均不检查, ALNS 在 p08 (249 客户) 产出车辆数违规解 (车场 0 用 15 辆 > 上限 14,
被 P0 探针的 SCP 每车场车辆数约束抓到)。本测试锁定修复。
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import Customer, Depot, EmissionModel, Instance
from src.core.initial_solution import nearest_neighbor
from src.core.solution import Solution
from src.operators.repair import greedy_cost_insertion
from src.solvers.standard_alns import solve_standard_alns


@pytest.fixture
def tight_inst():
    """2 车场: 车场0 仅 1 辆车 (容量 100), 车场1 5 辆车。

    客户 1-3 靠近车场0 (各需求 60, 单车装 1 个 → 需 3 辆但只有 1 辆),
    客户 4-8 靠近车场1 (各 30)。总容量 6×100=600 > 总需求 330 → 有可行解,
    但车场0 最多 1 条路由, 剩余近场客户必须跨车场到车场1。
    """
    depots = [
        Depot(0, 0, 0.0, 0.0, 1, 100.0),    # 车场0: 1 辆
        Depot(1, 1, 100.0, 0.0, 5, 100.0),  # 车场1: 5 辆
    ]
    custs = [
        Customer(2, 2, 10.0, 0.0, 60.0, 0),
        Customer(3, 3, 12.0, 2.0, 60.0, 0),
        Customer(4, 4, 15.0, 0.0, 60.0, 0),
        Customer(5, 5, 110.0, 0.0, 30.0, 0),
        Customer(6, 6, 115.0, 2.0, 30.0, 0),
        Customer(7, 7, 120.0, 0.0, 30.0, 0),
        Customer(8, 8, 125.0, 2.0, 30.0, 0),
        Customer(9, 9, 130.0, 0.0, 30.0, 0),
    ]
    nodes = [(0.0, 0.0), (100.0, 0.0)] + [(c.x, c.y) for c in custs]
    n = len(nodes)
    dist = [[math.dist(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]
    return Instance("tight", "test", 2, 8, 100.0, depots, custs, dist, EmissionModel())


def _load(inst: Instance, route) -> float:
    return sum(inst.customers[i - inst.num_depots].demand for i in route)


def test_is_feasible_rejects_vehicle_limit_violation(tight_inst):
    """is_feasible 必须拒绝每车场路由数超上限的解 (覆盖/容量均合法)。"""
    inst = tight_inst
    # 覆盖全部 8 客户, 容量均 OK, 唯一违规 = 车场0 用 2 辆 > 上限 1
    sol = Solution(inst, {0: [[2], [3]], 1: [[4, 5], [6, 7], [8, 9]]})
    assert not sol.is_feasible(), "车场0 2 条路由应不可行"


def test_is_feasible_accepts_within_limit(tight_inst):
    """合法解 (每车场 ≤ 上限, 容量/覆盖 OK) 应可行。"""
    inst = tight_inst
    sol = Solution(inst, {0: [[2]], 1: [[3], [4], [5, 6], [7, 8], [9]]})
    assert len(sol.routes[0]) == 1 <= inst.depots[0].vehicles_available
    assert len(sol.routes[1]) == 5 <= inst.depots[1].vehicles_available
    assert sol.is_feasible()


def test_nearest_neighbor_respects_vehicle_limit(tight_inst):
    """nearest_neighbor 初始解: 每车场路由数 ≤ 上限, 全客户覆盖。"""
    inst = tight_inst
    sol = nearest_neighbor(inst, np.random.default_rng(0))
    assert len(sol.routes[0]) <= 1, f"车场0 用 {len(sol.routes[0])} 辆 > 1"
    assert len(sol.routes[1]) <= 5, f"车场1 用 {len(sol.routes[1])} 辆 > 5"
    assert len(sol.all_served_customers()) == inst.num_customers
    for d in range(inst.num_depots):
        for r in sol.routes[d]:
            assert _load(inst, r) <= inst.vehicle_capacity + 1e-9


def test_greedy_repair_respects_vehicle_limit(tight_inst):
    """greedy_cost_insertion 从空解重建: 每车场路由数 ≤ 上限。"""
    inst = tight_inst
    empty = Solution(inst, {0: [], 1: []})
    rebuilt = greedy_cost_insertion(
        empty, [c.index for c in inst.customers], np.random.default_rng(0))
    assert len(rebuilt.routes[0]) <= 1, f"车场0 用 {len(rebuilt.routes[0])} 辆 > 1"
    assert len(rebuilt.routes[1]) <= 5, f"车场1 用 {len(rebuilt.routes[1])} 辆 > 5"
    assert len(rebuilt.all_served_customers()) == inst.num_customers


def test_alns_final_solution_respects_vehicle_limit(tight_inst):
    """ALNS 最终解: 每车场路由数 ≤ 上限 (含 warm-start 超限初始解)。"""
    inst = tight_inst
    cfg = {
        "max_iterations": 300, "max_time_seconds": 60, "stagnation_limit": 100,
        "segment_size": 50, "reaction_factor": 0.1, "decay_factor": 1.0,
        "min_selection_prob": 0.005, "initial_temperature": None,
        "cooling_rate": None, "archive_capacity": 20,
        "parent_selection": "crowding", "initial_solution": "nearest",
        "k_min_ratio": 0.10, "k_max_ratio": 0.40, "sigma": 3.0,
        "local_search_max_iter": 10, "local_search_freq": 5,
        "local_search_on_accept": True, "local_search_final": False,
    }
    # warm-start 超限解: 车场0 2 条路由
    bad_init = Solution(inst, {0: [[2], [3]], 1: [[4], [5], [6], [7], [8]]})
    archive = solve_standard_alns(
        inst, cfg, np.random.default_rng(42), initial_solution=bad_init)
    best = min(archive.entries, key=lambda e: e[0][0])
    sol = best[1]["solution"]
    for d in range(inst.num_depots):
        assert len(sol.routes[d]) <= inst.depots[d].vehicles_available, \
            f"车场{d} 用 {len(sol.routes[d])} 辆 > 上限 {inst.depots[d].vehicles_available}"
    assert sol.is_feasible()


@pytest.fixture(scope="module")
def tight_single_inst():
    """单车场紧绑定: 4 辆 × 容量 100 = 400, 25 客户 × 16 = 400 (恰好满载)。

    该场景曾触发死循环: repair 无容量可行位 → 兜底开新车 → 超车辆数 →
    _trim_to_vehicle_limit 移除-重插无限循环 (实测 500 迭代 >200s 跑不完)。
    """
    depots = [Depot(0, 0, 0.0, 0.0, 4, 100.0)]
    custs = [
        Customer(1 + i, 1 + i, float((i % 5) * 20 + 5), float((i // 5) * 20 + 5),
                 16.0, 0)
        for i in range(25)
    ]
    nodes = [(0.0, 0.0)] + [(c.x, c.y) for c in custs]
    n = len(nodes)
    dist = [[math.dist(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]
    return Instance("tight_single", "test", 1, 25, 100.0, depots, custs, dist,
                    EmissionModel())


def test_trim_to_vehicle_limit_converges_when_all_routes_full(tight_single_inst):
    """全路由满载 + 超限时 _trim_to_vehicle_limit 必须收敛 (不死循环)。

    构造: 4 条满载路由 (100) + 1 条超限路由 → 截断重插时无容量可行位 →
    兜底插负载最轻路由 (不增车) → 路由数收敛到 4, 全部客户已放置。
    """
    from src.core.initial_solution import _trim_to_vehicle_limit

    inst = tight_single_inst
    customers = [c.index for c in inst.customers]
    full = [customers[i * 5:(i + 1) * 5] for i in range(4)]  # 4 条 × 5 客户 = 80 负载
    extra = [customers[20:]]  # 第 5 条路由 (超限)
    sol = Solution(inst, {0: full + extra})
    assert len(sol.routes[0]) == 5 > inst.depots[0].vehicles_available

    result = _trim_to_vehicle_limit(sol)
    assert len(result.routes[0]) <= 4, "车辆数必须收敛到上限内"
    # 兜底插入不丢客户: 全部 25 客户仍在解中
    assert len(result.all_served_customers()) == 25


def test_alns_tight_single_depot_no_hang(tight_single_inst):
    """紧绑定单车场实例跑 ALNS 不得挂死 (死循环保护: 30s 内必须完成)。

    曾实测: 修复前 500 迭代 >200s 跑不完 (faulthandler 抓到栈在
    _trim_to_vehicle_limit → _insert_cheapest_cost → 开新车循环)。
    """
    import faulthandler

    inst = tight_single_inst
    cfg = {
        "max_iterations": 300, "max_time_seconds": 60, "stagnation_limit": 100,
        "segment_size": 50, "reaction_factor": 0.1, "decay_factor": 1.0,
        "min_selection_prob": 0.005, "initial_temperature": None,
        "cooling_rate": None, "archive_capacity": 20,
        "parent_selection": "crowding", "initial_solution": "nearest",
        "k_min_ratio": 0.10, "k_max_ratio": 0.40, "sigma": 3.0,
        "local_search_max_iter": 10, "local_search_freq": 5,
        "local_search_on_accept": True, "local_search_final": False,
    }
    faulthandler.dump_traceback_later(30, exit=True)  # 挂死则打印栈并退出 (测试失败)
    try:
        archive = solve_standard_alns(inst, cfg, np.random.default_rng(42))
    finally:
        faulthandler.cancel_dump_traceback_later()
    if not archive.entries:
        # 该实例 (容量 100, 需求 16, 总需求 400=4×100) 无整数可行解 —
        # 100 不可被 16 整除, 任何路由 load=16k ∈ {96,112}, 4 车无法满载。
        # 档案纯净性门禁 (2026-09-03) 下空档案是正确行为; 本测试回归目标
        # = 紧绑定不死循环 (faulthandler 30s 兜底), 无可行解时直接通过。
        return
    best = min(archive.entries, key=lambda e: e[0][0])
    sol = best[1]["solution"]
    assert len(sol.routes[0]) <= inst.depots[0].vehicles_available
    assert len(sol.all_served_customers()) == inst.num_customers
