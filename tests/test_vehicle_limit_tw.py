"""TW × 车辆数紧绑定回归测试 (2026-09-02 探针发现 pr11 车辆数违规)。

背景: pr11 (每车场 1 辆车) 上 ALNS 最终解 depot3 出现 2 条路由 (车辆数违规):
1. repair._insert_cheapest_cost_tw 兜底在"所有车场车辆已满"时无条件就近
   append 新路由 → 超车辆上限;
2. engine 主循环非 LS 迭代只 _enforce_capacity, 车辆数违规漏进档案
   (LS 后的 _enforce_feasibility 才管车辆数)。
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import Customer, Depot, EmissionModel, Instance
from src.core.solution import Solution
from src.operators.repair import greedy_cost_tw_insertion, _lightest_route_insert
from src.solvers.standard_alns import solve_standard_alns


@pytest.fixture(scope="module")
def tight_tw_inst():
    """2 车场各 1 辆车 (容量 100): 车满 + 无 TW 可行位时兜底不得增车。

    客户分两簇 (近车场0 / 近车场1), 各 5 客户需求 30 → 单路由最多 3 客户,
    5 客户必须分 2 条路由 → 车辆数必然不够 → 触发兜底路径。
    """
    depots = [
        Depot(0, 0, 0.0, 0.0, 1, 100.0),
        Depot(1, 1, 100.0, 0.0, 1, 100.0),
    ]
    custs = []
    for i in range(5):   # 近车场0 (客户 index 2-6)
        custs.append(Customer(2 + i, 2 + i, 8.0 + i, 2.0, 30.0, 0,
                              ready_time=0.0, due_time=60.0, service_time=5.0))
    for i in range(5):   # 近车场1 (客户 index 7-11)
        custs.append(Customer(7 + i, 7 + i, 92.0 + i, 2.0, 30.0, 0,
                              ready_time=0.0, due_time=60.0, service_time=5.0))
    nodes = [(0.0, 0.0), (100.0, 0.0)] + [(c.x, c.y) for c in custs]
    n = len(nodes)
    dist = [[math.dist(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]
    return Instance("tight_tw", "test", 2, 10, 100.0, depots, custs, dist,
                    EmissionModel())


def test_repair_fallback_never_exceeds_vehicle_limit(tight_tw_inst):
    """车满 + 容量全拒 (需求 30 插不进负载 90 的路由) → 兜底不增车, 客户仍被放置。"""
    inst = tight_tw_inst
    # 车场0 1 条路由 (3 客户, 负载 90), 车场1 1 条路由 (3 客户) → 两车场均已达 1 辆上限
    depot0_custs = [c.index for c in inst.customers if c.index <= 6][:3]
    depot1_custs = [c.index for c in inst.customers if c.index >= 7][:3]
    sol = Solution(inst, {0: [depot0_custs], 1: [depot1_custs]})
    # 剩余 4 客户需求各 30: 90+30=120 > 容量 100 → 所有路由容量全拒
    remaining = [c.index for c in inst.customers
                 if c.index not in depot0_custs and c.index not in depot1_custs]
    assert len(remaining) == 4
    rebuilt = greedy_cost_tw_insertion(sol, remaining, np.random.default_rng(0))
    for d in range(inst.num_depots):
        assert len(rebuilt.routes[d]) <= inst.depots[d].vehicles_available, \
            f"车场{d} 路由数 {len(rebuilt.routes[d])} > 上限 1 (兜底增车)"
    assert len(rebuilt.all_served_customers()) == 10, "全部客户必须被放置 (兜底插入)"


def test_alns_tight_tw_final_archive_feasible():
    """pr11 (4 车场各 1 辆, 48 客户宽窗) 300 迭代: 档案全部可行 (车辆数合规)。

    修复前: 最终 best 解 depot3 2 条路由 (车辆数违规, is_feasible=False)。
    """
    from src.core.instance import instance_from_parsed
    from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

    text = open(Path(__file__).resolve().parent.parent / "data/mdvrptw_raw/pr11.txt",
                encoding="utf-8").read()
    inst = instance_from_parsed(parse_mdvrptw(text), name="pr11")
    cfg = {
        "max_iterations": 300, "max_time_seconds": 1e9, "stagnation_limit": 300,
        "segment_size": 100, "reaction_factor": 0.1, "decay_factor": 1.0,
        "min_selection_prob": 0.005, "initial_temperature": None,
        "cooling_rate": None, "archive_capacity": 20,
        "parent_selection": "crowding", "initial_solution": "nearest",
        "k_min_ratio": 0.10, "k_max_ratio": 0.40, "sigma": 3.0,
        "local_search_max_iter": 10, "local_search_freq": 5,
        "local_search_on_accept": True, "local_search_final": False,
    }
    archive = solve_standard_alns(inst, cfg, np.random.default_rng(42))
    assert archive.entries, "档案为空"
    for e in archive.entries:
        sol = e[1]["solution"]
        assert sol.is_feasible(), \
            f"档案含不可行解 cost={e[0][0]}: 车辆数=" \
            f"{[len(rl) for rl in sol.routes.values()]} 上限=" \
            f"{[d.vehicles_available for d in inst.depots]}"
        for d in range(inst.num_depots):
            assert len(sol.routes[d]) <= inst.depots[d].vehicles_available
