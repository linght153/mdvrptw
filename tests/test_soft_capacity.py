"""软容量搜索 (探针 2, 2026-09-04) 回归测试。

背景: 容量锁死假设 → 软容量穿越 A/B。机制 = config 键 soft_capacity (默认 False):
repair 容量门放宽 + 接受目标罚后化 + 档案仍只收可行解 + 周期修复播种。
红线: 默认配置行为必须逐位不变 (test_default_behavior_lock)。
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace as dc_replace

from src.core.instance import Customer, Depot, EmissionModel, Instance
from src.solvers.standard_alns import solve_standard_alns
from src.alns.selection import SegmentRewardSelection
from src.alns.engine import ALNSEngine
from src.operators.local_search import rvnd_improve
from src.operators.repair import greedy_cost_tw_insertion


def _build_lock_inst():
    """tight 风格可行实例: 2 车场各 1 车 (容量 100), 客户 2 簇各 5, 需求 15。

    总需求 150 ≤ 2×100 → 有可行解; 每簇 5×15=75 ≤ 100 → 单车可服务一簇。
    用于默认行为锁存 (改动前快照: 300 iter s42 → best 41.130084, 档案 1 条)。
    """
    depots = [Depot(0, 0, 0.0, 0.0, 1, 100.0), Depot(1, 1, 100.0, 0.0, 1, 100.0)]
    custs = []
    for i in range(5):
        custs.append(Customer(2 + i, 2 + i, 8.0 + i, 2.0, 15.0, 0,
                              ready_time=0.0, due_time=60.0, service_time=5.0))
    for i in range(5):
        custs.append(Customer(7 + i, 7 + i, 92.0 + i, 2.0, 15.0, 0,
                              ready_time=0.0, due_time=60.0, service_time=5.0))
    nodes = [(0.0, 0.0), (100.0, 0.0)] + [(c.x, c.y) for c in custs]
    n = len(nodes)
    dist = [[math.dist(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]
    return Instance("tight_lock", "test", 2, 10, 100.0, depots, custs, dist,
                    EmissionModel())


@pytest.fixture(scope="module")
def lock_inst():
    return _build_lock_inst()


LOCK_CONFIG = {
    "max_iterations": 300, "max_time_seconds": 1e9, "stagnation_limit": 100000,
    "segment_size": 100, "reaction_factor": 0.1, "decay_factor": 1.0,
    "min_selection_prob": 0.005, "initial_temperature": None, "cooling_rate": None,
    "archive_capacity": 20, "parent_selection": "crowding",
    "initial_solution": "nearest", "k_min_ratio": 0.10, "k_max_ratio": 0.40,
    "sigma": 3.0, "local_search_max_iter": 10, "local_search_freq": 5,
    "local_search_on_accept": True, "local_search_final": False,
}


def _run_engine(inst, config, seed):
    """装配 TW 实例引擎 (与 solve_standard_alns 同构), 返回 (archive, engine)。"""
    customers_copy = [dc_replace(c) for c in inst.customers]
    inst = dc_replace(inst, customers=customers_copy)
    from src.operators.destroy import DESTROY_OPERATORS
    from src.operators.repair import greedy_cost_tw_insertion as tw_repair

    destroy_names = list(DESTROY_OPERATORS.keys())
    selector = SegmentRewardSelection(
        destroy_names, ["greedy_cost_tw"],
        segment_size=config.get("segment_size", 100),
        reaction_factor=config.get("reaction_factor", 0.1),
        decay_factor=config.get("decay_factor", 1.0),
        min_selection_prob=config.get("min_selection_prob", 0.005),
    )
    engine = ALNSEngine(
        instance=inst, config=config, rng=np.random.default_rng(seed),
        selector=selector, repair_ops={"greedy_cost_tw": tw_repair},
        local_search=rvnd_improve,
    )
    archive = engine.run()
    return archive, engine


def _best(archive):
    feas = [(e[0][0], e[1]["solution"]) for e in archive.entries
            if e[1]["solution"].is_feasible()]
    if not feas:
        return None, None
    c = min(c for c, _ in feas)
    sol = next(s for c2, s in feas if abs(c2 - c) < 1e-9)
    return c, sol


def test_default_behavior_lock(lock_inst):
    """默认 (无 soft 键) 行为锁存 — 与 2026-09-04 改动前快照逐位一致。

    改动前快照 (300 iter, s42): best_cost=41.130084, archive_size=1,
    feasible, tw_viol=0, served=10。
    """
    archive = solve_standard_alns(lock_inst, dict(LOCK_CONFIG),
                                  np.random.default_rng(42))
    best_cost, sol = _best(archive)
    assert best_cost == pytest.approx(41.130084, abs=1e-6)
    assert archive.size == 1
    assert sol.is_feasible()
    assert len(sol.tw_violations()) == 0
    assert len(sol.all_served_customers()) == 10


def test_soft_capacity_false_equals_default(lock_inst):
    """soft_capacity=False 显式设置与默认无键逐位一致。"""
    cfg = dict(LOCK_CONFIG, soft_capacity=False)
    a1 = solve_standard_alns(lock_inst, dict(LOCK_CONFIG),
                             np.random.default_rng(42))
    a2 = solve_standard_alns(lock_inst, cfg, np.random.default_rng(42))
    c1, _ = _best(a1)
    c2, _ = _best(a2)
    assert c1 == pytest.approx(c2, abs=1e-9)


def test_soft_run_pure_archive_and_deterministic(lock_inst):
    """软容量开启: 运行完整 + 档案纯净 (全可行/全覆盖/TW 零违规) + 同 seed 可复现。"""
    cfg = dict(LOCK_CONFIG, soft_capacity=True)
    results = []
    for _ in range(2):
        archive, engine = _run_engine(lock_inst, dict(cfg), 42)
        assert engine.soft_capacity is True
        assert engine._soft_lam > 0.0
        assert engine.soft_stats["seed_attempts"] >= 1
        best_cost, sol = _best(archive)
        assert sol is not None and sol.is_feasible()
        assert len(sol.tw_violations()) == 0
        assert len(sol.all_served_customers()) == 10
        for e in archive.entries:
            assert e[1]["solution"].is_feasible()
        results.append(best_cost)
    assert results[0] == pytest.approx(results[1], abs=1e-9)  # 确定性
    # 小实例全局最优可达 (锁存值), 软容量不得劣化到离谱 (≤ +10%)
    assert results[0] <= 41.130084 * 1.10


def test_repair_soft_gate_allows_overload(lock_inst):
    """repair 容量门: soft 时允许插满路由 (超载), hard 时跨车场另开。"""
    # 构造: 车场0 1 辆车已装 95 (客户 a), 车场1 1 辆车空; 客户 b 需求 10 紧邻车场0
    depots = [Depot(0, 0, 0.0, 0.0, 1, 100.0), Depot(1, 1, 100.0, 0.0, 1, 100.0)]
    a = Customer(2, 2, 3.0, 0.0, 95.0, 0, due_time=1000.0, service_time=0.0)
    b = Customer(3, 3, 6.0, 0.0, 10.0, 0, due_time=1000.0, service_time=0.0)
    nodes = [(0.0, 0.0), (100.0, 0.0), (3.0, 0.0), (6.0, 0.0)]
    dist = [[math.dist(nodes[i], nodes[j]) for j in range(4)] for i in range(4)]
    inst = Instance("soft_gate", "test", 2, 2, 100.0, depots, [a, b], dist,
                    EmissionModel())
    from src.core.solution import Solution

    rng = np.random.default_rng(0)
    # hard: 车场0 满 (95+10>100) 且无车可开 → 去车场1 开新车
    hard_sol = Solution(inst, {0: [[2]], 1: []})
    out_hard = greedy_cost_tw_insertion(hard_sol, [3], rng)
    assert out_hard.routes[1] == [[3]] and out_hard.routes[0] == [[2]]  # 车场0 保持原状

    # soft: 容量门放宽 → 插入车场0 原路由 (负载 105 > 100)
    inst_soft = dc_replace(inst)
    setattr(inst_soft, "soft_capacity", True)
    soft_sol = Solution(inst_soft, {0: [[2]], 1: []})
    out_soft = greedy_cost_tw_insertion(soft_sol, [3], rng)
    assert 3 in out_soft.routes[0][0]
    assert sum(inst_soft.customers[i - 2].demand for i in out_soft.routes[0][0]) == 105.0
    assert 3 not in [c for rl in out_soft.routes[1] for c in rl]


def test_cap_excess_unit(lock_inst):
    """_cap_excess: 超载正确求和, 可行解为 0。"""
    inst = lock_inst
    from src.core.solution import Solution
    from src.alns.engine import ALNSEngine

    engine = ALNSEngine(inst, dict(LOCK_CONFIG), np.random.default_rng(0))
    # 超载解: 需求 15×5=75 塞进 3 条... 直接构造负载 115 的路由
    sol = Solution(inst, {0: [[2, 3, 4, 5, 6, 7]], 1: []})  # 6×15=90 ≤100
    assert engine._cap_excess(sol) == 0.0
    sol2 = Solution(inst, {0: [[2, 3, 4, 5, 6, 7, 8]], 1: []})  # 7×15=105
    assert engine._cap_excess(sol2) == pytest.approx(5.0)
    sol3 = Solution(inst, {0: [[2, 3, 4, 5, 6, 7, 8, 9]], 1: []})  # 8×15=120
    assert engine._cap_excess(sol3) == pytest.approx(20.0)
