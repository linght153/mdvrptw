"""车场分配重平衡 (方向①, 2026-09-06) 回归测试。

机制 = config 键 rebalance_mode (默认 none): mismatched_removal 错配优先移除
算子 + 周期重分派 (engine 侧 redispatch pass)。默认关 → 行为逐位不变
(既有全套测试回归锁存; test_rebalance_default_none 另加缺省断言)。
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace as dc_replace

from src.core.instance import Customer, Depot, EmissionModel, Instance
from src.core.solution import Solution
from src.core.objectives import calculate_cost
from src.alns.selection import SegmentRewardSelection
from src.alns.engine import ALNSEngine
from src.operators.local_search import rvnd_improve
from src.operators.destroy import mismatched_removal

from tests.test_soft_capacity import _build_lock_inst, LOCK_CONFIG, _run_engine


def _rebal_inst():
    """2 车场 × 2 车 (容量 60), 16 客户 (需求 10, 两簇各 8, 宽窗 due=300)。

    每路由 ≤6 客户 → 存在多种可行车场分派; 任意簇内分配 TW 可行 (宽窗),
    用于构造"错配但可行"的解做红测。
    """
    depots = [Depot(0, 0, 0.0, 0.0, 2, 60.0), Depot(1, 1, 100.0, 0.0, 2, 60.0)]
    custs = []
    for i in range(8):
        custs.append(Customer(2 + i, 2 + i, 6.0 + i, 3.0, 10.0, 0,
                              ready_time=0.0, due_time=300.0, service_time=2.0))
    for i in range(8):
        custs.append(Customer(10 + i, 10 + i, 94.0 + i, 3.0, 10.0, 0,
                              ready_time=0.0, due_time=300.0, service_time=2.0))
    nodes = [(0.0, 0.0), (100.0, 0.0)] + [(c.x, c.y) for c in custs]
    n = len(nodes)
    dist = [[math.dist(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]
    return Instance("rebal_test", "test", 2, 16, 60.0, depots, custs, dist,
                    EmissionModel(base_emission_rate=0.0,
                                  load_correction_alpha=0.0))


def _mk_engine(inst, config, seed=1):
    """装配 TW 实例引擎 (不 run), 供 redispatch/机制单测。"""
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
    return inst, engine


def _mismatch_penalty(sol):
    """错配罚金 ≈ Σ 2 × (d(实际车场, 客户) − d(最近车场, 客户)) 近似口径。"""
    inst = sol.instance
    dm = inst.distance_matrix
    nd = inst.num_depots
    total = 0.0
    for di, routes in sol.routes.items():
        for route in routes:
            for ci in route:
                nearest = min(range(nd), key=lambda d: dm[d][ci])
                total += 2.0 * max(0.0, dm[di][ci] - dm[nearest][ci])
    return float(total)


def _craft_mismatched_sol(inst):
    """错配但可行解: 一个 depot1 簇客户 (全局 17, 近 x≈101) 塞进 depot0 路由。

    depot0: [2,3,4,5,17] (load 50 ≤60) + [6,7,8,9] (load 40)
    depot1: [10,11,12,13,14] (load 50) + [15,16] (load 20)
    16 客户全覆盖, 2×2 车不超限, 宽窗 TW 可行 → is_feasible()=True。
    """
    sol = Solution(inst)
    sol.routes[0] = [[2, 3, 4, 5, 17], [6, 7, 8, 9]]
    sol.routes[1] = [[10, 11, 12, 13, 14], [15, 16]]
    for c in inst.customers:
        c.assigned_depot_index = -1
    for di, rl in sol.routes.items():
        for route in rl:
            for ci in route:
                inst.customers[ci - inst.num_depots].assigned_depot_index = di
    return sol


def _correct_sol(inst):
    """正确分派解: A 簇客户全 depot0, B 簇客户全 depot1。"""
    sol = Solution(inst)
    sol.routes[0] = [[2, 3, 4, 5, 6, 7], [8, 9]]
    sol.routes[1] = [[10, 11, 12, 13, 14, 15], [16, 17]]
    return sol


# ── config gating ─────────────────────────────────────────────

def test_rebalance_default_none():
    """config 缺省 == none, interval=0, ratio=0.20 (默认关)。"""
    _, engine = _run_engine(_build_lock_inst(), dict(LOCK_CONFIG), 42)
    assert engine.rebalance_mode == "none"
    assert engine.rebalance_interval == 0
    assert engine.rebalance_ratio == pytest.approx(0.20)


def test_invalid_rebalance_mode_raises():
    """非法 rebalance_mode → ValueError。"""
    inst = _build_lock_inst()
    with pytest.raises(ValueError, match="rebalance_mode"):
        _run_engine(inst, dict(LOCK_CONFIG, rebalance_mode="bogus"), 42)


# ── mismatched_removal 算子单测 ───────────────────────────────

def test_mismatched_removal_prefers_mismatched():
    """k=1: 移除序列优先含唯一错配客户 17 (实际 depot0, 最近 depot1)。"""
    inst = _rebal_inst()
    sol = _craft_mismatched_sol(inst)
    assert sol.is_feasible(), "crafted mismatched 解必须可行"
    # 客户 17 的最近车场是 depot1
    dm = inst.distance_matrix
    assert min(range(2), key=lambda d: dm[d][17]) == 1

    partial, removed = mismatched_removal(sol, 1, np.random.default_rng(0))
    assert removed == [17], f"k=1 应移除唯一错配客户, got {removed}"
    assert 17 not in partial.all_served_customers()


def test_mismatched_removal_tops_up_to_k():
    """k > 错配数: 错配客户必在移除集, 总量补足到 k。"""
    inst = _rebal_inst()
    sol = _craft_mismatched_sol(inst)
    partial, removed = mismatched_removal(sol, 5, np.random.default_rng(0))
    assert 17 in removed
    assert len(removed) == 5
    for ci in removed:
        assert ci not in partial.all_served_customers()


def test_mismatched_removal_no_mismatch_fallback():
    """无错配: 退化为 random_removal 补齐, 不崩且移除 k 个。"""
    inst = _rebal_inst()
    sol = _correct_sol(inst)
    partial, removed = mismatched_removal(sol, 3, np.random.default_rng(0))
    assert len(removed) == 3
    for ci in removed:
        assert ci not in partial.all_served_customers()


def test_mismatched_removal_empty_solution():
    """空解: 不崩, removed 为空。"""
    inst = _rebal_inst()
    sol = Solution(inst)
    partial, removed = mismatched_removal(sol, 3, np.random.default_rng(0))
    assert removed == []
    assert len(partial.all_served_customers()) == 0


# ── redispatch pass 单测 ──────────────────────────────────────

def test_redispatch_pass_improves_mismatched():
    """注入错配解 → pass 后错配罚金下降 + 成本不增 + 仍可行。"""
    inst, engine = _mk_engine(_rebal_inst(), dict(LOCK_CONFIG))
    sol = _craft_mismatched_sol(inst)
    assert _mismatch_penalty(sol) > 0.0
    before_cost = calculate_cost(sol)

    repaired = engine._redispatch_pass(sol)
    assert repaired.is_feasible(), "redispatch 后解必须可行"
    assert calculate_cost(repaired) <= before_cost + 1e-9
    assert _mismatch_penalty(repaired) <= _mismatch_penalty(sol) + 1e-9
    # 宽窗实例: 该错配明显可修复 → 断言实际发生移动 (严格改善)
    assert _mismatch_penalty(repaired) < _mismatch_penalty(sol) - 1e-6
    # 错配客户 17 应回到最近车场 depot1
    loc = next(di for di, rl in repaired.routes.items()
               for r in rl if 17 in r)
    assert loc == 1


def test_redispatch_pass_clean_solution_unchanged():
    """正确分派解: pass 不劣化, 无错配可移 → 仍可行且成本不增。"""
    inst, engine = _mk_engine(_rebal_inst(), dict(LOCK_CONFIG))
    sol = _correct_sol(inst)
    assert sol.is_feasible()
    before_cost = calculate_cost(sol)
    repaired = engine._redispatch_pass(sol)
    assert repaired.is_feasible()
    assert calculate_cost(repaired) <= before_cost + 1e-9


# ── engine 集成 ───────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["removal", "redispatch", "both"])
def test_engine_rebalance_integration(mode):
    """rebalance_interval=1 + mode 跑 20 iter 不崩、档案含可行解。"""
    inst = _rebal_inst()
    cfg = dict(LOCK_CONFIG, max_iterations=20,
               rebalance_mode=mode, rebalance_interval=1, rebalance_ratio=0.20)
    archive, engine = _run_engine(inst, cfg, seed=42)
    assert engine.rebalance_stats["removal_passes"] > 0 or \
        engine.rebalance_stats["redispatch_passes"] > 0
    feas = [e for e in archive.entries if e[1]["solution"].is_feasible()]
    assert feas, "档案必须含可行解"
    for e in archive.entries:
        sol = e[1]["solution"]
        assert len(sol.tw_violations()) == 0
        assert len(sol.all_served_customers()) == 16


def test_rebalance_runner_budget_and_iteration_names_do_not_collide():
    """命名纪律回归 (2026-09-10): 时间预算批与固定迭代批输出不得同名。

    背景: benchmark_rebalance_ab.py 的 600s 扩展批曾与 6000iter 批写同一文件,
    使 pr20×s42-44×both 的 6000iter 记录被覆写, 表 3 双开列溯源断链。
    """
    import importlib.util
    runner = (Path(__file__).resolve().parent.parent / "experiments"
              / "benchmark_rebalance_ab.py")
    spec = importlib.util.spec_from_file_location("_bench_rebalance_ab", runner)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.out_tag(0.0, 6000) == "", "固定迭代批不得加预算标记"
    assert mod.out_tag(600.0, 6000) == "_600s", "预算批必须带 _{budget}s 标记"
    name = lambda tb: f"pr20_s42_both{mod.out_tag(tb, 6000)}.json"
    assert name(0.0) != name(600.0)
