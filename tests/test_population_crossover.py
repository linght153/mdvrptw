"""档案多样性裁剪 + 路线复制交叉 (探针 2026-09-04) 回归测试。

背景: SP 上界探针发现精英档案多样性坍缩 (cost-only 裁剪 → 父代同质) →
多样性感知裁剪 + 种群交叉最小版 A/B。默认关 → 行为逐位不变 (既有锁存测试)。
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alns.archive import ParetoArchive
from src.core.instance import Customer, Depot, EmissionModel, Instance
from src.core.solution import Solution
from src.operators.crossover import route_copy_crossover
from src.solvers.standard_alns import solve_standard_alns
from dataclasses import replace as dc_replace

from tests.test_soft_capacity import _build_lock_inst, LOCK_CONFIG, _run_engine


def lock_inst():
    return _build_lock_inst()


def make_sol(inst, depot0, depot1):
    return Solution(inst, {0: [depot0], 1: [depot1]})


def test_archive_diversity_keeps_structural_rep():
    """多样性裁剪: 满容时保留不同结构的代表; cost-only 裁剪只留最便宜。"""
    inst = lock_inst()
    custs = [c.index for c in inst.customers]
    depot0_all = sorted(c for c in custs if c <= 6)
    depot1_all = sorted(c for c in custs if c >= 7)
    # X 骨架: 车场0 全接前半客户; Y 骨架: 车场0 只接 2 个, 其余去车场1
    # (Y 容量不可行无妨 — 档案单测只关心路由键集, 引擎层才管纯净性)
    sol_x = make_sol(inst, depot0_all, depot1_all)            # 10 客户 2 路由
    sol_y = make_sol(inst, depot0_all[:2], depot0_all[2:] + depot1_all)
    assert sol_x.is_feasible()

    for div in (False, True):
        arch = ParetoArchive(capacity=3, single_objective=True,
                             diversity=div, diversity_threshold=0.2)
        # 先塞一堆 X 克隆 (cost 100..108), 再塞 Y (cost 120 — 最差但结构不同)
        for i, cost in enumerate([100.0, 102.0, 104.0, 106.0, 108.0]):
            arch.add((cost, 0.0), {"solution": sol_x})
        arch.add((120.0, 0.0), {"solution": sol_y})
        costs = sorted(e[0][0] for e in arch.entries)
        has_y = any(abs(e[0][0] - 120.0) < 1e-9 for e in arch.entries)
        assert len(arch.entries) == 3
        if div:
            assert has_y, "多样性裁剪必须保留结构不同的代表 (Y)"
            assert costs[0] == 100.0, "best 必须保留"
        else:
            assert not has_y, "cost-only 裁剪必须裁掉最差 (Y)"
            assert costs == [100.0, 102.0, 104.0]


def test_archive_diversity_best_always_kept():
    """多样性裁剪: 全局最优 (cost 最低) 恒保留。"""
    inst = lock_inst()
    custs = [c.index for c in inst.customers]
    d0 = sorted(c for c in custs if c <= 6)
    d1 = sorted(c for c in custs if c >= 7)
    sol_x = make_sol(inst, d0, d1)
    sol_y = make_sol(inst, d0[:1], d0[1:] + d1)
    arch = ParetoArchive(capacity=2, single_objective=True,
                         diversity=True, diversity_threshold=0.5)
    arch.add((100.0, 0.0), {"solution": sol_x})
    arch.add((200.0, 0.0), {"solution": sol_y})   # 远差但不同结构
    arch.add((101.0, 0.0), {"solution": sol_x})   # X 克隆 (距 X=0)
    costs = [e[0][0] for e in arch.entries]
    assert 100.0 in costs and len(arch.entries) == 2


def test_crossover_child_feasible_and_deterministic():
    """路线复制交叉: 后代覆盖全客户 + 可行 + 同 seed 可复现 + 继承父代路由。"""
    inst = lock_inst()
    custs = [c.index for c in inst.customers]
    d0 = sorted(c for c in custs if c <= 6)
    d1 = sorted(c for c in custs if c >= 7)
    # 双亲: 同车场分派、不同路由序列的骨架 (跨车场服务在该 TW 实例不可行;
    # 序列差异即结构差异 — 真实档案的差异还含车场分派)
    parent_a = make_sol(inst, d0, d1)                 # 顺序 2..6 / 7..11
    parent_b = make_sol(inst, list(reversed(d0)), list(reversed(d1)))
    assert parent_a.is_feasible() and parent_b.is_feasible()

    children = []
    for _ in range(2):
        child = route_copy_crossover(parent_a, parent_b,
                                     np.random.default_rng(7), inherit_prob=0.6)
        assert child.is_feasible(), "后代必须可行 (TW 零违规 + 覆盖 + 车辆数)"
        assert len(child.all_served_customers()) == 10
        assert child.total_vehicles() <= 2
        children.append(child)

    r1 = {d: [list(r) for r in rl] for d, rl in children[0].routes.items()}
    r2 = {d: [list(r) for r in rl] for d, rl in children[1].routes.items()}
    assert r1 == r2, "同 seed 交叉后代必须逐位一致"

    # 继承性: 后代至少含一条与某父代完全相同的路由 (继承率 0.6 大概率)
    parent_keys = set()
    for p in (parent_a, parent_b):
        for di, rl in p.routes.items():
            for r in rl:
                parent_keys.add((di, tuple(r)))
    child_keys = set()
    for di, rl in children[0].routes.items():
        for r in rl:
            child_keys.add((di, tuple(r)))
    assert child_keys & parent_keys, "后代应继承父代路由 (混合继承语义)"


def test_engine_cx_arm_runs_pure():
    """引擎 cx 臂 (多样性档案 + 交叉): 运行完整 + 档案纯净 + cx 发生 + 可复现。

    注: tight-15 实例贪心修复总收敛到同一最优排序 (档案 1 条成本, 交叉无
    双亲可选) → 用 2×2 车场 16 客户合成实例 (多可行分派/排序, 档案有结构)。
    """
    inst = _build_cx_inst()
    cfg = dict(LOCK_CONFIG, archive_diversity=True,
               population_crossover=True, crossover_rate=0.5)
    results = []
    cx_counts = []
    for _ in range(2):
        archive, engine = _run_engine(inst, dict(cfg), 42)
        assert engine.cx_stats["iterations"] > 0, "交叉迭代必须实际发生"
        assert engine.archive_diversity is True
        for e in archive.entries:
            sol = e[1]["solution"]
            assert sol.is_feasible()
            assert len(sol.tw_violations()) == 0
            assert len(sol.all_served_customers()) == 16
        feas = [e[0][0] for e in archive.entries if e[1]["solution"].is_feasible()]
        results.append(min(feas))
        cx_counts.append(engine.cx_stats["iterations"])
    assert results[0] == pytest.approx(results[1], abs=1e-9)
    assert cx_counts[0] == cx_counts[1]


def _build_cx_inst():
    """2 车场 × 2 车 (容量 60) + 16 客户 (需求 10, 两簇各 8, 宽窗)。

    每路由 ≤6 客户 → 多可行车场分派/排序; emission 归零 (单目标精英池语义)。
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
    return Instance("cx_test", "test", 2, 16, 60.0, depots, custs, dist,
                    EmissionModel(base_emission_rate=0.0,
                                  load_correction_alpha=0.0))
