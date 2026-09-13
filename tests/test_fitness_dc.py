"""A2 探针回归测试: 多样性贡献进 fitness 的裁剪 (2026-09-04)。

fitness = cost − κ×best_cost×dc (dc = 到档案其余解的平均路由 Jaccard 距离)。
与 div (阈值无条件保结构代表) 的区别: fitness 版对"结构新但成本高"的解
只给 κ×best×dc 的多样性税 — 税内活、税外裁。默认关 → 行为逐位不变。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alns.archive import ParetoArchive
from src.core.solution import Solution
from src.solvers.standard_alns import solve_standard_alns

from tests.test_soft_capacity import _build_lock_inst, LOCK_CONFIG, _run_engine


def lock_inst():
    return _build_lock_inst()


def make_sol(inst, depot0, depot1):
    return Solution(inst, {0: [depot0], 1: [depot1]})


def test_fitness_dc_keeps_close_cost_structural_newcomer():
    """fitness 裁剪: 成本差 ≤ 多样性税的新结构解存活 (与 div 同向)。"""
    inst = lock_inst()
    custs = [c.index for c in inst.customers]
    depot0_all = sorted(c for c in custs if c <= 6)
    depot1_all = sorted(c for c in custs if c >= 7)
    sol_x = make_sol(inst, depot0_all, depot1_all)
    sol_y = make_sol(inst, depot0_all[:2], depot0_all[2:] + depot1_all)

    arch = ParetoArchive(capacity=3, single_objective=True,
                         fitness_dc=True, fitness_kappa=0.10)
    for cost in [100.0, 102.0, 104.0]:
        arch.add((cost, 0.0), {"solution": sol_x})
    # Y: 结构不同, 成本 107 — 差 7 单位 ≤ κ×best×dc 上限 (~10×0.7)
    arch.add((107.0, 0.0), {"solution": sol_y})
    has_y = any(abs(e[0][0] - 107.0) < 1e-9 for e in arch.entries)
    assert has_y, "税内新结构必须保留"
    assert min(e[0][0] for e in arch.entries) == 100.0, "best 必须保留"


def test_fitness_dc_prunes_too_expensive_newcomer():
    """fitness 裁剪与 div 的区别: 成本差超多样性税的新结构仍被裁。"""
    inst = lock_inst()
    custs = [c.index for c in inst.customers]
    depot0_all = sorted(c for c in custs if c <= 6)
    depot1_all = sorted(c for c in custs if c >= 7)
    sol_x = make_sol(inst, depot0_all, depot1_all)
    sol_y = make_sol(inst, depot0_all[:2], depot0_all[2:] + depot1_all)

    arch = ParetoArchive(capacity=3, single_objective=True,
                         fitness_dc=True, fitness_kappa=0.05)
    for cost in [100.0, 102.0, 104.0]:
        arch.add((cost, 0.0), {"solution": sol_x})
    # Y cost 120: 差 20 单位 > κ×best×dc 上限 (5×0.7≈3.5) → 应被裁
    arch.add((120.0, 0.0), {"solution": sol_y})
    has_y = any(abs(e[0][0] - 120.0) < 1e-9 for e in arch.entries)
    assert not has_y, "税外新结构应被裁 (与 div 无条件保代表的差异)"


def test_fitness_dc_keeps_best_under_all_clones():
    """全克隆场景: 任何模式 best (cost 最小) 都不被裁。"""
    inst = lock_inst()
    custs = [c.index for c in inst.customers]
    depot0_all = sorted(c for c in custs if c <= 6)
    depot1_all = sorted(c for c in custs if c >= 7)
    sol_x = make_sol(inst, depot0_all, depot1_all)

    arch = ParetoArchive(capacity=2, single_objective=True,
                         fitness_dc=True, fitness_kappa=0.05)
    for cost in [100.0, 102.0, 104.0]:
        arch.add((cost, 0.0), {"solution": sol_x})
    costs = sorted(e[0][0] for e in arch.entries)
    assert len(arch.entries) == 2
    assert costs[0] == 100.0
    assert costs[1] == 102.0


def test_engine_a2_arm_runs_feasible():
    """引擎 a2 臂 (fitness_dc) 冒烟: 可行解 + 默认关行为由既有锁存测试保证。"""
    inst = lock_inst()
    cfg = dict(LOCK_CONFIG, archive_fitness_dc=True, archive_fitness_kappa=0.05)
    archive, engine = _run_engine(inst, cfg, seed=42)
    feas = [e for e in archive.entries if e[1]["solution"].is_feasible()]
    assert feas, "档案必须含可行解"
    assert archive.best_cost() > 0
    assert engine.archive_fitness_dc and engine.archive_fitness_kappa == 0.05
