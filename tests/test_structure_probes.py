"""结构生成三探针 (F/E/P, 2026-09-05) 回归测试。

F = 罚域轨迹末端硬修复口径 (engine.last_solution 暴露 + softfix 纯罚配置);
E = 教育后代池外新路由键计数 (cx_stats.novel_route_keys);
P = 互补双亲交叉 (crossover_pair=diverse, 默认 random 逐位不变)。
全部行为中性: 默认配置下与改动前逐位一致 (132 PASS 锁存基底)。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.solvers.standard_alns import solve_standard_alns

from tests.test_soft_capacity import _build_lock_inst, LOCK_CONFIG, _run_engine


def lock_inst():
    return _build_lock_inst()


def test_invalid_crossover_pair_raises():
    """非法 crossover_pair → ValueError。"""
    inst = lock_inst()
    with pytest.raises(ValueError, match="crossover_pair"):
        _run_engine(inst, dict(LOCK_CONFIG, crossover_pair="bogus"), 42)


def test_default_pair_is_random():
    """默认 crossover_pair == \"random\" (显式 random 与默认逐位一致)。"""
    inst = lock_inst()
    _, engine = _run_engine(inst, dict(LOCK_CONFIG), 42)
    assert engine.crossover_pair == "random"
    a1 = solve_standard_alns(lock_inst(), dict(LOCK_CONFIG),
                             np.random.default_rng(42))
    a2 = solve_standard_alns(lock_inst(),
                             dict(LOCK_CONFIG, crossover_pair="random"),
                             np.random.default_rng(42))
    assert a1.best_cost() == pytest.approx(a2.best_cost(), abs=1e-9)


def test_last_solution_exposed():
    """run() 后暴露轨迹末端解 (行为中性, 与档案 best 同量级)。"""
    inst = lock_inst()
    _, engine = _run_engine(inst, dict(LOCK_CONFIG), 42)
    assert engine.last_solution is not None
    assert engine.last_obj[0] > 0


def test_diverse_pair_runs_and_counts():
    """pdiv 配置: 引擎跑通 + cx_stats 键完整 (children/novel_route_keys)。"""
    inst = lock_inst()
    cfg = dict(LOCK_CONFIG, population_crossover=True, crossover_rate=0.5,
               crossover_pair="diverse", archive_diversity=True)
    archive, engine = _run_engine(inst, cfg, seed=42)
    assert engine.crossover_pair == "diverse"
    assert set(engine.cx_stats) == {"iterations", "children",
                                    "novel_route_keys"}
    assert engine.cx_stats["children"] == engine.cx_stats["iterations"]
    assert engine.cx_stats["novel_route_keys"] >= 0
    feas = [e for e in archive.entries if e[1]["solution"].is_feasible()]
    assert feas, "档案必须含可行解"


def test_softfix_pure_penalty_runs():
    """softfix (soft + 不周期播种): 引擎跑通 + 末端修复出口可用。"""
    inst = lock_inst()
    cfg = dict(LOCK_CONFIG, soft_capacity=True, soft_repair_interval=0)
    archive, engine = _run_engine(inst, cfg, seed=42)
    assert engine.soft_repair_interval == 0
    rep = engine._soft_repair_hard(engine.last_solution.copy())
    assert rep is not None
    # lock 实例可行域非空 — 修复应产出可行解 (断言按行为契约: 修复不更差于
    # 无解; 具体可行性由实例结构保证: 2 车 × 容量 100 ≥ 总需求 150)
    assert rep.is_feasible(), "lock 实例的软末端应可硬修复回可行"
    assert engine.last_obj[0] > 0
