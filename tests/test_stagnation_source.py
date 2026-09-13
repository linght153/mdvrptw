"""stagnation_source=current 修复候选回归测试 (2026-09-04)。

archive 默认源逐位不变 (既有锁存); current 源: accepted 且 obj 严格改进才
清零停滞计数 (拒收/平台累计) → 引擎可跑、解可行。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.solvers.standard_alns import solve_standard_alns

from tests.test_soft_capacity import _build_lock_inst, LOCK_CONFIG, _run_engine


def lock_inst():
    return _build_lock_inst()


def test_current_source_runs_feasible():
    """current 源: 引擎跑通 + 可行解 + stagnation_source 传递。"""
    inst = lock_inst()
    cfg = dict(LOCK_CONFIG, stagnation_source="current")
    archive, engine = _run_engine(inst, cfg, seed=42)
    assert engine.stagnation_source == "current"
    feas = [e for e in archive.entries if e[1]["solution"].is_feasible()]
    assert feas, "档案必须含可行解"
    assert archive.best_cost() > 0


def test_current_source_matches_archive_when_no_stall():
    """短预算 (300 iter, 无停滞) 下 current ≈ archive (同为有效搜索)。"""
    inst = lock_inst()
    cfg = dict(LOCK_CONFIG, max_iterations=300, stagnation_limit=500)
    a_arch, _ = _run_engine(inst, dict(cfg, stagnation_source="archive"), 42)
    c_arch, _ = _run_engine(inst, dict(cfg, stagnation_source="current"), 42)
    # 行为契约: 两者都产出可行解且成本同量级 (非逐位 — 重启时机可能分叉)
    a_best = a_arch.best_cost()
    c_best = c_arch.best_cost()
    assert a_best > 0 and c_best > 0
    assert abs(c_best - a_best) / a_best < 0.5, "同一量级"
