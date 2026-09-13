"""score_scheme 解耦探针 (2026-09-05) 回归测试。

背景: 6000 iter 口径 base vs admit 差 3.7pp 的机制链 = archive.add 成败 →
score (2 if added) → SegmentReward 段权重 → destroy 分布 (7358b70)。
修复候选 = score 与准入解耦。config 键 score_scheme (默认 "admission" 逐位
不变): "activity" = 可行且未崩溃即反馈 2.0 (复刻 admit 的 score 分布但不
放宽准入); "quality" = P&R 改进阶梯 (new<best 2.0 / new<parent 1.5 /
accepted 1.0 / 拒 0.0)。
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


def test_invalid_scheme_raises():
    """非法 score_scheme → ValueError (配置显式报错, 不静默)。"""
    inst = lock_inst()
    with pytest.raises(ValueError, match="score_scheme"):
        _run_engine(inst, dict(LOCK_CONFIG, score_scheme="bogus"), 42)


def test_default_scheme_is_admission():
    """默认无键 → score_scheme == \"admission\" (行为逐位不变由既有锁存)。"""
    inst = lock_inst()
    _, engine = _run_engine(inst, dict(LOCK_CONFIG), 42)
    assert engine.score_scheme == "admission"
    # 显式 admission 与默认无键逐位一致 (同 seed 确定性引擎)
    a1 = solve_standard_alns(lock_inst(), dict(LOCK_CONFIG),
                             np.random.default_rng(42))
    a2 = solve_standard_alns(lock_inst(), dict(LOCK_CONFIG, score_scheme="admission"),
                             np.random.default_rng(42))
    assert a1.best_cost() == pytest.approx(a2.best_cost(), abs=1e-9)


def test_activity_runs_feasible():
    """activity: 引擎跑通 + 属性传递 + 档案含可行解。"""
    inst = lock_inst()
    cfg = dict(LOCK_CONFIG, score_scheme="activity")
    archive, engine = _run_engine(inst, cfg, seed=42)
    assert engine.score_scheme == "activity"
    feas = [e for e in archive.entries if e[1]["solution"].is_feasible()]
    assert feas, "档案必须含可行解"
    assert archive.best_cost() > 0


def test_quality_runs_feasible():
    """quality: 引擎跑通 + 属性传递 + 档案含可行解。"""
    inst = lock_inst()
    cfg = dict(LOCK_CONFIG, score_scheme="quality")
    archive, engine = _run_engine(inst, cfg, seed=42)
    assert engine.score_scheme == "quality"
    feas = [e for e in archive.entries if e[1]["solution"].is_feasible()]
    assert feas, "档案必须含可行解"
    assert archive.best_cost() > 0


def test_probe_stats_recorded():
    """probe_stats 观察计数存在且 300 iter 短预算 (limit 1e5) 无重启。"""
    inst = lock_inst()
    _, engine = _run_engine(inst, dict(LOCK_CONFIG), 42)
    ps = engine.probe_stats
    assert set(ps) == {"adds", "accepts", "equal_accepts", "restarts",
                       "stagnation_peak"}
    assert ps["restarts"] == 0
    assert ps["adds"] + ps["accepts"] <= 300
    assert ps["stagnation_peak"] >= 0


def test_schemes_same_scale_short_budget():
    """短预算 (300 iter, 无平台期) 三臂均产出同量级可行解 (非逐位契约)。"""
    inst = lock_inst()
    bests = []
    for scheme in ("admission", "activity", "quality"):
        a, _ = _run_engine(inst, dict(LOCK_CONFIG, score_scheme=scheme), 42)
        bests.append(a.best_cost())
    assert all(b > 0 for b in bests)
    ref = bests[0]
    for b in bests[1:]:
        assert abs(b - ref) / ref < 0.5, "同一量级"
