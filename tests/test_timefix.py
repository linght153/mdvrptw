"""timefix (2026-09-12) 回归测试 — 时间预算修复 ×2 (durfix 后续 P0 复盘产物)。

Bug 1: 引擎时间预算锚点在"主循环开始"而非 run 入口 → 初始化耗时逃逸预算。
  durfix 后 pcxge 种群初始化在 pr15/16/20 达 20~103s, 全部不计入 600s →
  时间预算运行实测超时 +22~+97s (base/pyvrp 严格 600s, 对照失真)。
Bug 2: 时间预算 + 池全不可行 + 档案空 → 成员刷新风暴 (每次重建 ~13s 全败)
  烧光预算且零可行解 (pr20 s3 600s 实测确定性复现; 6000iter 模式同 seed 健康)。
  修复 = 触发式重预算爬坡引导 (30% 预算后仍未破局 → 对最优成员爬坡,
  成功替换最劣成员并播种档案; 仅时间预算模式, 迭代模式不触发)。

两条修复均只作用于时间预算模式 (max_time < 1e8): 6000iter 模式 (1e9)
行为逐位不变 (已批 B1-B5/B6 校准结果不受影响)。
合成 fixture 初始池恒可行 → Bug 2 的"触发-恢复"端到端以 pr20 s3 实测
(scripts 留档) 与批重跑为准, 此处锁定判定逻辑与引导机构 (白盒)。
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alns import engine as engine_module
from src.alns.engine import ALNSEngine
from src.alns.archive import ParetoArchive
from src.core.objectives import calculate_objectives

from tests.test_soft_capacity import LOCK_CONFIG
from tests.test_population import (
    _tw_inst, _make_engine, _pop_cfg, _all_at_depot,
)


# ── Bug 1: 预算锚点覆盖初始化 ──────────────────────────────

def test_budget_anchor_covers_init_normal_path(monkeypatch):
    """慢初始化 (1.5s) + 预算 1.0s → 总墙钟 ≈ 初始化, 不得再叠加预算。

    旧行为 (锚点在循环开始): 墙钟 ≈ 1.5 + 1.0 + 尾 ≈ 2.6s。
    """
    inst = _tw_inst()
    cfg = dict(LOCK_CONFIG, initial_solution="random",
               max_iterations=10 ** 9, max_time_seconds=1.0)
    real_build = engine_module.build_initial_solution

    def slow_build(*args, **kwargs):
        time.sleep(1.5)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(engine_module, "build_initial_solution", slow_build)
    engine = _make_engine(inst, cfg, seed=42)
    t0 = time.perf_counter()
    engine.run()
    wall = time.perf_counter() - t0
    assert wall <= 2.0, f"预算未覆盖初始化: wall={wall:.2f}s (应 ≈1.5s)"


def test_budget_anchor_covers_init_population_path(monkeypatch):
    """种群路径同理: 4 成员 × 0.45s 慢构建 = 1.8s 初始化 > 1.0s 预算 →
    主循环不得再获得满额预算。旧行为墙钟 ≈ 1.8 + 1.0 + 尾 ≈ 2.9s。"""
    inst = _tw_inst()
    cfg = _pop_cfg(max_iterations=10 ** 9, max_time_seconds=1.0)
    real_member = ALNSEngine._build_pool_member

    def slow_member(self):
        time.sleep(0.45)
        return real_member(self)

    monkeypatch.setattr(ALNSEngine, "_build_pool_member", slow_member)
    engine = _make_engine(inst, cfg, seed=42)
    t0 = time.perf_counter()
    engine.run()
    wall = time.perf_counter() - t0
    assert wall <= 2.3, f"种群初始化未计入预算: wall={wall:.2f}s (应 ≈1.8s)"


# ── Bug 2: 引导触发判定 ────────────────────────────────────

def test_timefix_gate_iter_mode_always_off():
    """迭代模式 (max_time=1e9) 即使档案长期为空也永不触发 → 已批轨迹不变。"""
    engine = _make_engine(_tw_inst(), _pop_cfg(), seed=42)
    assert engine.max_time >= 1e8
    st = time.perf_counter() - 10 ** 6  # 模拟已耗巨大时间
    assert engine._timefix_should_fire(10 ** 6, st) is False


def test_timefix_predicate_conditions():
    """判定链: 预算未过 30% → False; 过线且档案空 → True; 档案非空 → False;
    尝试上限/最小间隔生效。"""
    engine = _make_engine(
        _tw_inst(), _pop_cfg(max_time_seconds=100.0), seed=42)
    st = time.perf_counter()
    assert engine._timefix_should_fire(1, st) is False          # 未过 30%
    late = st - 40.0                                            # 已过 40%
    assert engine._timefix_should_fire(1, late) is True

    sol = _all_at_depot(engine.instance, 0)
    engine.archive.add(calculate_objectives(sol),
                       {"iteration": 0, "solution": sol})
    assert engine._timefix_should_fire(2, late) is False        # 档案非空

    # 上限与间隔 (档案状态不参与 → 新引擎)
    engine2 = _make_engine(
        _tw_inst(), _pop_cfg(max_time_seconds=100.0), seed=42)
    engine2._timefix_attempts = 5
    assert engine2._timefix_should_fire(10 ** 6, late) is False  # 尝试上限
    engine2._timefix_attempts = 0
    engine2._timefix_last_iter = 500
    assert engine2._timefix_should_fire(550, late) is False      # 间隔不足
    assert engine2._timefix_should_fire(600, late) is True       # 间隔满


def test_timefix_bootstrap_replaces_worst_and_seeds_archive():
    """引导机构: 爬坡成功 → 返回被替换下标, 该位成员变为可行, 档案被播种。"""
    engine = _make_engine(_tw_inst(), _pop_cfg(), seed=42)
    inst = engine.instance
    pool_sol = [_all_at_depot(inst, 0) for _ in range(4)]
    pool_obj = [calculate_objectives(s) for s in pool_sol]
    pool_keys = [ParetoArchive._route_keys({"solution": s}) for s in pool_sol]

    stub = _all_at_depot(inst, 1)  # 任意真实可行解 (爬坡桩)
    engine._timefix_climb = lambda sol: stub.copy()  # type: ignore[method-assign]

    ret = engine._timefix_feasibility_bootstrap(pool_sol, pool_obj, pool_keys)
    assert ret is not None and 0 <= ret < 4
    assert pool_sol[ret].is_feasible()
    assert float(pool_obj[ret][0]) == float(calculate_objectives(stub)[0])
    assert engine.archive.size == 1


def test_timefix_bootstrap_climb_fail_noop():
    """爬坡失败 (返回 None) → 池与档案均不动, 返回 None。"""
    engine = _make_engine(_tw_inst(), _pop_cfg(), seed=42)
    inst = engine.instance
    pool_sol = [_all_at_depot(inst, 0) for _ in range(4)]
    pool_obj = [calculate_objectives(s) for s in pool_sol]
    pool_keys = [ParetoArchive._route_keys({"solution": s}) for s in pool_sol]

    engine._timefix_climb = lambda sol: None  # type: ignore[method-assign]
    ret = engine._timefix_feasibility_bootstrap(pool_sol, pool_obj, pool_keys)
    assert ret is None
    assert engine.archive.size == 0


def test_timefix_loop_integration_iter_mode_spy(monkeypatch):
    """整跑 (迭代模式): 引导永不调用, 统计键存在且为 0 (键集审计留痕)。"""
    calls = []

    def spy(self, *args, **kwargs):
        calls.append(1)
        return None

    monkeypatch.setattr(ALNSEngine, "_timefix_feasibility_bootstrap", spy)
    engine = _make_engine(_tw_inst(), _pop_cfg(max_iterations=30), seed=42)
    engine.run()
    assert calls == []
    s = engine.population_stats
    assert s["timefix_bootstrap_attempts"] == 0
    assert s["timefix_bootstrap_success"] == 0
