"""破坏性教育 (方向②, 2026-09-06) 回归测试。

机制 = config 键 educate_mode (默认 none): cx 后代小规模破坏-重建-教育
(destroy/repair + 短 LS), 教育后可行且成本更低才替换 new_sol。默认关 →
行为逐位不变 (既有全套测试回归锁存; test_educate_default_none 缺省断言)。
验收: 教育后解可行 + 教育产生划分变化证据 (非纯 LS 退化)。
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
from src.alns.archive import ParetoArchive
from src.alns.selection import SegmentRewardSelection
from src.alns.engine import ALNSEngine
from src.operators.crossover import route_copy_crossover
from src.operators.local_search import rvnd_improve

from tests.test_soft_capacity import LOCK_CONFIG, _run_engine


def _educ_inst():
    """2 车场 × 2 车 (容量 60), 16 客户 (需求 10, 两簇各 8, 宽窗 due=300)。

    宽窗 + 多车 → 双亲可构造不同分派骨架, 教育后结构可迁移。
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
    return Instance("educ_test", "test", 2, 16, 60.0, depots, custs, dist,
                    EmissionModel(base_emission_rate=0.0,
                                  load_correction_alpha=0.0))


def _mk_engine(inst, config, seed=1):
    """装配 TW 实例引擎 (不 run), 供 _educate_child 单测。"""
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


def _make_sol(inst, routes0, routes1):
    sol = Solution(inst)
    sol.routes[0] = [list(r) for r in routes0]
    sol.routes[1] = [list(r) for r in routes1]
    for c in inst.customers:
        c.assigned_depot_index = -1
    for di, rl in sol.routes.items():
        for route in rl:
            for ci in route:
                inst.customers[ci - inst.num_depots].assigned_depot_index = di
    return sol


def _parents(inst):
    """两个分派/排序不同的可行双亲 (划分变化教育的输入)。"""
    # A 簇客户 2..9, B 簇客户 10..17
    a = _make_sol(inst, [[2, 3, 4, 5, 6, 7], [8, 9]],
                  [[10, 11, 12, 13, 14, 15], [16, 17]])
    # B: 把 A 簇尾 2 客户与 B 簇首 2 客户跨车场混编 → 划分不同
    b = _make_sol(inst, [[2, 3, 4, 5, 10, 11], [6, 7]],
                  [[12, 13, 14, 15, 16, 17], [8, 9]])
    return a, b


# ── config gating ─────────────────────────────────────────────

def test_educate_default_none():
    """config 缺省 educate_mode == \"none\"。"""
    _, engine = _run_engine(_educ_inst(), dict(LOCK_CONFIG), 42)
    assert engine.educate_mode == "none"
    # 默认 none 不扩展 cx_stats (既有结构探针断言键集完整)
    assert set(engine.cx_stats) == {"iterations", "children", "novel_route_keys"}


def test_invalid_educate_mode_raises():
    """非法 educate_mode → ValueError。"""
    inst = _educ_inst()
    with pytest.raises(ValueError, match="educate_mode"):
        _run_engine(inst, dict(LOCK_CONFIG, educate_mode="bogus"), 42)


# ── _educate_child 单测 ───────────────────────────────────────

def test_educate_child_feasible_and_no_worse():
    """教育后解必须可行, 且成本不高于教育前 (不劣化当前迭代轨迹)。"""
    inst = _educ_inst()
    a, b = _parents(inst)
    assert a.is_feasible() and b.is_feasible()
    for seed in range(4):
        inst2, engine = _mk_engine(_educ_inst(), dict(LOCK_CONFIG, educate_mode="deep"),
                                   seed=seed)
        child = route_copy_crossover(a, b, np.random.default_rng(seed),
                                     inherit_prob=0.6)
        before = calculate_cost(child)
        out = engine._educate_child(child)
        assert out.is_feasible(), "教育后解必须可行"
        assert calculate_cost(out) <= before + 1e-9
        assert isinstance(engine.cx_stats["educate_improved"], (bool, np.bool_))
        assert engine.cx_stats["educate_delta"] >= -1e-9


def test_educate_produces_partition_change_evidence():
    """教育产生'划分变化'证据: 至少一次教育实际改进且后代结构 (路由键) 改变。

    防退化: 若实现退化成纯 LS (不做破坏-重建), 破坏比例下的结构迁移证据会
    缺失 — 本测试锁存"破坏+重建"确实发生。
    """
    inst = _educ_inst()
    a, b = _parents(inst)
    changed = False
    for seed in range(8):
        inst2, engine = _mk_engine(_educ_inst(), dict(LOCK_CONFIG, educate_mode="deep"),
                                   seed=seed)
        child = route_copy_crossover(a, b, np.random.default_rng(100 + seed),
                                     inherit_prob=0.6)
        out = engine._educate_child(child)
        assert out.is_feasible()
        if engine.cx_stats["educate_improved"]:
            ck = ParetoArchive._route_keys({"solution": child})
            ok = ParetoArchive._route_keys({"solution": out})
            if ok != ck:
                changed = True
    assert changed, "教育至少一次应产出与后代不同的路由键 (破坏-重建证据)"


# ── engine 集成 ───────────────────────────────────────────────

def test_educate_engine_integration_runs():
    """educate_mode=deep + cx 跑 20 iter: 不崩、档案可行、cx_stats 键完整。"""
    inst = _educ_inst()
    cfg = dict(LOCK_CONFIG, max_iterations=20, archive_diversity=True,
               population_crossover=True, crossover_rate=0.5,
               educate_mode="deep")
    archive, engine = _run_engine(inst, cfg, seed=42)
    assert "educate_improved" in engine.cx_stats
    assert "educate_delta" in engine.cx_stats
    assert engine.cx_stats["iterations"] > 0, "cx 迭代必须实际发生"
    feas = [e for e in archive.entries if e[1]["solution"].is_feasible()]
    assert feas, "档案必须含可行解"
    for e in archive.entries:
        sol = e[1]["solution"]
        assert len(sol.tw_violations()) == 0
        assert len(sol.all_served_customers()) == 16


def test_educate_novel_keys_not_lower_than_none():
    """小规模对照: deep 的 novel_route_keys 不显著低于 none (教育不纯 LS 退化)。"""
    inst = _educ_inst()
    base = dict(LOCK_CONFIG, max_iterations=20, archive_diversity=True,
                population_crossover=True, crossover_rate=0.5)
    _, eng_none = _run_engine(inst, dict(base), seed=42)
    _, eng_deep = _run_engine(inst, dict(base, educate_mode="deep"), seed=42)
    assert eng_deep.cx_stats["novel_route_keys"] >= \
        eng_none.cx_stats["novel_route_keys"] - 2
