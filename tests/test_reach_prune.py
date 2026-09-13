"""方向 A 时间可达弧剪枝: 矩阵正确性 + 逐位等价测试 (2026-09-03)。

无损性铁律: 剪枝只跳过"硬不可达弧" (e_a+s_a+t_ab > l_b, 任何可行解
都不可能连续访问 a→b), 不改变任何通过候选的评估 → 同 seed 同迭代下
剪枝开/关必须逐位等价。不等价 = 误剪 bug。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace as dc_replace

from src.core.instance import instance_from_parsed
from src.solvers.standard_alns import solve_standard_alns
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

DATA = Path(__file__).resolve().parent.parent / "data" / "mdvrptw_raw"

CONFIG = {
    "max_iterations": 150, "max_time_seconds": 1e9, "stagnation_limit": 300,
    "segment_size": 100, "reaction_factor": 0.1, "decay_factor": 1.0,
    "min_selection_prob": 0.005, "initial_temperature": None, "cooling_rate": None,
    "archive_capacity": 20, "parent_selection": "crowding",
    "initial_solution": "nearest", "k_min_ratio": 0.10, "k_max_ratio": 0.40,
    "sigma": 3.0, "local_search_max_iter": 10, "local_search_freq": 5,
    "local_search_on_accept": True, "local_search_final": False,
}


def _load(name, **kw):
    text = open(DATA / f"{name}.txt", encoding="utf-8").read()
    inst = instance_from_parsed(parse_mdvrptw(text), name=name)
    return dc_replace(inst, **kw) if kw else inst


def test_reach_matrix_matches_definition():
    """矩阵值与硬不可达公式逐对一致 (抽查全客户对)。"""
    inst = _load("pr07")  # 72 客户窄窗, 不可达弧 ~40%
    R = inst.reach_matrix
    assert R is not None
    base = inst.num_depots
    cs = {c.index: c for c in inst.customers}
    checked = mismatches = 0
    for a in inst.customers:
        for b in inst.customers:
            if a.index == b.index:
                continue
            expect = (a.ready_time + a.service_time
                      + inst.distance_matrix[a.index][b.index] <= b.due_time)
            got = bool(R[a.index - base][b.index - base])
            checked += 1
            if expect != got:
                mismatches += 1
    assert mismatches == 0, f"{mismatches}/{checked} 对与公式不一致"
    # 窄窗实例应有相当比例不可达弧 (探针: 37-40%)
    unreach = 1.0 - R.mean()
    assert 0.25 < unreach < 0.55, f"不可达弧比例异常: {unreach:.1%}"


def test_no_tw_no_matrix():
    """无 TW 实例矩阵为 None (零开销) — 用去掉 TW 的 pr01 验证。"""
    inst = _load("pr01")
    no_tw = dc_replace(inst, customers=[dc_replace(c, ready_time=0.0,
                                                   due_time=float("inf"))
                                        for c in inst.customers])
    assert no_tw.reach_matrix is None


@pytest.mark.parametrize("name", ["pr01", "pr11"])
def test_bitwise_equivalence_prune_on_off(name):
    """剪枝开/关同 seed 同迭代 → 档案逐位等价 (无损性铁律)。"""
    inst_on = _load(name)
    inst_off = _load(name, use_reach_prune=False)
    rng_on = np.random.default_rng(42)
    rng_off = np.random.default_rng(42)
    arc_on = solve_standard_alns(inst_on, dict(CONFIG), rng_on)
    arc_off = solve_standard_alns(inst_off, dict(CONFIG), rng_off)
    best_on = min(e[0][0] for e in arc_on.entries)
    best_off = min(e[0][0] for e in arc_off.entries)
    assert best_on == best_off, \
        f"{name}: 剪枝开 {best_on} != 关 {best_off} — 误剪!"
    assert len(arc_on.entries) == len(arc_off.entries)
    # 路由结构逐位一致
    sol_on = min((e[1]["solution"] for e in arc_on.entries),
                 key=lambda s: sum(sum(r) for rl in s.routes.values() for r in rl))
    sol_off = min((e[1]["solution"] for e in arc_off.entries),
                  key=lambda s: sum(sum(r) for rl in s.routes.values() for r in rl))
    routes_on = {d: [list(r) for r in rl] for d, rl in sol_on.routes.items()}
    routes_off = {d: [list(r) for r in rl] for d, rl in sol_off.routes.items()}
    assert routes_on == routes_off
