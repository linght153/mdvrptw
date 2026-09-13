"""LS 弧差 + 增量 TW 传播等价性测试 (2026-09-03, pr20 宽窗收敛修复)。

- 弧差 (insert/remove/replace/stitch) 与 _route_cost 全重算代数恒等
  (浮点末位差 ≤1e-9 级, 阈值 1e-9 余量 ≥1e3 倍)
- 替换/尾段增量 TW 判定与 _route_tw_excess 全重算一致
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from src.operators.local_search import (
    _arc_delta_insert,
    _arc_delta_remove,
    _arc_delta_replace,
    _arc_delta_stitch,
    _route_cost,
)
from src.operators.repair import (
    _cross_tail_excess,
    _insert_cheapest_cost_tw,
    _replace_incremental_excess,
    _route_timeline,
    _route_tw_excess,
)
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

DATA = Path(__file__).resolve().parent.parent / "data" / "mdvrptw_raw"


def _load(name):
    text = open(DATA / f"{name}.txt", encoding="utf-8").read()
    return instance_from_parsed(parse_mdvrptw(text), name=name)


def _feasible_routes(inst):
    """修复器构建可行路由 (违规路由跳过)。"""
    from src.core.solution import Solution
    sol = Solution(inst)
    removed = [c.index for c in inst.customers]
    rng = np.random.default_rng(7)
    rng.shuffle(removed)
    for ci in removed:
        _insert_cheapest_cost_tw(sol, ci)
    ok = []
    for di, rl in sol.routes.items():
        for ri, route in enumerate(rl):
            if _route_tw_excess(inst, di, route, ri) <= 1e-9 and route:
                ok.append((di, ri, list(route)))
    return ok


@pytest.mark.parametrize("name", ["pr01", "pr20"])
def test_arc_deltas_match_full_cost(name):
    """insert/remove/replace 弧差 == _route_cost 全重算差 (1e-6 容差)。"""
    inst = _load(name)
    routes = _feasible_routes(inst)
    assert routes
    rng = np.random.default_rng(5)
    all_cust = [c.index for c in inst.customers]
    for di, ri, route in routes[:10]:
        for _ in range(25):
            op = rng.integers(0, 3)
            if op == 0:  # insert
                pos = int(rng.integers(0, len(route) + 1))
                ci = int(rng.choice([c for c in all_cust if c not in route]))
                full = _route_cost(inst, di, route[:pos] + [ci] + route[pos:]) \
                    - _route_cost(inst, di, route)
                arc = _arc_delta_insert(inst, di, route, pos, ci)
            elif op == 1:  # remove
                pos = int(rng.integers(0, len(route)))
                new_r = route[:pos] + route[pos + 1:]
                full = _route_cost(inst, di, new_r) - _route_cost(inst, di, route)
                arc = _arc_delta_remove(inst, di, route, pos)
            else:  # replace
                pos = int(rng.integers(0, len(route)))
                ci = int(rng.choice([c for c in all_cust if c not in route]))
                new_r = route[:pos] + [ci] + route[pos + 1:]
                full = _route_cost(inst, di, new_r) - _route_cost(inst, di, route)
                arc = _arc_delta_replace(inst, di, route, pos, ci)
            assert abs(arc - full) < 1e-6, \
                f"{name} op{op} pos{pos}: arc {arc} != full {full}"


@pytest.mark.parametrize("name", ["pr01", "pr20"])
def test_stitch_delta_matches_full_cost(name):
    """2-opt* 接合弧差 == 全重算。"""
    inst = _load(name)
    routes = _feasible_routes(inst)
    rng = np.random.default_rng(9)
    for _ in range(60):
        i_a, i_b = rng.integers(0, len(routes), size=2)
        di_a, ri_a, ra = routes[i_a]
        di_b, ri_b, rb = routes[i_b]
        if len(ra) < 2 or len(rb) < 2:
            continue
        cut_a = int(rng.integers(1, len(ra)))
        cut_b = int(rng.integers(1, len(rb)))
        ra2 = ra[:cut_a] + rb[cut_b:]
        rb2 = rb[:cut_b] + ra[cut_a:]
        full = (_route_cost(inst, di_a, ra2) - _route_cost(inst, di_a, ra)
                + _route_cost(inst, di_b, rb2) - _route_cost(inst, di_b, rb))
        arc = _arc_delta_stitch(inst, di_a, di_b, ra, cut_a, rb, cut_b)
        assert abs(arc - full) < 1e-6


@pytest.mark.parametrize("name", ["pr01", "pr20"])
def test_replace_incremental_verdict(name):
    """替换增量 vs 全重算: 接受 (≤1e-9) 逐位, 拒绝判定一致。"""
    inst = _load(name)
    routes = _feasible_routes(inst)
    rng = np.random.default_rng(13)
    all_cust = [c.index for c in inst.customers]
    checked = 0
    for di, ri, route in routes[:10]:
        arr, done, cum_exc = _route_timeline(inst, di, route, ri)
        for _ in range(20):
            pos = int(rng.integers(0, len(route)))
            ci = int(rng.choice([c for c in all_cust if c not in route]))
            new_r = route[:pos] + [ci] + route[pos + 1:]
            full = _route_tw_excess(inst, di, new_r, ri)
            inc = _replace_incremental_excess(inst, di, route, ri, pos, ci,
                                              arr, done, cum_exc)
            if inc <= 1e-9:
                assert inc == full, f"接受位置数值不一致 {inc} vs {full}"
            else:
                assert full > 1e-9, f"误弃! inc={inc} full={full}"
            checked += 1
    assert checked > 100


@pytest.mark.parametrize("name", ["pr01", "pr20"])
def test_cross_tail_incremental_verdict(name):
    """2-opt* 尾段增量 vs 全重算: 接受 (≤1e-9) 逐位, 拒绝判定一致。"""
    inst = _load(name)
    routes = _feasible_routes(inst)
    rng = np.random.default_rng(17)
    checked = 0
    for _ in range(80):
        i_a, i_b = rng.integers(0, len(routes), size=2)
        di_a, ri_a, ra = routes[i_a]
        di_b, ri_b, rb = routes[i_b]
        if len(ra) < 2 or len(rb) < 2:
            continue
        cut_a = int(rng.integers(1, len(ra)))
        cut_b = int(rng.integers(1, len(rb)))
        ra2 = ra[:cut_a] + rb[cut_b:]
        rb2 = rb[:cut_b] + ra[cut_a:]
        arr_a, done_a, cum_a = _route_timeline(inst, di_a, ra, ri_a)
        arr_b, done_b, cum_b = _route_timeline(inst, di_b, rb, ri_b)
        # A' 侧: 尾段 B[cut_b:] 接在 A[cut_a-1] 后
        inc_a = _cross_tail_excess(inst, di_b, rb, cut_b,
                                   done_a[cut_a - 1], ra[cut_a - 1],
                                   arr_b, done_b, cum_b)
        # B' 侧
        inc_b = _cross_tail_excess(inst, di_a, ra, cut_a,
                                   done_b[cut_b - 1], rb[cut_b - 1],
                                   arr_a, done_a, cum_a)
        full_a = _route_tw_excess(inst, di_a, ra2, ri_a)
        full_b = _route_tw_excess(inst, di_b, rb2, ri_b)
        for inc, full in ((inc_a, full_a), (inc_b, full_b)):
            if inc <= 1e-9:
                assert inc == full, f"接受位置数值不一致 {inc} vs {full}"
            else:
                assert full > 1e-9, f"误弃! inc={inc} full={full}"
            checked += 1
    assert checked > 100
