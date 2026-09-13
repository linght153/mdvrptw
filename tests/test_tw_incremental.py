"""增量 TW 传播 (_insert_incremental_excess) 与全重算 (_route_tw_excess)
等价性测试 (2026-09-03, 宽窗收敛修复)。

吸收点提前终止必须与整路由全重算给出完全一致的可行性判定:
- 原路由可行 (excess=0): 返回值逐位一致
- 原路由违规 (excess>0): 判定 (>1e-9 弃) 一致
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from src.operators.repair import (
    _insert_cheapest_cost_tw,
    _insert_incremental_excess,
    _route_timeline,
    _route_tw_excess,
    greedy_cost_tw_insertion,
)
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

DATA = Path(__file__).resolve().parent.parent / "data" / "mdvrptw_raw"


def _load(name):
    text = open(DATA / f"{name}.txt", encoding="utf-8").read()
    return instance_from_parsed(parse_mdvrptw(text), name=name)


def _feasible_routes(inst):
    """用修复器构建可行路由 (每车场各车一条; 违规路由跳过)。"""
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
    return ok, sol


@pytest.mark.parametrize("name", ["pr01", "pr20"])
def test_incremental_exact_on_feasible_routes(name):
    """可行路由上: 增量返回值与全重算逐位一致 (全部位置 × 抽样客户)。"""
    inst = _load(name)
    routes, _ = _feasible_routes(inst)
    assert routes, "应构建出可行路由"
    rng = np.random.default_rng(11)
    all_cust = [c.index for c in inst.customers]
    checked = 0
    for di, ri, route in routes[:12]:
        arr, done, cum_exc = _route_timeline(inst, di, route, ri)
        assert cum_exc[-1] <= 1e-9
        in_route = set(route)
        cands = [c for c in all_cust if c not in in_route]
        for pos in range(len(route) + 1):
            for ci in rng.choice(cands, size=min(4, len(cands)), replace=False):
                full = _route_tw_excess(inst, di, route[:pos] + [ci] + route[pos:], ri)
                inc = _insert_incremental_excess(
                    inst, di, route, ri, pos, int(ci), arr, done, cum_exc)
                # 接受位置 (≤1e-9) 必须逐位一致 (此时传播到吸收点/末尾,
                # 无截断); 拒绝位置增量可能截断于首个违规 (数值可小于
                # 全值) — 判弃方向必须一致 (设计契约)。
                if inc <= 1e-9:
                    assert inc == full, \
                        f"{name} d{di} r{ri} pos{pos} ci{ci}: inc {inc} != full {full}"
                else:
                    assert full > 1e-9, \
                        f"{name} d{di} r{ri} pos{pos} ci{ci}: inc 弃但 full {full} 可行!"
                checked += 1
    assert checked > 100


@pytest.mark.parametrize("name", ["pr01", "pr20"])
def test_incremental_verdict_consistent_on_violating_routes(name):
    """违规路由 (随机): 增量与全重算的 >1e-9 判定一致 (数值仅 old>0 时
    可能因吸收继承高估, 判弃方向不受影响)。"""
    inst = _load(name)
    rng = np.random.default_rng(23)
    all_cust = [c.index for c in inst.customers]
    nd = inst.num_depots
    rng.shuffle(all_cust)
    mism = checked = 0
    for trial in range(60):
        # 随机路由 (任意长度, 含违规)
        k = int(rng.integers(1, min(25, len(all_cust))))
        route = [int(c) for c in rng.choice(all_cust, size=k, replace=False)]
        arr, done, cum_exc = _route_timeline(inst, 0, route, 0)
        cands = [c for c in all_cust if c not in route]
        for _ in range(3):
            ci = int(rng.choice(cands))
            pos = int(rng.integers(0, len(route) + 1))
            full = _route_tw_excess(inst, 0, route[:pos] + [ci] + route[pos:], 0)
            inc = _insert_incremental_excess(inst, 0, route, 0, pos, ci,
                                             arr, done, cum_exc)
            checked += 1
            if (inc > 1e-9) != (full > 1e-9):
                mism += 1
    assert mism == 0, f"{name}: {mism}/{checked} 判定不一致"


def test_empty_route_insert():
    """空路由插入: 增量与全重算一致 (pos=0)。"""
    inst = _load("pr01")
    ci = inst.customers[5].index
    arr, done, cum_exc = _route_timeline(inst, 0, [], 0)
    inc = _insert_incremental_excess(inst, 0, [], 0, 0, ci, arr, done, cum_exc)
    full = _route_tw_excess(inst, 0, [ci], 0)
    assert inc == full
