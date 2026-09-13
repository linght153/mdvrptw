"""回归测试: 跨路由 LS 算子 (relocate/swap/2-opt*) 不得产生容量违规解。

背景: relocate/swap/two_opt_star 只检查成本下降与 TW 可行性,
从不检查容量 — 在紧容量实例 (p01 cap=80) 上实测最大负载 278/106/168
(vs 容量 80), 且 engine 在 LS 后只做 TW 修复, 导致 Pareto 档案混入
容量违规解 (pr05 实测负载 196 vs cap 180)。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import load_instance
from src.core.solution import Solution
from src.operators.local_search import (
    relocate_improve,
    swap_improve,
    two_opt_star_improve,
)


@pytest.fixture(scope="module")
def p01_tight_solution():
    """p01 (cap=80) 紧容量可行解: 贪心装满 11 条路由, 全部 <= 80。"""
    inst = load_instance("p01")
    customers = [c.index for c in inst.customers]
    cap = inst.max_vehicle_capacity
    routes = {}
    remaining = list(customers)
    di = 0
    while remaining:
        route = []
        load = 0.0
        while remaining:
            ci = remaining[0]
            dem = inst.customers[ci - inst.num_depots].demand
            if load + dem <= cap:
                route.append(ci)
                load += dem
                remaining.pop(0)
            else:
                break
        routes.setdefault(di, []).append(route)
        di = (di + 1) % inst.num_depots  # 轮转车场, 每车场路由数 ≤ 车辆上限
    sol = Solution(inst, routes)
    assert sol.is_feasible(), "构造的紧容量解必须可行"
    return inst, sol


def _max_load(sol: Solution) -> float:
    return max(
        (sol.route_load_fast(r) for rs in sol.routes.values() for r in rs),
        default=0.0,
    )


def test_relocate_never_violates_capacity(p01_tight_solution):
    inst, sol = p01_tight_solution
    improved = relocate_improve(sol, max_iterations=50)
    assert improved.is_feasible(), (
        f"relocate 产生容量违规解: max_load={_max_load(improved):.0f} "
        f"> cap={inst.max_vehicle_capacity:.0f}"
    )


def test_swap_never_violates_capacity(p01_tight_solution):
    inst, sol = p01_tight_solution
    improved = swap_improve(sol, max_iterations=50)
    assert improved.is_feasible(), (
        f"swap 产生容量违规解: max_load={_max_load(improved):.0f} "
        f"> cap={inst.max_vehicle_capacity:.0f}"
    )


def test_two_opt_star_never_violates_capacity(p01_tight_solution):
    inst, sol = p01_tight_solution
    improved = two_opt_star_improve(sol, max_iterations=50)
    assert improved.is_feasible(), (
        f"2-opt* 产生容量违规解: max_load={_max_load(improved):.0f} "
        f"> cap={inst.max_vehicle_capacity:.0f}"
    )


def test_engine_enforces_feasibility_after_ls():
    """engine 兜底: 即使 LS 回调返回容量违规解, 档案也不得混入违规条目。

    bad_ls 把每个车场的路由合并为一条 (更便宜但超容), 若无兜底会
    以更优目标值支配初始解进入档案 — 重现 pr05/p01 的违规入档路径。
    """
    inst = load_instance("p01")
    calls = {"n": 0}

    def bad_ls(sol, max_iterations=0, rng=None):
        calls["n"] += 1
        out = sol.copy()
        for di, rlist in out.routes.items():
            if len(rlist) > 1:
                merged = [c for r in rlist for c in r]
                out.routes[di] = [merged] if merged else []
        return out

    from src.alns.engine import ALNSEngine

    cfg = {
        "max_iterations": 30, "segment_size": 50, "archive_capacity": 20,
        "stagnation_limit": 100, "local_search_freq": 5,
        "local_search_max_iter": 5, "local_search_on_accept": True,
        "local_search_final": False,
    }
    engine = ALNSEngine(
        instance=inst, config=cfg, rng=np.random.default_rng(42),
        local_search=bad_ls,
    )
    archive = engine.run()
    assert calls["n"] > 0, "bad_ls 从未被调用, 测试无效"
    assert archive.size > 0, "档案不应为空"
    for obj, meta in archive.entries:
        sol = meta["solution"]
        assert sol.is_feasible(), (
            f"档案混入容量违规解: cost={obj[0]:.2f} "
            f"max_load={_max_load(sol):.0f} > cap={inst.max_vehicle_capacity:.0f}"
        )
