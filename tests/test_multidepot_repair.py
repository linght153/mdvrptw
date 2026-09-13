"""多车场分配修复测试: ALNS 应使用全部车场。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from src.operators.repair import greedy_cost_tw_insertion
from src.solvers.standard_alns import solve_standard_alns
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw


@pytest.fixture(scope="module")
def pr01_instance():
    text = (Path(__file__).resolve().parent.parent / "data/mdvrptw_raw/pr01.txt").read_text(encoding="utf-8")
    return instance_from_parsed(parse_mdvrptw(text), name="pr01")


def _cfg(iters=800):
    return {
        "max_iterations": iters,
        "max_time_seconds": 40,
        "stagnation_limit": 150,
        "segment_size": 50,
        "reaction_factor": 0.1,
        "decay_factor": 1.0,
        "min_selection_prob": 0.005,
        "initial_temperature": None,
        "cooling_rate": None,
        "archive_capacity": 20,
        "parent_selection": "crowding",
        "initial_solution": "nearest",
        "k_min_ratio": 0.10,
        "k_max_ratio": 0.40,
        "sigma": 3.0,
        "local_search_max_iter": 10,
        "local_search_freq": 5,
        "local_search_on_accept": True,
        "local_search_final": False,
    }


def test_greedy_tw_uses_multiple_depots(pr01_instance):
    """greedy_cost_tw 重插应使用多个车场 (不只车场 0)。"""
    inst = pr01_instance
    from src.core.solution import Solution

    empty = Solution(inst, {d: [] for d in range(inst.num_depots)})
    rebuilt = greedy_cost_tw_insertion(
        empty, [c.index for c in inst.customers], np.random.default_rng(0))
    used_depots = [d for d, routes in rebuilt.routes.items() if routes]
    assert len(used_depots) >= 2, f"只用 {used_depots} 个车场"
    assert len(rebuilt.tw_violations()) == 0


def test_alns_uses_multiple_depots(pr01_instance):
    """ALNS 最终解应使用多个车场 (多车场问题核心)。"""
    inst = pr01_instance
    archive = solve_standard_alns(inst, _cfg(), np.random.default_rng(42))
    best = min(archive.entries, key=lambda e: e[0][0])
    sol = best[1]["solution"]
    used_depots = [d for d, routes in sol.routes.items() if routes]
    assert len(used_depots) >= 2, f"ALNS 只用 {used_depots} 个车场"
    assert len(sol.tw_violations()) == 0


def test_alns_cost_closer_to_authoritative(pr01_instance):
    """多车场修复后 ALNS 成本应显著下降 (接近权威解 1083.98)。"""
    inst = pr01_instance
    archive = solve_standard_alns(inst, _cfg(1500), np.random.default_rng(42))
    best_cost = min(e[0][0] for e in archive.entries)
    # 修复前 1858.55; 多车场后应 < 1500 (40s 预算)
    assert best_cost < 1500, f"best={best_cost:.2f} 仍过高"
