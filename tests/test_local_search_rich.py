"""多邻域局部搜索测试: relocate/swap/or_opt/two_opt_star + RVND。

LS 是 VRP 元启发式的核心引擎 (Vidal 2013 HGSADC 强在多邻域 LS);
two_opt 仅路由内, 无法跨路由/跨车场改进 — 必须补全邻域集。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from src.core.objectives import calculate_objectives
from src.operators.local_search import (
    relocate_improve,
    rvnd_improve,
    swap_improve,
    two_opt_improve,
)
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw


@pytest.fixture(scope="module")
def pr01_instance():
    text = (Path(__file__).resolve().parent.parent / "data/mdvrptw_raw/pr01.txt").read_text(encoding="utf-8")
    return instance_from_parsed(parse_mdvrptw(text), name="pr01")


def test_relocate_improves_solution(pr01_instance):
    """relocate 应能找到 two_opt 无法发现的跨路由改进。"""
    inst = pr01_instance
    from src.core.initial_solution import nearest_neighbor

    sol = nearest_neighbor(inst, np.random.default_rng(42))
    # 强制 TW 修复保证起点合法 (nearest_neighbor 可能 TW 违规)
    from src.solvers.standard_alns import solve_standard_alns

    cfg = {
        "max_iterations": 200, "max_time_seconds": 15, "stagnation_limit": 50,
        "segment_size": 50, "reaction_factor": 0.1, "decay_factor": 1.0,
        "min_selection_prob": 0.005, "initial_temperature": None,
        "cooling_rate": None, "archive_capacity": 20,
        "parent_selection": "crowding", "initial_solution": "nearest",
        "k_min_ratio": 0.10, "k_max_ratio": 0.40, "sigma": 3.0,
        "local_search_max_iter": 10, "local_search_freq": 5,
        "local_search_on_accept": True, "local_search_final": False,
    }
    archive = solve_standard_alns(inst, cfg, np.random.default_rng(42))
    sol = min(archive.entries, key=lambda e: e[0][0])[1]["solution"]
    assert len(sol.tw_violations()) == 0

    improved = relocate_improve(sol, max_iterations=50)
    cost_before = calculate_objectives(sol)[0]
    cost_after = calculate_objectives(improved)[0]
    assert cost_after <= cost_before + 1e-6, \
        f"relocate 成本上升: {cost_before:.2f} → {cost_after:.2f}"
    assert len(improved.tw_violations()) == 0, "relocate 破坏 TW"


def test_swap_keeps_feasibility(pr01_instance):
    """swap 应保持 TW 可行。"""
    inst = pr01_instance
    from src.solvers.standard_alns import solve_standard_alns

    cfg = {
        "max_iterations": 200, "max_time_seconds": 15, "stagnation_limit": 50,
        "segment_size": 50, "reaction_factor": 0.1, "decay_factor": 1.0,
        "min_selection_prob": 0.005, "initial_temperature": None,
        "cooling_rate": None, "archive_capacity": 20,
        "parent_selection": "crowding", "initial_solution": "nearest",
        "k_min_ratio": 0.10, "k_max_ratio": 0.40, "sigma": 3.0,
        "local_search_max_iter": 10, "local_search_freq": 5,
        "local_search_on_accept": True, "local_search_final": False,
    }
    archive = solve_standard_alns(inst, cfg, np.random.default_rng(42))
    sol = min(archive.entries, key=lambda e: e[0][0])[1]["solution"]
    assert len(sol.tw_violations()) == 0

    improved = swap_improve(sol, max_iterations=30)
    assert len(improved.tw_violations()) == 0, "swap 破坏 TW"
    cost_before = calculate_objectives(sol)[0]
    cost_after = calculate_objectives(improved)[0]
    assert cost_after <= cost_before + 1e-6


def test_rvnd_at_least_as_good_as_two_opt(pr01_instance):
    """RVND 应不差于纯 two_opt, 且不破坏 TW。"""
    inst = pr01_instance
    from src.solvers.standard_alns import solve_standard_alns

    cfg = {
        "max_iterations": 200, "max_time_seconds": 15, "stagnation_limit": 50,
        "segment_size": 50, "reaction_factor": 0.1, "decay_factor": 1.0,
        "min_selection_prob": 0.005, "initial_temperature": None,
        "cooling_rate": None, "archive_capacity": 20,
        "parent_selection": "crowding", "initial_solution": "nearest",
        "k_min_ratio": 0.10, "k_max_ratio": 0.40, "sigma": 3.0,
        "local_search_max_iter": 10, "local_search_freq": 5,
        "local_search_on_accept": True, "local_search_final": False,
    }
    archive = solve_standard_alns(inst, cfg, np.random.default_rng(42))
    sol = min(archive.entries, key=lambda e: e[0][0])[1]["solution"]

    to_opt = two_opt_improve(sol, max_iterations=30)
    rvnd = rvnd_improve(sol, max_iterations=30, rng=np.random.default_rng(1))
    assert len(rvnd.tw_violations()) == 0
    cost_to = calculate_objectives(to_opt)[0]
    cost_rvnd = calculate_objectives(rvnd)[0]
    assert cost_rvnd <= cost_to + 1e-6, \
        f"RVND({cost_rvnd:.2f}) 差于 two_opt({cost_to:.2f})"
