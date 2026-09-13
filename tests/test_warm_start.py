"""warm-start 热启动集成测试。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from src.core.objectives import calculate_cost
from src.core.initial_solution import nearest_neighbor
from src.solvers.standard_alns import solve_standard_alns
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw


@pytest.fixture(scope="module")
def pr01_instance():
    text = (Path(__file__).resolve().parent.parent / "data/mdvrptw_raw/pr01.txt").read_text(encoding="utf-8")
    return instance_from_parsed(parse_mdvrptw(text), name="pr01")


def _config(iters=300):
    return {
        "max_iterations": iters,
        "max_time_seconds": 60,
        "stagnation_limit": 100,
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


def test_warm_start_accepted_as_parameter(pr01_instance):
    """solve_standard_alns 接受 initial_solution 参数。"""
    inst = pr01_instance
    init = nearest_neighbor(inst, np.random.default_rng(0))
    archive = solve_standard_alns(inst, _config(50), np.random.default_rng(1), initial_solution=init)
    assert len(archive.entries) >= 1


def test_warm_start_no_crash_with_partial_solution(pr01_instance):
    """warm-start 传部分解 (仅覆盖部分客户) 不崩溃。"""
    inst = pr01_instance
    routes = {0: [[inst.customers[0].index, inst.customers[1].index]]}
    from src.core.solution import Solution

    partial = Solution(inst, routes)
    archive = solve_standard_alns(inst, _config(30), np.random.default_rng(2), initial_solution=partial)
    assert len(archive.entries) >= 1


def test_cold_start_still_works(pr01_instance):
    """不传 initial_solution (冷启动) 行为不变 (向后兼容)。"""
    inst = pr01_instance
    archive = solve_standard_alns(inst, _config(50), np.random.default_rng(3))
    assert len(archive.entries) >= 1
