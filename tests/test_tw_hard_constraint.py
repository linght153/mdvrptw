"""TW 硬约束修复测试: is_feasible 含 TW + ALNS 默认 TW 可行。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from src.core.solution import Solution
from src.solvers.standard_alns import solve_standard_alns
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw


@pytest.fixture(scope="module")
def pr01_instance():
    text = (Path(__file__).resolve().parent.parent / "data/mdvrptw_raw/pr01.txt").read_text(encoding="utf-8")
    return instance_from_parsed(parse_mdvrptw(text), name="pr01")


def _config(iters=600):
    return {
        "max_iterations": iters,
        "max_time_seconds": 30,
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


def test_is_feasible_rejects_tw_violation():
    """is_feasible 现在必须检查 TW (硬约束)。"""
    from src.core.instance import Customer, Depot, EmissionModel, Instance

    depots = [Depot(0, 0, 0.0, 0.0, 2, 100.0)]
    custs = [Customer(1, 1, 10.0, 0.0, 1.0, 0, 0.0, 5.0, 0.0)]  # due=5, 到达 10 违规
    dist = [[0.0, 10.0], [10.0, 0.0]]
    inst = Instance("t", "t", 1, 1, 100.0, depots, custs, dist, EmissionModel())
    sol = Solution(inst, {0: [[1]]})
    assert len(sol.tw_violations()) == 1
    assert not sol.is_feasible()  # TW 违规 → 不可行


def test_alns_produces_tw_feasible_solution(pr01_instance):
    """修复后 ALNS 默认解必须 TW 可行 (零违规)。"""
    inst = pr01_instance
    archive = solve_standard_alns(inst, _config(), np.random.default_rng(42))
    best = min(archive.entries, key=lambda e: e[0][0])
    sol = best[1]["solution"]
    assert len(sol.tw_violations()) == 0, f"TW 违规 {len(sol.tw_violations())}"
    assert sol.is_feasible()


def test_alns_solution_approaches_authoritative(pr01_instance):
    """修复后 ALNS 成本应接近权威解 1083.98 (不再 861.32)。"""
    inst = pr01_instance
    archive = solve_standard_alns(inst, _config(1500), np.random.default_rng(42))
    best_cost = min(e[0][0] for e in archive.entries)
    # 权威解 1083.98; ALNS 预算有限, 允许 15% 内
    assert best_cost >= 1083.98 * 0.85, f"best={best_cost:.2f} 低于权威解 15%"
    assert best_cost > 861.32 * 1.05, "仍是无 TW 的 MDVRP 值 (修复失败)"
