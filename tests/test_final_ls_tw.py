"""TW 实例 final LS 跳过测试: two_opt 纯距离重排破坏 TW 解。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from src.solvers.standard_alns import solve_standard_alns
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw


@pytest.fixture(scope="module")
def pr01_instance():
    text = (Path(__file__).resolve().parent.parent / "data/mdvrptw_raw/pr01.txt").read_text(encoding="utf-8")
    return instance_from_parsed(parse_mdvrptw(text), name="pr01")


def _cfg():
    return {
        "max_iterations": 20000, "max_time_seconds": 60, "stagnation_limit": 500,
        "segment_size": 100, "reaction_factor": 0.1, "decay_factor": 1.0,
        "min_selection_prob": 0.005, "initial_temperature": None,
        "cooling_rate": None, "archive_capacity": 80,
        "parent_selection": "crowding", "initial_solution": "nearest",
        "k_min_ratio": 0.10, "k_max_ratio": 0.40, "sigma": 3.0,
        "local_search_max_iter": 30, "local_search_freq": 10,
        "local_search_on_accept": True, "local_search_final": True,
    }


def test_tw_instance_skips_final_ls(pr01_instance):
    """TW 实例应跳过 final LS (two_opt 纯距离重排破坏 TW 解)。

    实测: final LS 开启时成本 1766.67 (120s), 关闭/跳过后 1209.57。
    """
    inst = pr01_instance
    archive = solve_standard_alns(inst, _cfg(), np.random.default_rng(42))
    best = min(archive.entries, key=lambda e: e[0][0])
    sol = best[1]["solution"]
    assert len(sol.tw_violations()) == 0, "TW 违规"
    # final LS 跳过: 成本应 < 1500 (卡死时 1991.59)
    assert best[0][0] < 1500, f"cost={best[0][0]:.2f} 仍受 final LS 影响"


def test_non_tw_instance_keeps_final_ls():
    """非 TW 实例应保留 final LS (不受影响)。"""
    from src.core.instance import Customer, Depot, EmissionModel, Instance

    depots = [Depot(0, 0, 0.0, 0.0, 2, 100.0), Depot(1, 1, 50.0, 0.0, 2, 100.0)]
    # due_time=inf → 无 TW (Instance.has_time_windows 判定)
    custs = [Customer(i, i, float(i * 10), 5.0, 2.0, 0, 0.0, float("inf"), 0.0)
             for i in range(1, 5)]
    n = 2 + 4
    dist = [[abs(i - j) * 10.0 for j in range(n)] for i in range(n)]
    inst = Instance("t", "t", 2, 4, 100.0, depots, custs, dist, EmissionModel())
    assert not inst.has_time_windows

    cfg = _cfg()
    cfg["max_time_seconds"] = 10
    cfg["max_iterations"] = 500
    archive = solve_standard_alns(inst, cfg, np.random.default_rng(42))
    assert len(archive.entries) >= 1
    # 非 TW 实例 final LS 应正常运行 (不崩溃)
    best = min(archive.entries, key=lambda e: e[0][0])
    assert best[1]["solution"].is_feasible()
