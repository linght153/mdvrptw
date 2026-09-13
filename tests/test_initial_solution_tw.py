"""TW 初始解可行性回归测试 (2026-09-03 基线首跑发现)。

背景: pr14/17/18/20 (宽窗紧车: 每场 1-4 辆) 120s 基线 0 可行解 —
nearest_neighbor 初始解车辆数超限 (阶段 2 无条件 append) + TW 传播违规
(纯距离构建), engine enforce 链修不动 → 搜索全程无可行解入档。
修复: 阶段 2 复用跨车场插入 (不超车数) + TW 实例阶段 1/2 TW-aware。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from src.core.initial_solution import build_initial_solution
from src.solvers.standard_alns import solve_standard_alns
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

DATA = Path(__file__).resolve().parent.parent / "data" / "mdvrptw_raw"


def _load(name):
    text = open(DATA / f"{name}.txt", encoding="utf-8").read()
    return instance_from_parsed(parse_mdvrptw(text), name=name)


@pytest.mark.parametrize("name", ["pr14", "pr17", "pr18", "pr20"])
def test_nearest_initial_vehicle_compliant(name):
    """紧车宽窗实例: nearest 初始解车辆数合规 + 全覆盖 (曾超限: pr17 depot1 3条/上限1)。"""
    inst = _load(name)
    sol = build_initial_solution(inst, np.random.default_rng(42), "nearest")
    for di, rl in sol.routes.items():
        assert len(rl) <= inst.depots[di].vehicles_available, \
            f"{name} 车场{di}: {len(rl)} 条路由 > 上限 {inst.depots[di].vehicles_available}"
    assert len(sol.all_served_customers()) == inst.num_customers, f"{name} 覆盖不全"


@pytest.mark.parametrize("name", ["pr14", "pr17", "pr18"])
def test_alns_finds_feasible_on_tight_wide(name):
    """紧车宽窗实例短跑必须出可行解 (曾 120s 档案 0 可行)。"""
    inst = _load(name)
    cfg = {"max_iterations": 10_000_000, "max_time_seconds": 45.0,
           "stagnation_limit": 300, "segment_size": 100,
           "reaction_factor": 0.1, "decay_factor": 1.0,
           "min_selection_prob": 0.005, "initial_temperature": None,
           "cooling_rate": None, "archive_capacity": 20,
           "parent_selection": "crowding", "initial_solution": "nearest",
           "k_min_ratio": 0.10, "k_max_ratio": 0.40, "sigma": 3.0,
           "local_search_max_iter": 10, "local_search_freq": 5,
           "local_search_on_accept": True, "local_search_final": False}
    arc = solve_standard_alns(inst, cfg, np.random.default_rng(42))
    feas = [e for e in arc.entries if e[1]["solution"].is_feasible()]
    assert feas, f"{name} 45s 内无可行解入档"
    best = min(e[0][0] for e in feas)
    assert best < float("inf")
