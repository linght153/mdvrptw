"""MDVRPTW 引擎基线 runner: 单实例 × 单 seed, 独立进程。

用法: python experiments/benchmark_pr_baseline.py pr01 42 [budget_s]
输出: results/baseline_pr_120s/{inst}_s{seed}.json
口径: 120s 时间预算 (时间预算方差红线: 串行 + 独立进程跑), 配置与 smoke 一致。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from src.solvers.standard_alns import solve_standard_alns
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

OUT = Path(__file__).resolve().parent.parent / "results" / "baseline_pr_120s"
OUT.mkdir(parents=True, exist_ok=True)
SUFFIX = {"base": "_base", "prune": ""}

CONFIG = {
    "max_iterations": 10_000_000, "max_time_seconds": 120.0, "stagnation_limit": 300,
    "segment_size": 100, "reaction_factor": 0.1, "decay_factor": 1.0,
    "min_selection_prob": 0.005, "initial_temperature": None, "cooling_rate": None,
    "archive_capacity": 20, "parent_selection": "crowding",
    "initial_solution": "nearest", "k_min_ratio": 0.10, "k_max_ratio": 0.40,
    "sigma": 3.0, "local_search_max_iter": 10, "local_search_freq": 5,
    "local_search_on_accept": True, "local_search_final": False,
}


def main():
    name = sys.argv[1]
    seed = int(sys.argv[2])
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
    mode = sys.argv[4] if len(sys.argv) > 4 else "prune"  # base|prune
    CONFIG["max_time_seconds"] = budget

    from dataclasses import replace as dc_replace
    text = open(Path(__file__).resolve().parent.parent / f"data/mdvrptw_raw/{name}.txt",
                encoding="utf-8").read()
    inst = instance_from_parsed(parse_mdvrptw(text), name=name)
    if mode == "base":
        inst = dc_replace(inst, use_reach_prune=False)

    t0 = time.time()
    archive = solve_standard_alns(inst, CONFIG, np.random.default_rng(seed))
    wall = time.time() - t0

    feas = [(e[0][0], e[1]["solution"]) for e in archive.entries
            if e[1]["solution"].is_feasible()]
    best_cost = min((c for c, _ in feas), default=None)
    best_sol = None
    if best_cost is not None:
        best_sol = min((s for c, s in feas if abs(c - best_cost) < 1e-9),
                       key=lambda s: len(s.tw_violations()))
    out = {
        "instance": name, "seed": seed, "budget_s": budget,
        "wall_s": round(wall, 1),
        "best_feasible_cost": round(best_cost, 4) if best_cost else None,
        "vehicles": best_sol.total_vehicles() if best_sol else None,
        "tw_violations": len(best_sol.tw_violations()) if best_sol else None,
        "served": len(best_sol.all_served_customers()) if best_sol else None,
        "num_customers": inst.num_customers,
        "archive_entries": len(archive.entries),
        "iterations": max(e[1].get("iteration", 0) for e in archive.entries) if archive.entries else None,
        "mode": mode,
    }
    # 文件名: 120s 默认预算沿用旧约定 (无后缀 = prune, _base = 剪枝关);
    # 其他预算带 {budget}s 后缀, 避免覆盖 120s 数据
    suffix = SUFFIX[mode]
    if budget != 120.0:
        suffix = f"{suffix}_{int(budget)}s"
    path = OUT / f"{name}_s{seed}{suffix}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
