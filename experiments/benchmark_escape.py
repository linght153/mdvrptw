"""pr20 逃逸策略实验: V1-V4 变体 × seeds (120s 时间公平)。

用法: python experiments/benchmark_escape.py {inst} {seed} {variant}
variant: v1 重启频率 (stagnation 300→100)
         v2 停滞深破坏 (k_max 0.6n→0.9n)
         v3 SA 快速冷却 (cooling_rate ≈ exp(ln(1e-4)/1500))
         v4 重启清空档案 (restart_clear_archive)
输出: results/escape_pr20/{variant}_{inst}_s{seed}.json
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

BASE = {
    "max_iterations": 10_000_000, "max_time_seconds": 120.0, "stagnation_limit": 300,
    "segment_size": 100, "reaction_factor": 0.1, "decay_factor": 1.0,
    "min_selection_prob": 0.005, "initial_temperature": None, "cooling_rate": None,
    "archive_capacity": 20, "parent_selection": "crowding",
    "initial_solution": "nearest", "k_min_ratio": 0.10, "k_max_ratio": 0.40,
    "sigma": 3.0, "local_search_max_iter": 10, "local_search_freq": 5,
    "local_search_on_accept": True, "local_search_final": False,
}

VARIANTS = {
    "v1": {"stagnation_limit": 100},                          # 重启阈值 33
    "v2": {"deep_destroy_max_ratio": 0.9},                    # 停滞深破坏 90%
    "v3": {"cooling_rate": float(np.exp(np.log(1e-4) / 1500))},  # 1500 iter 冷却
    "v4": {"restart_clear_archive": True},
    "v5": {"stagnation_deep_ratio": 0.15},                    # 深破坏提前 (停滞 45)
    "v6": {"stagnation_limit": 100, "stagnation_deep_ratio": 0.15},  # 组合
}

OUT = Path(__file__).resolve().parent.parent / "results" / "escape_pr20"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    inst_name = sys.argv[1]
    seed = int(sys.argv[2])
    variant = sys.argv[3]
    cfg = dict(BASE, **VARIANTS[variant])
    text = open(Path(__file__).resolve().parent.parent
                / f"data/mdvrptw_raw/{inst_name}.txt", encoding="utf-8").read()
    inst = instance_from_parsed(parse_mdvrptw(text), name=inst_name)
    t0 = time.time()
    archive = solve_standard_alns(inst, cfg, np.random.default_rng(seed))
    feasible = [(e[0][0], e[1]) for e in archive.entries]
    best_cost = min(c for c, _ in feasible) if feasible else None
    iters = (max(e[1].get("iteration", 0) for e in archive.entries)
             if archive.entries else None)
    out = {
        "instance": inst_name, "seed": seed, "variant": variant,
        "best_feasible_cost": best_cost, "wall_s": round(time.time() - t0, 2),
        "iterations": iters, "archive_entries": len(archive.entries),
    }
    (OUT / f"{variant}_{inst_name}_s{seed}.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
