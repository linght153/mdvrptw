"""用权威 .res BKS 更新比对: ALNS 静态解 vs pyvrp vs 权威解。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.core.instance import instance_from_parsed
from src.core.objectives import calculate_cost
from src.solvers.standard_alns import solve_standard_alns
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

# 权威解 (官方 .res)
AUTHORITATIVE = {
    "pr01": 1083.98, "pr02": 1763.07, "pr03": 2408.42,
    "pr04": 2958.23, "pr05": 3134.04,
}
# pyvrp 30s 参照 (已有基准)
PYVRP = {"pr01": 1277.00, "pr02": 2016.00, "pr03": 3071.00,
         "pr04": 3255.00, "pr05": 3652.00}

CONFIG = {
    "max_iterations": 1500, "max_time_seconds": 20.0, "stagnation_limit": 300,
    "segment_size": 100, "reaction_factor": 0.1, "decay_factor": 1.0,
    "min_selection_prob": 0.005, "initial_temperature": None, "cooling_rate": None,
    "archive_capacity": 20, "parent_selection": "crowding",
    "initial_solution": "nearest", "k_min_ratio": 0.10, "k_max_ratio": 0.40,
    "sigma": 3.0, "local_search_max_iter": 10, "local_search_freq": 5,
    "local_search_on_accept": True, "local_search_final": False,
}


def main():
    print(f"{'实例':<8}{'权威解':<12}{'ALNS静态':<12}{'ALNS gap':<12}"
          f"{'pyvrp':<12}{'pyvrp gap'}")
    print("-" * 68)
    for name in ["pr01", "pr02"]:  # pr01/pr02 快, pr03+ 需更久
        text = open(f"data/mdvrptw_raw/{name}.txt", encoding="utf-8").read()
        inst = instance_from_parsed(parse_mdvrptw(text), name=name)
        archive = solve_standard_alns(inst, CONFIG, np.random.default_rng(42))
        best = min(e[0][0] for e in archive.entries) if archive.entries else float("inf")

        bks = AUTHORITATIVE[name]
        alns_gap = (best - bks) / bks * 100
        pyvrp_gap = (PYVRP[name] - bks) / bks * 100
        print(f"{name:<8}{bks:<12.2f}{best:<12.2f}{alns_gap:<12.2f}%"
              f"{PYVRP[name]:<12.2f}{pyvrp_gap:.2f}%")


if __name__ == "__main__":
    main()
