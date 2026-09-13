"""静态 600s 重跑: 单组 runner (实例×排放×seed 独立进程, 防 OOM)。

用法: python experiments/rerun_static_600s_single.py pr01 COPERT 42
配置与 benchmark_static_green_600s.py 一致 (600s 预算)。
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import EmissionModel, instance_from_parsed
from src.core.objectives import calculate_objectives
from src.core.initial_solution import nearest_neighbor
from src.solvers.standard_alns import solve_standard_alns
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

CONFIG = {
    "max_iterations": 50000, "max_time_seconds": 600.0, "stagnation_limit": 2000,
    "segment_size": 100, "reaction_factor": 0.1, "decay_factor": 1.0,
    "min_selection_prob": 0.005, "initial_temperature": None, "cooling_rate": None,
    "archive_capacity": 80, "parent_selection": "crowding", "initial_solution": "nearest",
    "k_min_ratio": 0.10, "k_max_ratio": 0.40, "sigma": 3.0,
    "local_search_max_iter": 30, "local_search_freq": 10,
    "local_search_on_accept": True, "local_search_final": True,
}


def main():
    name, em_label, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    text = open(f"data/mdvrptw_raw/{name}.txt", encoding="utf-8").read()
    inst = instance_from_parsed(parse_mdvrptw(text), name=name)
    if em_label == "COPERT":
        inst.emission_model = EmissionModel(
            type="COPERT", euro_standard="euro6", speed_kmh=40.0)

    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    archive = solve_standard_alns(inst, CONFIG, rng)
    elapsed = time.perf_counter() - started
    points = [tuple(entry[0]) for entry in archive.entries
              if entry[1].get("solution") is None
              or entry[1]["solution"].is_feasible()]
    init_sol = nearest_neighbor(inst, np.random.default_rng(seed))
    initial_obj = calculate_objectives(init_sol)

    out = {
        "instance": name, "emission": em_label, "seed": seed,
        "points": [list(p) for p in points],
        "initial_obj": list(initial_obj),
        "elapsed_s": round(elapsed, 1),
    }
    Path("results/rerun600").mkdir(parents=True, exist_ok=True)
    with open(f"results/rerun600/{name}_{em_label}_s{seed}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[OK] {name} {em_label} s{seed}: {len(points)} 点, "
          f"best={min(p[0] for p in points):.2f}, {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
