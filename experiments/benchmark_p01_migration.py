"""基准复现验证：p01 实例 25k 迭代 Standard ALNS。

对照旧项目 README 基准：Standard ALNS p01 均值 589.3、最优 576.9（BKS 576.87）。
用 3 个 seed 各跑一次，比较均值与最优。
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import load_instance
from src.solvers.standard_alns import solve_standard_alns

CONFIG = {
    "max_iterations": 25000,
    "max_time_seconds": 600,
    "stagnation_limit": 500,
    "segment_size": 100,
    "reaction_factor": 0.1,
    "decay_factor": 1.0,
    "min_selection_prob": 0.005,
    "initial_temperature": None,
    "cooling_rate": None,
    "archive_capacity": 80,
    "parent_selection": "crowding",
    "initial_solution": "nearest",
    "k_min_ratio": 0.10,
    "k_max_ratio": 0.40,
    "sigma": 3.0,
    "local_search_max_iter": 30,
    "local_search_freq": 10,
    "local_search_on_accept": True,
    "local_search_final": True,
}

SEEDS = [42, 43, 44]
BKS = 576.87  # Cordeau p01 公开最优解


def main():
    inst = load_instance("p01")
    results = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        started = time.perf_counter()
        archive = solve_standard_alns(inst, CONFIG, rng)
        elapsed = time.perf_counter() - started
        best = min(entry[0][0] for entry in archive.entries)
        results.append(best)
        print(f"seed={seed}: best_cost={best:.2f}  gap={100*(best-BKS)/BKS:+.3f}%  time={elapsed:.1f}s")

    mean = float(np.mean(results))
    best = float(np.min(results))
    print(f"\n均值: {mean:.2f}  (旧项目基准: 589.3)")
    print(f"最优: {best:.2f}  (旧项目基准: 576.9, BKS: {BKS})")
    print(f"均值差距 vs 旧项目: {100*(mean-589.3)/589.3:+.2f}%")


if __name__ == "__main__":
    main()
