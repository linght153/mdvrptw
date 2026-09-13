"""方向 A A/B 交替 profile: 可达剪枝 vs 基线, 同进程交替抵消负载漂移。

用法: python experiments/benchmark_ab_reach.py pr06 pr20
输出: 每实例 5 轮 × {base, prune} 20 iter 的墙钟, 报告 ms/iter 中位数与加速比。
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace as dc_replace

from src.core.instance import instance_from_parsed
from src.solvers.standard_alns import solve_standard_alns
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

CONFIG = {
    "max_iterations": 20, "max_time_seconds": 1e9, "stagnation_limit": 300,
    "segment_size": 100, "reaction_factor": 0.1, "decay_factor": 1.0,
    "min_selection_prob": 0.005, "initial_temperature": None, "cooling_rate": None,
    "archive_capacity": 20, "parent_selection": "crowding",
    "initial_solution": "nearest", "k_min_ratio": 0.10, "k_max_ratio": 0.40,
    "sigma": 3.0, "local_search_max_iter": 10, "local_search_freq": 5,
    "local_search_on_accept": True, "local_search_final": False,
}


def timed_run(inst, seed):
    t0 = time.time()
    solve_standard_alns(inst, dict(CONFIG), np.random.default_rng(seed))
    return (time.time() - t0) / CONFIG["max_iterations"] * 1000  # ms/iter


def main():
    names = sys.argv[1:] or ["pr06"]
    for name in names:
        text = open(Path(__file__).resolve().parent.parent / f"data/mdvrptw_raw/{name}.txt",
                    encoding="utf-8").read()
        inst = instance_from_parsed(parse_mdvrptw(text), name=name)
        inst_base = dc_replace(inst, use_reach_prune=False)
        base_ms, prune_ms = [], []
        for rnd in range(5):
            base_ms.append(timed_run(inst_base, 42 + rnd))
            prune_ms.append(timed_run(inst, 42 + rnd))
        b, p = np.median(base_ms), np.median(prune_ms)
        print(f"{name}: base {b:.1f} ms/iter | prune {p:.1f} ms/iter | "
              f"加速 {(1 - p / b) * 100:.1f}%")
        print(f"   每轮: base {[f'{x:.0f}' for x in base_ms]} | "
              f"prune {[f'{x:.0f}' for x in prune_ms]}")


if __name__ == "__main__":
    main()
