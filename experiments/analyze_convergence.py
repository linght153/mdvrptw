"""分析 ALNS 收敛行为: 120s 后是否还有改进空间。

通过轻量运行 (60s) 观察 archive 目标值随迭代的改进轨迹,
判断 600s 预算的预期收益。
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.core.instance import instance_from_parsed
from src.solvers.standard_alns import solve_standard_alns
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

text = open("data/mdvrptw_raw/pr01.txt", encoding="utf-8").read()
inst = instance_from_parsed(parse_mdvrptw(text), name="pr01")

cfg = {
    "max_iterations": 50000, "max_time_seconds": 120, "stagnation_limit": 2000,
    "segment_size": 100, "reaction_factor": 0.1, "decay_factor": 1.0,
    "min_selection_prob": 0.005, "initial_temperature": None,
    "cooling_rate": None, "archive_capacity": 80,
    "parent_selection": "crowding", "initial_solution": "nearest",
    "k_min_ratio": 0.10, "k_max_ratio": 0.40, "sigma": 3.0,
    "local_search_max_iter": 30, "local_search_freq": 10,
    "local_search_on_accept": True, "local_search_final": True,
}

archive = solve_standard_alns(inst, cfg, np.random.default_rng(42))
entries = sorted(archive.entries, key=lambda e: e[0][0])
print("档案最优目标值 (按 iteration):")
for e in entries:
    it = e[1].get("iteration", 0)
    print(f"  iter={it:6d}  cost={e[0][0]:10.2f}  emission={e[0][1]:8.2f}")
print(f"\n最优: {entries[0][0][0]:.2f} (权威 1083.98, gap {(entries[0][0][0]-1083.98)/1083.98*100:+.1f}%)")
