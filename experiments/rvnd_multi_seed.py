"""RVND 多 seed 稳定性验证: pr01 × 3 seeds × 120s。

确认 RVND (多邻域 LS) 优势跨 seed 成立, 再决定全量重跑。
"""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.core.instance import instance_from_parsed
from src.operators.local_search import rvnd_improve
from src.solvers.standard_alns import solve_standard_alns
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

text = open("data/mdvrptw_raw/pr01.txt", encoding="utf-8").read()
inst = instance_from_parsed(parse_mdvrptw(text), name="pr01")

CFG = {
    "max_iterations": 50000, "max_time_seconds": 120, "stagnation_limit": 2000,
    "segment_size": 100, "reaction_factor": 0.1, "decay_factor": 1.0,
    "min_selection_prob": 0.005, "initial_temperature": None,
    "cooling_rate": None, "archive_capacity": 80,
    "parent_selection": "crowding", "initial_solution": "nearest",
    "k_min_ratio": 0.10, "k_max_ratio": 0.40, "sigma": 3.0,
    "local_search_max_iter": 10, "local_search_freq": 5,
    "local_search_on_accept": True, "local_search_final": True,
}

print("RVND 多 seed (pr01, 120s):\n" + "-" * 56)
costs = []
for seed in [42, 43, 44]:
    from src.alns.engine import ALNSEngine
    from src.alns.selection import SegmentRewardSelection
    from src.operators.repair import greedy_cost_tw_insertion

    destroy_names = list(__import__("src.operators.destroy", fromlist=["DESTROY_OPERATORS"]).DESTROY_OPERATORS.keys())
    selector = SegmentRewardSelection(
        destroy_names, ["greedy_cost_tw"], segment_size=100, reaction_factor=0.1,
        decay_factor=1.0, min_selection_prob=0.005)
    engine = ALNSEngine(instance=inst, config=CFG,
                        rng=np.random.default_rng(seed), selector=selector,
                        repair_ops={"greedy_cost_tw": greedy_cost_tw_insertion},
                        local_search=rvnd_improve)
    t0 = time.perf_counter()
    archive = engine.run()
    best = min(archive.entries, key=lambda e: e[0][0])
    sol = best[1]["solution"]
    cost = best[0][0]
    costs.append(cost)
    print(f"  seed={seed}: cost={cost:.2f} "
          f"gap={(cost-1083.98)/1083.98*100:+.1f}% "
          f"车辆={sum(len(v) for v in sol.routes.values())} "
          f"耗时={time.perf_counter()-t0:.0f}s")
    sys.stdout.flush()

import statistics
print(f"\n  均值: {statistics.mean(costs):.2f} ± {statistics.stdev(costs):.2f}")
print(f"  vs .res 1083.98: gap {statistics.mean(costs)/1083.98*100-100:+.1f}%")
print(f"  vs HGSADC 1074.12: gap {statistics.mean(costs)/1074.12*100-100:+.1f}%")
