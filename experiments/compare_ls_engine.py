"""对比实验: engine LS = two_opt vs RVND (多邻域), 120s 预算 pr01。

预期: RVND (relocate/swap/2-opt*/two_opt 组合) 应优于纯 two_opt,
因为跨路由算子能探索 two_opt 无法到达的解空间。
"""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.core.instance import instance_from_parsed
from src.operators.local_search import rvnd_improve, two_opt_improve
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


def run(ls_fn, label, seed=42):
    from src.alns.engine import ALNSEngine
    from src.alns.selection import SegmentRewardSelection

    destroy_names = list(__import__("src.operators.destroy", fromlist=["DESTROY_OPERATORS"]).DESTROY_OPERATORS.keys())
    repair_names = ["greedy_cost_tw"]
    from src.operators.repair import greedy_cost_tw_insertion

    repair_ops = {"greedy_cost_tw": greedy_cost_tw_insertion}
    selector = SegmentRewardSelection(
        destroy_names, repair_names, segment_size=100, reaction_factor=0.1,
        decay_factor=1.0, min_selection_prob=0.005)
    engine = ALNSEngine(instance=inst, config=CFG, rng=np.random.default_rng(seed),
                        selector=selector, repair_ops=repair_ops,
                        local_search=ls_fn)
    t0 = time.perf_counter()
    archive = engine.run()
    best = min(archive.entries, key=lambda e: e[0][0])
    sol = best[1]["solution"]
    print(f"{label}: cost={best[0][0]:.2f} "
          f"gap={(best[0][0]-1083.98)/1083.98*100:+.1f}% (vs .res)"
          f" TW={len(sol.tw_violations())} "
          f"车辆={sum(len(v) for v in sol.routes.values())} "
          f"耗时={time.perf_counter()-t0:.0f}s")
    return best[0][0]


print("engine LS 对比 (pr01, 120s):\n" + "-" * 60)
run(two_opt_improve, "two_opt   ")
run(rvnd_improve, "RVND      ")
