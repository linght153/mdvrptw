"""方向②破坏性教育 A/B runner (预注册 2026-09-06)。

用法: python experiments/benchmark_educate_ab.py pr20 42 [iters]
输出: results/educate_ab/{inst}_s{seed}_educate.json (成本+档案/教育统计)
       + {inst}_s{seed}_educate_routes.json (末端 best dump)
协议: docs/educate-ab-prereg-20260906.md。cx 同配置 (div + population_
crossover), 仅加 educate_mode="deep" (规格默认档)。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from src.alns.selection import SegmentRewardSelection
from src.alns.engine import ALNSEngine
from src.operators.local_search import rvnd_improve
from src.operators.destroy import DESTROY_OPERATORS
from src.operators.repair import greedy_cost_tw_insertion
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw
from dataclasses import replace as dc_replace

OUT = Path(__file__).resolve().parent.parent / "results" / "educate_ab"
OUT.mkdir(parents=True, exist_ok=True)

BASE = {
    "max_iterations": 6000, "max_time_seconds": 1e9, "stagnation_limit": 6000,
    "segment_size": 100, "reaction_factor": 0.1, "decay_factor": 1.0,
    "min_selection_prob": 0.005, "initial_temperature": None, "cooling_rate": None,
    "archive_capacity": 20, "parent_selection": "crowding",
    "initial_solution": "nearest", "k_min_ratio": 0.10, "k_max_ratio": 0.40,
    "sigma": 3.0, "local_search_max_iter": 10, "local_search_freq": 5,
    "local_search_on_accept": True, "local_search_final": False,
}
CX = {"archive_diversity": True, "population_crossover": True,
      "crossover_rate": 0.2, "crossover_inherit_prob": 0.6}
EDUCATE = {"educate_mode": "deep", "educate_burst_ratio": 0.20,
           "educate_burst_max_iter": 1, "educate_ls_budget": 300}


def main():
    name = sys.argv[1]
    seed = int(sys.argv[2])
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else BASE["max_iterations"]
    text = open(Path(__file__).resolve().parent.parent
                / f"data/mdvrptw_raw/{name}.txt", encoding="utf-8").read()
    inst = instance_from_parsed(parse_mdvrptw(text), name=name)
    customers_copy = [dc_replace(c) for c in inst.customers]
    inst = dc_replace(inst, customers=customers_copy)

    config = dict(BASE)
    config["max_iterations"] = iters
    config.update(CX)
    config.update(EDUCATE)
    destroy_names = list(DESTROY_OPERATORS.keys())
    selector = SegmentRewardSelection(
        destroy_names, ["greedy_cost_tw"],
        segment_size=config["segment_size"],
        reaction_factor=config["reaction_factor"],
        decay_factor=config["decay_factor"],
        min_selection_prob=config["min_selection_prob"],
    )
    t0 = time.time()
    engine = ALNSEngine(instance=inst, config=config,
                        rng=np.random.default_rng(seed), selector=selector,
                        repair_ops={"greedy_cost_tw": greedy_cost_tw_insertion},
                        local_search=rvnd_improve)
    archive = engine.run()
    wall = time.time() - t0
    best_entry = min(archive.entries, key=lambda e: e[0][0])
    best_cost = best_entry[0][0]
    sol = best_entry[1]["solution"]
    assert sol.is_feasible(), "dump 解必须可行"
    routes = {int(di): [list(r) for r in rl] for di, rl in sol.routes.items()}
    (OUT / f"{name}_s{seed}_educate_routes.json").write_text(
        json.dumps({"instance": name, "seed": seed, "best_cost": best_cost,
                    "routes": routes, "wall_s": round(wall, 1)},
                   ensure_ascii=False), encoding="utf-8")
    from src.alns.archive import ParetoArchive
    pool_keys = set()
    for e in archive.entries:
        pool_keys |= ParetoArchive._route_keys(e[1])
    cx_stats = getattr(engine, "cx_stats", {})
    out = {
        "instance": name, "seed": seed, "arm": "educate",
        "best_cost": best_cost, "wall_s": round(wall, 1),
        "iterations": config["max_iterations"],
        "vehicles": sol.total_vehicles(), "tw_violations": len(sol.tw_violations()),
        "archive_entries": archive.size,
        "archive_distinct_routes": len(pool_keys),
        "archive_mean_jaccard": round(
            float(np.mean([ParetoArchive._jaccard_dist(
                pool_keys, ParetoArchive._route_keys(e[1]))
                for e in archive.entries])), 4) if archive.size > 1 else None,
        "cx_iterations": cx_stats.get("iterations", 0),
        "educate_attempts": int(cx_stats.get("educate_attempts", 0)),
        "educate_success": int(cx_stats.get("educate_success", 0)),
        "educate_success_ratio": round(
            cx_stats.get("educate_success", 0) /
            max(cx_stats.get("educate_attempts", 1), 1), 4),
        "educate_last_delta": float(cx_stats.get("educate_delta", 0.0)),
        "cx_stats": {k: v for k, v in cx_stats.items()},
    }
    # educate 触发与成功累计 (engine cx_stats 计数)
    (OUT / f"{name}_s{seed}_educate.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {name} s{seed} educate: best {best_cost:.2f} "
          f"cx_stats {cx_stats} wall {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
