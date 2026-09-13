"""M3 改进门+教育 A/B runner (2026-09-07): 单实例 × 单 seed × 单臂, 独立进程。

用法: python experiments/benchmark_population_m3.py pr20 42 popcxg|popcxge
输出: results/population_m3/{inst}_s{seed}_{arm}.json
臂: popcxg = partition@rate0.2 + gate=improve | popcxge = popcxg + educate
    (burst 0.20, ls_budget 50); κ=0
配对参照: pop = population_m1/...pop.json, popcx_part = population_m2/...popcx_part.json
结构字段公式与 experiments/analyze_structure_diff.py analyze() 同款。
协议: docs/ma-alns-m3-prereg-20260907.md
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from src.core.solution import duration_stats
from src.alns.selection import SegmentRewardSelection
from src.alns.engine import ALNSEngine
from src.operators.local_search import rvnd_improve
from src.operators.destroy import DESTROY_OPERATORS
from src.operators.repair import greedy_cost_tw_insertion
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw
from dataclasses import replace as dc_replace

OUT = (Path(__file__).resolve().parent.parent
       / os.environ.get("MDVRPTW_RESULTS_ROOT", "results")) / "population_m3"
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

_POP = {"population_mode": True, "population_size": 8,
        "population_tournament": 2, "population_dc_kappa": 0.0}
_PART = {"population_cx_mode": "partition", "population_cx_rate": 0.2}

ARMS = {
    "popcxg": dict(_POP, **_PART, population_cx_gate="improve"),
    "popcxge": dict(_POP, **_PART, population_cx_gate="improve",
                    population_cx_educate=True, population_educate_burst_ratio=0.20,
                    population_educate_ls_budget=50),
}


def route_set(sol):
    return frozenset((di, tuple(r)) for di, rl in sol.routes.items() for r in rl)


def pool_spread(population_final) -> float | None:
    if not population_final or len(population_final) < 2:
        return None
    keys = [route_set(sol) for _, sol in population_final]
    js = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            u = len(keys[i] | keys[j])
            js.append(1.0 - (len(keys[i] & keys[j]) / u if u else 1.0))
    return float(np.mean(js))


def structure_metrics(inst, sol):
    nd = inst.num_depots
    n = inst.num_customers
    dm = np.asarray(inst.distance_matrix, dtype=float)
    actual = np.full(n, -1, dtype=int)
    for di, rl in sol.routes.items():
        for r in rl:
            for g in r:
                actual[g - nd] = di
    assert (actual >= 0).all()
    d_near = dm[:nd, nd:].min(axis=0)
    actual_d = dm[actual, np.arange(nd, nd + n)]
    extra = actual_d - d_near
    rank = (dm[:nd, nd:] < actual_d[None, :]).sum(axis=0)
    mm = rank > 0
    return {
        "best_mismatch_penalty": round(float(np.sum(2.0 * extra)), 1),
        "best_n_mismatch": int(mm.sum()),
        "best_geo_consistent_frac": round(float((rank == 0).mean()), 4),
    }


def main():
    name = sys.argv[1]
    seed = int(sys.argv[2])
    arm = sys.argv[3]
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else BASE["max_iterations"]
    cfg = dict(BASE, max_iterations=iters, stagnation_limit=iters)
    cfg.update(ARMS[arm])

    text = open(Path(__file__).resolve().parent.parent
                / f"data/mdvrptw_raw/{name}.txt", encoding="utf-8").read()
    inst = instance_from_parsed(parse_mdvrptw(text), name=name)
    inst = dc_replace(inst, customers=[dc_replace(c) for c in inst.customers])

    sel = SegmentRewardSelection(
        list(DESTROY_OPERATORS.keys()), ["greedy_cost_tw"],
        segment_size=cfg["segment_size"], reaction_factor=cfg["reaction_factor"],
        decay_factor=cfg["decay_factor"], min_selection_prob=cfg["min_selection_prob"])
    t0 = time.time()
    engine = ALNSEngine(inst, cfg, np.random.default_rng(seed), selector=sel,
                        repair_ops={"greedy_cost_tw": greedy_cost_tw_insertion},
                        local_search=rvnd_improve)
    archive = engine.run()
    wall = time.time() - t0

    feas = [(e[0][0], e[1]["solution"]) for e in archive.entries
            if e[1]["solution"].is_feasible()]
    best_cost = min((c for c, _ in feas), default=None)
    best_sol = None
    if best_cost is not None:
        best_sol = min((s for c, s in feas if abs(c - best_cost) < 1e-9),
                       key=lambda s: len(s.tw_violations()))

    route_sets = [route_set(e[1]["solution"]) for e in archive.entries]
    distinct = len(set().union(*route_sets)) if route_sets else 0
    jac = None
    if len(route_sets) > 1:
        js = []
        for i in range(len(route_sets)):
            for j in range(i + 1, len(route_sets)):
                u = len(route_sets[i] | route_sets[j])
                js.append(len(route_sets[i] & route_sets[j]) / u if u else 1.0)
        jac = float(np.mean(js))

    pf = engine.population_final
    out = {
        "instance": name, "seed": seed, "arm": arm, "arm_tag": arm,
        "iterations": iters,
        "wall_s": round(wall, 1),
        "best_feasible_cost": round(best_cost, 4) if best_cost else None,
        "vehicles": best_sol.total_vehicles() if best_sol else None,
        "tw_violations": len(best_sol.tw_violations()) if best_sol else None,
        "served": len(best_sol.all_served_customers()) if best_sol else None,
        "num_customers": inst.num_customers,
        "archive_entries": len(archive.entries),
        "archive_distinct_routes": int(distinct),
        "archive_mean_jaccard": round(jac, 4) if jac is not None else None,
        "cx_iterations": engine.cx_stats["iterations"],
        "probe_stats": engine.probe_stats,
        "population_stats": dict(engine.population_stats)
        if engine.population_stats is not None else None,
        "pool_final_spread": round(pool_spread(pf), 4) if pf else None,
        "pool_feasible_members": (
            sum(1 for _, s in pf if s.is_feasible()) if pf else None),
    }
    if best_sol is not None:
        out.update(structure_metrics(inst, best_sol))
    out.update(duration_stats(best_sol))
    path = OUT / f"{name}_s{seed}_{arm}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
