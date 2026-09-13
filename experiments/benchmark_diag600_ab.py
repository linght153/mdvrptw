"""600s 组合诊断 runner (预注册 2026-09-06)。

用法: python experiments/benchmark_diag600_ab.py pr20 42 base|educate|combo
输出: results/diag600_ab/{inst}_s{seed}_{arm}.json (成本+机制统计+结构
指标) + {inst}_s{seed}_{arm}_routes.json (末端 best dump)
协议: docs/diag600-prereg-20260906.md。600s 预算 stagnation 300 (与
dump_base600s_routes 同配置), 串行独立进程, 同批配对。
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
from src.alns.archive import ParetoArchive
from src.operators.local_search import rvnd_improve
from src.operators.destroy import DESTROY_OPERATORS
from src.operators.repair import greedy_cost_tw_insertion
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw
from dataclasses import replace as dc_replace

OUT = Path(__file__).resolve().parent.parent / "results" / "diag600_ab"
OUT.mkdir(parents=True, exist_ok=True)

CFG = {
    "max_iterations": 10_000_000, "max_time_seconds": 600.0,
    "stagnation_limit": 300, "segment_size": 100, "reaction_factor": 0.1,
    "decay_factor": 1.0, "min_selection_prob": 0.005,
    "initial_temperature": None, "cooling_rate": None, "archive_capacity": 20,
    "parent_selection": "crowding", "initial_solution": "nearest",
    "k_min_ratio": 0.10, "k_max_ratio": 0.40, "sigma": 3.0,
    "local_search_max_iter": 10, "local_search_freq": 5,
    "local_search_on_accept": True, "local_search_final": False,
}
# cx 同配置 (与 6000iter educate 批 / 09-04 cx 臂一致): educate 只在
# population_crossover 分支内生效
CX = {"archive_diversity": True, "population_crossover": True,
      "crossover_rate": 0.2, "crossover_inherit_prob": 0.6}
ARMS = {
    "base": {},
    "educate": dict(CX, **{"educate_mode": "deep", "educate_burst_ratio": 0.20,
                           "educate_burst_max_iter": 1,
                           "educate_ls_budget": 300}),
    "combo": dict(CX, **{"educate_mode": "deep", "educate_burst_ratio": 0.20,
                         "educate_burst_max_iter": 1,
                         "educate_ls_budget": 300,
                         "rebalance_mode": "both",
                         "rebalance_interval": 200}),
}


def structure_metrics(inst, routes):
    nd = inst.num_depots
    dm = inst.distance_matrix
    actual = {}
    for di, rl in routes.items():
        for r in rl:
            for g in r:
                actual[g] = di
    pen, geo = 0.0, 0
    for g, di in actual.items():
        d_near = min(dm[k][g] for k in range(nd))
        d_act = dm[di][g]
        if d_act > d_near + 1e-9:
            pen += 2.0 * (d_act - d_near)
        else:
            geo += 1
    return {"mismatch_penalty": round(pen, 3),
            "geo_consistent_frac": round(geo / len(actual), 4)}


def main():
    name = sys.argv[1]
    seed = int(sys.argv[2])
    arm = sys.argv[3] if len(sys.argv) > 3 else "combo"
    budget = float(sys.argv[4]) if len(sys.argv) > 4 else CFG["max_time_seconds"]
    assert arm in ARMS, f"arm ∈ {list(ARMS)}"
    text = open(Path(__file__).resolve().parent.parent
                / f"data/mdvrptw_raw/{name}.txt", encoding="utf-8").read()
    inst = instance_from_parsed(parse_mdvrptw(text), name=name)
    customers_copy = [dc_replace(c) for c in inst.customers]
    inst = dc_replace(inst, customers=customers_copy)

    config = dict(CFG)
    config["max_time_seconds"] = budget
    config.update(ARMS[arm])
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
    (OUT / f"{name}_s{seed}_{arm}_routes.json").write_text(
        json.dumps({"instance": name, "seed": seed, "arm": arm,
                    "best_cost": best_cost, "routes": routes},
                   ensure_ascii=False), encoding="utf-8")
    pool_keys = set()
    for e in archive.entries:
        pool_keys |= ParetoArchive._route_keys(e[1])
    cx = getattr(engine, "cx_stats", {})
    rb = getattr(engine, "rebalance_stats", {})
    out = {"instance": name, "seed": seed, "arm": arm,
           "best_cost": best_cost, "wall_s": round(wall, 1),
           "iterations": int(getattr(engine, "iteration", 0)
                             or getattr(engine, "_iteration", 0)),
           "vehicles": sol.total_vehicles(),
           "tw_violations": len(sol.tw_violations()),
           "archive_distinct_routes": len(pool_keys),
           "cx_stats": dict(cx),
           "rebalance_stats": dict(rb),
           **structure_metrics(inst, routes)}
    (OUT / f"{name}_s{seed}_{arm}.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {name} s{seed} {arm}: best {best_cost:.2f} "
          f"pen {out['mismatch_penalty']:.0f} cx {cx.get('iterations', 0)} "
          f"succ {cx.get('educate_success', 0)}/{cx.get('educate_attempts', 0)} "
          f"rb {rb.get('redispatch_moves', 0)} {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
