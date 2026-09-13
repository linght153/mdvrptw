"""M5 runner (2026-09-07): 压缩臂 6000iter(Phase A) + 600s 平台表(Phase B)。

用法(Phase A, 6000iter):
  python experiments/benchmark_population_m5.py pr20 42 popcxgeC
Phase B(600s 时间预算, 必须串行独立进程):
  python experiments/benchmark_population_m5.py pr20 42 popcxgeC 600
  python experiments/benchmark_population_m5.py pr20 42 base 600
  python experiments/benchmark_population_m5.py pr20 42 popcxge 600
输出: results/population_m5/{inst}_s{seed}_{arm}.json (600s 组进 600s/ 子目录)
协议: docs/ma-alns-m5-prereg-20260907.md
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
       / os.environ.get("MDVRPTW_RESULTS_ROOT", "results")) / "population_m5"
OUT.mkdir(parents=True, exist_ok=True)

BASE6000 = {
    "max_iterations": 6000, "max_time_seconds": 1e9, "stagnation_limit": 6000,
    "segment_size": 100, "reaction_factor": 0.1, "decay_factor": 1.0,
    "min_selection_prob": 0.005, "initial_temperature": None, "cooling_rate": None,
    "archive_capacity": 20, "parent_selection": "crowding",
    "initial_solution": "nearest", "k_min_ratio": 0.10, "k_max_ratio": 0.40,
    "sigma": 3.0, "local_search_max_iter": 10, "local_search_freq": 5,
    "local_search_on_accept": True, "local_search_final": False,
}

# 600s 口径 = probe_connectivity CONFIG 同款 (CLAUDE.md 红线)
BASE600 = {
    "max_iterations": 10_000_000, "max_time_seconds": 600.0, "stagnation_limit": 300,
    "segment_size": 100, "reaction_factor": 0.1, "decay_factor": 1.0,
    "min_selection_prob": 0.005, "initial_temperature": None, "cooling_rate": None,
    "archive_capacity": 20, "parent_selection": "crowding",
    "initial_solution": "nearest", "k_min_ratio": 0.10, "k_max_ratio": 0.40,
    "sigma": 3.0, "local_search_max_iter": 10, "local_search_freq": 5,
    "local_search_on_accept": True, "local_search_final": False,
}

POPCXGE = {"population_mode": True, "population_size": 8,
           "population_tournament": 2, "population_dc_kappa": 0.0,
           "population_cx_mode": "partition", "population_cx_rate": 0.2,
           "population_cx_gate": "improve",
           "population_cx_educate": True, "population_educate_burst_ratio": 0.20,
           "population_educate_ls_budget": 50}
# 压缩臂 (M5): 教育次数减半 × 单次成本 ~1/3
POPCXGEC = dict(POPCXGE, population_cx_rate=0.1,
                population_educate_ls_budget=15)
# 中间档 (M6): 教育成本 ~2/3 × 次数 ~3/4 → 总 ~1/2
POPCXGEM = dict(POPCXGE, population_cx_rate=0.15,
                population_educate_ls_budget=25)
# 参数稳健性批 (2026-09-08): κ ∈ {0.05, 0.10} (基准 0.0), p_c ∈ {0.10, 0.30}
# (基准 0.20) — 配对基准 = population_m3 的 popcxge (同配置同种子)
POPCXGE_K05 = dict(POPCXGE, population_dc_kappa=0.05)
POPCXGE_K10 = dict(POPCXGE, population_dc_kappa=0.10)
POPCXGE_C10 = dict(POPCXGE, population_cx_rate=0.10)
POPCXGE_C30 = dict(POPCXGE, population_cx_rate=0.30)
# C4 no-refresh 对照 (2026-09-09): 关闭成员级停滞驱逐 — 配对基准 = popcxge
POPCXGE_NOREFRESH = dict(POPCXGE, population_member_refresh=False)

ARMS = {
    "base": {},
    "popcxge": POPCXGE,
    "popcxgeC": POPCXGEC,
    "popcxgem": POPCXGEM,
    "popcxge_k05": POPCXGE_K05,
    "popcxge_k10": POPCXGE_K10,
    "popcxge_c10": POPCXGE_C10,
    "popcxge_c30": POPCXGE_C30,
    "popcxge_norefresh": POPCXGE_NOREFRESH,
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
    mode = sys.argv[4] if len(sys.argv) > 4 else "6000"
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if mode == "6000":
        cfg = dict(BASE6000)
        iters = 6000
    elif mode == "600":
        # 历史 600s 平台表口径 (BASE600: stagnation 300 允许重启) — 勿动语义
        cfg = dict(BASE600)
        iters = None
    elif mode.isdigit():
        # 任意秒时间预算 (等墙钟对照: 语义 = BASE6000 主判别, 仅停止条件
        # 由"6000 次迭代"换成时间 — 保持不重启, 与 BASE600 的 stagnation 300
        # 语义区分)
        cfg = dict(BASE6000)
        cfg["max_iterations"] = 10_000_000
        cfg["stagnation_limit"] = 10_000_000
        cfg["max_time_seconds"] = float(mode)
        iters = None
    else:
        raise ValueError(f"mode must be 6000|600|<seconds>, got {mode!r}")
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
        "mode": mode, "budget_s": 600.0 if mode == "600" else None,
        "iterations": engine.iteration,
        "wall_s": round(wall, 1),
        "best_feasible_cost": round(best_cost, 4) if best_cost else None,
        "vehicles": best_sol.total_vehicles() if best_sol else None,
        "tw_violations": len(best_sol.tw_violations()) if best_sol else None,
        "served": len(best_sol.all_served_customers()) if best_sol else None,
        "num_customers": inst.num_customers,
        "archive_entries": len(archive.entries),
        "archive_distinct_routes": int(distinct),
        "archive_mean_jaccard": round(jac, 4) if jac is not None else None,
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
    if mode == "6000":
        sub = OUT
    elif mode == "600":
        sub = OUT / "600s"
    else:
        sub = OUT / f"{mode}s"
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / f"{name}_s{seed}_{arm}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
