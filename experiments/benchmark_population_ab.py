"""种群 A/B runner (探针 2026-09-04): 单实例 × 单 seed × 单臂, 独立进程。

用法: python experiments/benchmark_population_ab.py pr20 42 base|div|cx [iters]
输出: results/population_ab/{inst}_s{seed}_{arm}.json
臂: base=现引擎 | div=+archive_diversity | cx=div+population_crossover
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
       / os.environ.get("MDVRPTW_RESULTS_ROOT", "results")) / "population_ab"
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
ARM_KEYS = {
    "base": {},
    "div": {"archive_diversity": True},
    "cx": {"archive_diversity": True, "population_crossover": True,
           "crossover_rate": 0.2, "crossover_inherit_prob": 0.6},
}

# A2 臂: arm="a2", kappa 由第 5 argv 传入 (默认 0.05)
def arm_keys(arm: str, kappa: float | None) -> tuple[dict, str]:
    if arm == "a2":
        k = 0.05 if kappa is None else kappa
        return {"archive_fitness_dc": True, "archive_fitness_kappa": k}, f"a2_k{k}"
    if arm == "admit":
        return {"archive_admit_relaxed": True}, "admit"
    if arm == "cur":
        return {"stagnation_source": "current"}, "cur"
    if arm in ("d1", "d2"):
        return {"stagnation_recovery_divisor": int(arm[1])}, arm
    if arm in ("act", "qual"):
        # score 解耦探针 (2026-09-05): 只改 score 语义, 准入 = base
        full = "activity" if arm == "act" else "quality"
        return {"score_scheme": full}, arm
    if arm == "pdiv":
        # 互补双亲交叉 (探针 P 2026-09-05): cx 同配置, 第二父代 = 结构距离
        # 最大化 (crossover_pair=diverse) — 与既有 cx 臂 (random) 配对对比
        return {"archive_diversity": True, "population_crossover": True,
                "crossover_rate": 0.2, "crossover_inherit_prob": 0.6,
                "crossover_pair": "diverse"}, "pdiv"
    if arm == "softfix":
        # 罚域轨迹末端硬修复 (探针 F 2026-09-05): soft 纯罚 (不周期播种锚回),
        # 6000 iter 后取轨迹末端做 _soft_repair_hard — FIS 盆地存在性口径
        return {"soft_capacity": True, "soft_repair_interval": 0}, "softfix"
    return dict(ARM_KEYS[arm]), arm


def main():
    name = sys.argv[1]
    seed = int(sys.argv[2])
    arm = sys.argv[3] if len(sys.argv) > 3 else "base"
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else BASE["max_iterations"]
    kappa = float(sys.argv[5]) if len(sys.argv) > 5 else None
    extra, arm_tag = arm_keys(arm, kappa)
    cfg = dict(BASE, max_iterations=iters, stagnation_limit=iters)
    cfg.update(extra)

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
    # best 改进事件轨迹 (2026-09-05 score 解耦诊断): 每次入档记录, 事后压缩
    # 为"池最优首次下探"事件 (行为中性回调)
    adds_log = []
    engine.callbacks["on_archive_add"] = (
        lambda it, dn, rn, c: adds_log.append((it, dn, rn, round(c, 4))))
    archive = engine.run()
    wall = time.time() - t0

    # ── 探针 F (softfix): 罚域轨迹末端硬修复口径 ──────────
    repaired = None
    traj = None
    if arm == "softfix":
        from src.core.objectives import calculate_objectives
        end = engine.last_solution
        excess = engine._cap_excess(end)
        rep = engine._soft_repair_hard(end.copy())
        ok = rep.is_feasible()
        cost = float(calculate_objectives(rep)[0]) if ok else None
        repaired = {"feasible": ok,
                    "cost": round(cost, 4) if ok else None,
                    "tw_violations": len(rep.tw_violations()) if ok else None}
        traj = {"end_excess": round(float(excess), 1),
                "overloaded_iters": engine.soft_stats["overloaded_iters"],
                "max_excess": round(float(engine.soft_stats["max_excess"]), 1)}

    improvements = []
    run_min = float("inf")
    for it, dn, rn, c in adds_log:
        if c < run_min - 1e-9:
            run_min = c
            improvements.append({"iter": it, "destroy": dn, "repair": rn,
                                 "cost": c})

    feas = [(e[0][0], e[1]["solution"]) for e in archive.entries
            if e[1]["solution"].is_feasible()]
    best_cost = min((c for c, _ in feas), default=None)
    best_sol = None
    if best_cost is not None:
        best_sol = min((s for c, s in feas if abs(c - best_cost) < 1e-9),
                       key=lambda s: len(s.tw_violations()))

    # 档案多样性: 路由去重列数 + 平均 Jaccard
    route_sets = []
    for e in archive.entries:
        sol = e[1]["solution"]
        route_sets.append(frozenset((di, tuple(r)) for di, rl in sol.routes.items()
                                    for r in rl))
    distinct = len(set().union(*route_sets)) if route_sets else 0
    jac = None
    if len(route_sets) > 1:
        js = []
        for i in range(len(route_sets)):
            for j in range(i + 1, len(route_sets)):
                u = len(route_sets[i] | route_sets[j])
                js.append(len(route_sets[i] & route_sets[j]) / u if u else 1.0)
        jac = float(np.mean(js))

    out = {
        "instance": name, "seed": seed, "arm": arm, "arm_tag": arm_tag,
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
        "cx_stats": dict(engine.cx_stats),
        "probe_stats": engine.probe_stats,
        "improvements": improvements,
    }
    out.update(duration_stats(best_sol))
    if repaired is not None:
        out["repaired"] = repaired
        out["traj"] = traj
    diag = engine.selector.get_diagnostics()
    out["selector_weights"] = [round(float(w), 4) for w in diag["weights"]]
    out["selector_usage"] = [int(u) for u in diag["usage_count"]]
    out["op_calls"] = dict(engine._op_call_count)
    path = OUT / f"{name}_s{seed}_{arm_tag}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
