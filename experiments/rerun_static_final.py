"""静态重跑最终汇总: 120s×3seeds 统计 + 大实例充分迭代下界。

输出: results/static_green_all5_rerun.json (更新版, 含统计字段 + lower_bound)
用法: python experiments/rerun_static_final.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import EmissionModel, instance_from_parsed
from src.core.objectives import calculate_objectives
from src.core.initial_solution import nearest_neighbor
from src.core.pareto import non_dominated_sort
from src.evaluation.metrics import compute_metrics
from src.solvers.pyvrp_solver import solve_with_pyvrp, pyvrp_routes_to_solution
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

AUTHORITATIVE = {"pr01": 1083.98, "pr02": 1763.07, "pr03": 2408.42,
                 "pr04": 2958.23, "pr05": 3134.04}
HGSADC = {"pr01": 1074.12, "pr02": 1762.21, "pr03": 2373.65,
          "pr04": 2815.48, "pr05": 2964.65}
INSTANCES = ["pr01", "pr02", "pr03", "pr04", "pr05"]
SEEDS = [42, 43, 44]
# 充分迭代下界 (固定 50000 迭代, COPERT) — 大实例收敛受限的补充证据
LB_FILE = Path("results/lower_bound.json")


def load_lower_bound():
    if LB_FILE.exists():
        return json.loads(LB_FILE.read_text(encoding="utf-8"))
    return {}


def load_rerun(name, em):
    pts = []
    for seed in SEEDS:
        p = Path(f"results/rerun/{name}_{em}_s{seed}.json")
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        pts.extend(tuple(x) for x in d["points"])
    return pts


def main():
    lb_data = load_lower_bound()
    rows = []
    for name in INSTANCES:
        text = open(f"data/mdvrptw_raw/{name}.txt", encoding="utf-8").read()
        # ⚠️ pyvrp 参考点必须按 em 分支用同一排放模型计算 (曾用默认 MEET 模型
        # 算 COPERT 口径 → 单位混用 (MEET kg vs COPERT g) 制造虚假参考点,
        # 致 COPERT IGD 虚高 0.4795/0.4825, 2026-08-31 修复)
        for em in ["MEET", "COPERT"]:
            inst = instance_from_parsed(parse_mdvrptw(text), name=name)
            if em == "COPERT":
                inst.emission_model = EmissionModel(
                    type="COPERT", euro_standard="euro6", speed_kmh=40.0)
            pv = pyvrp_routes_to_solution(
                solve_with_pyvrp(inst, max_runtime_seconds=20, seed=42), inst)
            pv_obj = tuple(calculate_objectives(pv))
            points = load_rerun(name, em)
            best = min(p[0] for p in points)
            # 每 seed 最优 (统计用)
            per_seed_best = []
            for seed in SEEDS:
                p = Path(f"results/rerun/{name}_{em}_s{seed}.json")
                if p.exists():
                    d = json.loads(p.read_text(encoding="utf-8"))
                    per_seed_best.append(min(x[0] for x in d["points"]))
            pop = points + [pv_obj]
            ref_front = [pop[i] for i in non_dominated_sort(pop)[0]]
            inst2 = instance_from_parsed(
                parse_mdvrptw(open(f"data/mdvrptw_raw/{name}.txt", encoding="utf-8").read()),
                name=name)
            if em == "COPERT":
                inst2.emission_model = EmissionModel(
                    type="COPERT", euro_standard="euro6", speed_kmh=40.0)
            initial_obj = tuple(calculate_objectives(
                nearest_neighbor(inst2, np.random.default_rng(42))))
            metrics = compute_metrics(points, initial_obj, ref_front)
            lb = lb_data.get(name) if em == "COPERT" else None
            rows.append({
                "instance": name, "emission": em,
                "best": round(best, 4),
                "per_seed_best": [round(v, 2) for v in per_seed_best],
                "gap_res": round((best / AUTHORITATIVE[name] - 1) * 100, 3),
                "gap_hgs": round((best / HGSADC[name] - 1) * 100, 3),
                "n_points": len(points),
                "hv": metrics["hv"], "igd": metrics["igd"],
                "lower_bound_50000": lb,
            })
            print(f"{name} [{em}]: best={best:.2f} seeds={per_seed_best} "
                  f"gap_res={rows[-1]['gap_res']:+.2f}% HV={metrics['hv']:.4f} "
                  f"lb={lb}", flush=True)

    out = {"rows": rows, "authoritative": AUTHORITATIVE, "hgsadc": HGSADC,
           "seeds": SEEDS,
           "note": "全量重跑 (2026-08-13, 120s×3seeds 独立进程 + 大实例 50000 迭代下界)"}
    Path("results").mkdir(exist_ok=True)
    with open("results/static_green_all5_rerun.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[已保存] results/static_green_all5_rerun.json")


if __name__ == "__main__":
    main()
