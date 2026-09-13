"""方向①车场分配重平衡 A/B runner (预注册 2026-09-06)。

用法: python experiments/benchmark_rebalance_ab.py pr20 42 both [interval] [iters] [time_budget_s]
输出: results/rebalance_ab/{inst}_s{seed}_{arm}{_mode}.json (含末端解结构指标)
       + {inst}_s{seed}_{arm}{_mode}_routes.json (末端 best 解 dump)
arm: base | removal | redispatch | both (base 也 dump, 第 1 级过程指标参照)
协议: docs/rebalance-ab-prereg-20260906.md。6000iter 固定迭代, 独立进程串行。
⚠️ 命名纪律 (2026-09-10 修正): 带时间预算的批次在文件名追加 _{budget}s 标记 —
   曾因同名覆盖把 pr20×s42-44×both 的 6000iter 记录盖成 600s 运行 (表 3 数据溯源断链)。
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

OUT = Path(__file__).resolve().parent.parent / "results" / "rebalance_ab"
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
ARMS = ("base", "removal", "redispatch", "both")
INTERVAL = 200  # 主炮档位: 6000 iter → ~30 次 rebalance pass


def out_tag(time_budget: float, iters: int) -> str:
    """输出文件名模式标记 (命名纪律, 2026-09-10 回归锁定)。

    时间预算批必须带 `_{budget}s` 后缀, 与固定迭代批分文件 — benchmark_rebalance_ab
    的 600s 扩展批曾与 6000iter 批同名, 覆写 pr20×s42-44×both 的 6000iter 记录,
    致论文表 3 双开列溯源断链。见 tests/test_rebalance.py 同名回归测试。
    """
    return f"_{time_budget:.0f}s" if time_budget > 0 else ""


def arm_config(arm: str, interval: int) -> dict:
    if arm == "base":
        return {}
    return {"rebalance_mode": arm, "rebalance_interval": interval}


def structure_metrics(inst, routes):
    """末端解结构指标 (第 1 级过程指标; 口径同 analyze_structure_diff)。"""
    nd = inst.num_depots
    dm = inst.distance_matrix
    n = inst.num_customers
    actual = {}
    for di, rl in routes.items():
        for r in rl:
            for g in r:
                actual[g] = di
    pen = 0.0
    geo = 0
    for g, di in actual.items():
        d_act = dm[di][g]
        d_near = min(dm[k][g] for k in range(nd))
        if d_act > d_near + 1e-9:
            pen += 2.0 * (d_act - d_near)
        if abs(d_act - d_near) <= 1e-9:
            geo += 1
    return {"mismatch_penalty": round(pen, 3),
            "geo_consistent_frac": round(geo / len(actual), 4),
            "n_mismatch": int(sum(1 for g, di in actual.items()
                                  if dm[di][g] > min(dm[k][g]
                                                     for k in range(nd)) + 1e-9))}


def main():
    name = sys.argv[1]
    seed = int(sys.argv[2])
    arm = sys.argv[3] if len(sys.argv) > 3 else "both"
    interval = int(sys.argv[4]) if len(sys.argv) > 4 else INTERVAL
    iters = int(sys.argv[5]) if len(sys.argv) > 5 else BASE["max_iterations"]
    time_budget = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    assert arm in ARMS, f"arm ∈ {ARMS}"
    text = open(Path(__file__).resolve().parent.parent
                / f"data/mdvrptw_raw/{name}.txt", encoding="utf-8").read()
    inst = instance_from_parsed(parse_mdvrptw(text), name=name)
    customers_copy = [dc_replace(c) for c in inst.customers]
    inst = dc_replace(inst, customers=customers_copy)

    config = dict(BASE)
    config["max_iterations"] = iters
    if time_budget > 0:
        # 600s 口径 = dump_base600s_routes CONFIG (stagnation 300 + 重启),
        # 与既有 base-600s dump (connectivity/base600s_routes) 同配置可比
        config["max_time_seconds"] = time_budget
        config["max_iterations"] = 10_000_000
        config["stagnation_limit"] = 300
    config.update(arm_config(arm, interval))
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
    sm = structure_metrics(inst, routes)
    tag = out_tag(time_budget, iters)
    mode = tag[1:] if tag else f"{iters}iter"
    dump = {"instance": name, "seed": seed, "arm": arm, "mode": mode,
            "best_cost": best_cost,
            "routes": routes, "wall_s": round(wall, 1)}
    (OUT / f"{name}_s{seed}_{arm}{tag}_routes.json").write_text(
        json.dumps(dump, ensure_ascii=False), encoding="utf-8")
    out = {"instance": name, "seed": seed, "arm": arm, "mode": mode,
           "budget_s": (time_budget or None),
           "best_cost": best_cost, "wall_s": round(wall, 1),
           "iterations": engine.iteration_counter if hasattr(
               engine, "iteration_counter") else None,
           **sm}
    (OUT / f"{name}_s{seed}_{arm}{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {name} s{seed} {arm}: best {best_cost:.2f} {sm} "
          f"{wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
