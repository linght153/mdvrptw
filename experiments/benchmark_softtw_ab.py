"""TW 罚域探针 runner (预注册 2026-09-06)。

用法: python experiments/benchmark_softtw_ab.py pr20 42 [iters] [lambda_mult]
输出: results/softtw_ab/{inst}_s{seed}_softtw_{iters}iter_m{mult}.json + _routes.json
协议: docs/softtw-probe-prereg-20260906.md。6000iter 固定迭代 → 可并行
(确定性); 每进程 OMP_NUM_THREADS=1。
⚠️ 命名纪律 (2026-09-10 修正): 迭代数与 λ 档 (mult) 必须落文件名 — 曾因同名覆盖
   使 pr20-s42 的 6000iter 记录被 700iter 诊断运行覆写、λ 两档互相覆盖 (表 A2 溯源断链)。
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

OUT = Path(__file__).resolve().parent.parent / "results" / "softtw_ab"
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
SOFTTW = {"soft_tw": True, "soft_tw_lambda_mult": 1.0,
          "soft_tw_repair_interval": 100}


def out_tag(iters: int, mult: float) -> str:
    """输出文件名配置标记 (命名纪律, 2026-09-10 回归锁定)。

    迭代数与 λ 档 (mult) 必须落文件名 — softtw 探针曾因同名覆盖: pr20-s42 的
    6000iter 主记录被 700iter 温度曲线诊断运行覆写, λ 两档 (mult 1.0/0.003) 互相
    覆盖致高 λ 臂整列无盘上证据 (论文表 A2 溯源断链)。见 tests/test_soft_tw.py 回归测试。
    """
    return f"_{iters}iter_m{mult:g}"


def main():
    name = sys.argv[1]
    seed = int(sys.argv[2])
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else BASE["max_iterations"]
    mult = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    text = open(Path(__file__).resolve().parent.parent
                / f"data/mdvrptw_raw/{name}.txt", encoding="utf-8").read()
    inst = instance_from_parsed(parse_mdvrptw(text), name=name)
    customers_copy = [dc_replace(c) for c in inst.customers]
    inst = dc_replace(inst, customers=customers_copy)

    config = dict(BASE)
    config["max_iterations"] = iters
    config.update(SOFTTW)
    config["soft_tw_lambda_mult"] = mult
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
    tag = out_tag(iters, mult)
    (OUT / f"{name}_s{seed}_softtw{tag}_routes.json").write_text(
        json.dumps({"instance": name, "seed": seed, "best_cost": best_cost,
                    "routes": routes, "wall_s": round(wall, 1)},
                   ensure_ascii=False), encoding="utf-8")
    out = {"instance": name, "seed": seed, "arm": "softtw",
           "iters": iters, "lambda_mult": mult,
           "best_cost": best_cost, "wall_s": round(wall, 1),
           "iterations": int(engine.iteration),
           "soft_tw_stats": dict(getattr(engine, "soft_tw_stats", {}))}
    (OUT / f"{name}_s{seed}_softtw{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {name} s{seed} softtw: best {best_cost:.2f} "
          f"stats {out['soft_tw_stats']} {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
