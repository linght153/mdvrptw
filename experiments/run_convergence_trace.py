"""收敛轨迹采集 (论文图 5 数据源): BASE / MA-ALNS(popcxge) 6000iter × s42-44。

用法: python experiments/run_convergence_trace.py pr15 42 base
输出: results/convergence_trace/{inst}_s{seed}_{arm}_trace.json
  samples = [[iter, best_sofar], ...] (每 100 迭代采样; best = 档案可行最小
  成本, 与 runner 的 best_feasible_cost 同源)

引擎零改动: 复用 ALNSEngine 既有 on_iteration 回调 (base 路径 engine.py:920,
种群路径 :1570, 迭代末触发)。回调只读 archive, 不消耗 rng → 结果与
population_m5/population_ab/population_m3 既存批逐位一致; 脚本断言端点
best == 参照 JSON 的 best_feasible_cost (±1e-3) 才写盘。
配置与引擎构造逐行复用 experiments/benchmark_population_m5.py (BASE6000 +
ARMS['popcxge']); 6000 次迭代固定口径, 并行安全 (确定性)。
"""
import json
import os
import sys
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.instance import instance_from_parsed  # noqa: E402
from src.alns.selection import SegmentRewardSelection  # noqa: E402
from src.alns.engine import ALNSEngine  # noqa: E402
from src.operators.local_search import rvnd_improve  # noqa: E402
from src.operators.destroy import DESTROY_OPERATORS  # noqa: E402
from src.operators.repair import greedy_cost_tw_insertion  # noqa: E402
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw  # noqa: E402
from dataclasses import replace as dc_replace  # noqa: E402

runner = SourceFileLoader(
    "benchmark_population_m5",
    str(ROOT / "experiments" / "benchmark_population_m5.py"),
).load_module()

OUT = Path(os.environ.get(
    "MDVRPTW_TRACE_OUT", str(ROOT / "results" / "convergence_trace")))
_REF_ROOT = os.environ.get("MDVRPTW_TRACE_REF_ROOT")  # v3: 指向 durfix 结果根
SAMPLE_EVERY = 100


def _feas_min(archive):
    m = None
    for e in archive.entries:
        sol = e[1]["solution"]
        if sol.is_feasible():
            c = e[0][0]
            m = c if m is None else min(m, c)
    return m


def main():
    name, seed, arm = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    assert arm in ("base", "popcxge"), arm
    cfg = dict(runner.BASE6000)
    cfg.update(runner.ARMS[arm])

    text = open(ROOT / f"data/mdvrptw_raw/{name}.txt", encoding="utf-8").read()
    inst = instance_from_parsed(parse_mdvrptw(text), name=name)
    inst = dc_replace(inst, customers=[dc_replace(c) for c in inst.customers])

    sel = SegmentRewardSelection(
        list(DESTROY_OPERATORS.keys()), ["greedy_cost_tw"],
        segment_size=cfg["segment_size"], reaction_factor=cfg["reaction_factor"],
        decay_factor=cfg["decay_factor"], min_selection_prob=cfg["min_selection_prob"])
    engine = ALNSEngine(inst, cfg, np.random.default_rng(seed), selector=sel,
                        repair_ops={"greedy_cost_tw": greedy_cost_tw_insertion},
                        local_search=rvnd_improve)
    best_sofar = None
    samples = []

    def on_iter(iteration, current_obj, archive):
        nonlocal best_sofar
        m = _feas_min(archive)
        if m is not None:
            best_sofar = m if best_sofar is None else min(best_sofar, m)
        if iteration % SAMPLE_EVERY == 0:
            samples.append([iteration,
                            round(best_sofar, 4) if best_sofar is not None
                            else None])

    engine.register_callback("on_iteration", on_iter)
    t0 = time.time()
    archive = engine.run()
    wall = time.time() - t0

    feas = [(e[0][0], e[1]["solution"]) for e in archive.entries
            if e[1]["solution"].is_feasible()]
    endpoint = min((c for c, _ in feas))
    samples.append([engine.iteration, round(endpoint, 4)])
    # 端点与既存批次断言 (确定性契约)。v3 模式 (MDVRPTW_TRACE_REF_ROOT 置位)
    # 仅对 durfix 结果根下的 population_m5 参照断言; 否则沿用旧多路径逻辑。
    ref_paths = []
    if _REF_ROOT:
        # durfix 结果根下: base → population_ab, popcxge → population_m3
        # (mains s42-44 的归属; 若 m5 根存在则一并断言)
        sub = "population_ab" if arm == "base" else "population_m3"
        for d in ("population_m5", sub):
            p = ROOT / _REF_ROOT / d / f"{name}_s{seed}_{arm}.json"
            if p.exists():
                ref_paths.append(p)
    else:
        for d in ("population_m5",):
            p = ROOT / "results" / d / f"{name}_s{seed}_{arm}.json"
            if p.exists():
                ref_paths.append(p)
        if arm == "base":
            p = ROOT / "results" / "population_ab" / f"{name}_s{seed}_base.json"
            if p.exists():
                ref_paths.append(p)
        else:
            p = ROOT / "results" / "population_m3" / f"{name}_s{seed}_popcxge.json"
            if p.exists():
                ref_paths.append(p)
    ref = None
    for p in ref_paths:
        v = json.loads(p.read_text(encoding="utf-8"))["best_feasible_cost"]
        ref = v if ref is None else ref
        assert abs(endpoint - v) < 1e-3, (
            f"{name} s{seed} {arm}: {endpoint:.4f} != {v} ({p.name})")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}_s{seed}_{arm}_trace.json"
    path.write_text(json.dumps(
        {"instance": name, "seed": seed, "arm": arm, "mode": "6000",
         "samples": samples, "endpoint": round(endpoint, 4),
         "wall_s": round(wall, 1)}, ensure_ascii=False), encoding="utf-8")
    print(f"{name} s{seed} {arm}: endpoint {endpoint:.4f} == ref OK, "
          f"{len(samples)} samples, wall {wall:.0f}s → {path}", flush=True)


if __name__ == "__main__":
    main()
