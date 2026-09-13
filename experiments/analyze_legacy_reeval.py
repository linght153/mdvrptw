"""09-04 遗留臂 3-seed 复核分析 (2026-09-05)。配对差 + s42 敏感性 + 配对 t。

用法: python experiments/analyze_legacy_reeval.py
"""
import json
from pathlib import Path
import math

OUT = Path(__file__).resolve().parent.parent / "results" / "population_ab"

INSTANCES = ["pr15", "pr16", "pr20", "pr06"]
ARMS = ["div", "cx", "admit", "act", "qual"]
SEEDS = [42, 43, 44]


def load(inst, seed, arm):
    f = OUT / f"{inst}_s{seed}_{arm}.json"
    if not f.exists():
        return None
    return json.load(open(f))["best_feasible_cost"]


def paired_ttest(diffs):
    n = len(diffs)
    if n < 2:
        return None
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    t = mean / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    # 双侧 p (t 分布近似, n=3 自由度 2 保守处理: 报告 t 与方向)
    return mean, sd, t


print(f"{'inst':5} {'arm':6} {'per-seed Δpp':28} {'均值':>8} {'剔除s42':>8} {'配对t':>7}")
for inst in INSTANCES:
    base = {s: load(inst, s, "base") for s in SEEDS}
    if any(b is None for b in base.values()):
        print(f"{inst:5} base 数据不全 ({[s for s in SEEDS if base[s] is None]})")
        continue
    for arm in ARMS:
        diffs, d_ex = [], []
        for s in SEEDS:
            a = load(inst, s, arm)
            if a is not None:
                dp = (a - base[s]) / base[s] * 100
                diffs.append(dp)
                if s != 42:
                    d_ex.append(dp)
        if len(diffs) == 3:
            mean = sum(diffs) / 3
            mean_ex = sum(d_ex) / 2
            t = paired_ttest(diffs)
            print(f"{inst:5} {arm:6} {str([round(d, 2) for d in diffs]):28} "
                  f"{mean:>+8.2f} {mean_ex:>+8.2f} {t[2]:>7.2f}")
        elif diffs:
            print(f"{inst:5} {arm:6} 仅 {len(diffs)} seed {[round(d, 2) for d in diffs]}")
