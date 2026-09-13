"""方向 A 时间公平对比: base vs prune (120s × 3 seeds) gap 对比。

读 results/baseline_pr_120s/{inst}_s{seed}[_base].json + data/mdvrptw_bks_2026.json。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BKS = json.load(open(Path(__file__).resolve().parent.parent / "data/mdvrptw_bks_2026.json",
                     encoding="utf-8"))
RES = Path(__file__).resolve().parent.parent / "results" / "baseline_pr_120s"


def collect(inst, mode):
    costs, iters = [], []
    for seed in (42, 43, 44):
        sfx = "_base" if mode == "base" else ""
        p = RES / f"{inst}_s{seed}{sfx}.json"
        if not p.exists():
            return None
        d = json.load(open(p, encoding="utf-8"))
        if d["best_feasible_cost"] is None:
            return None
        costs.append(d["best_feasible_cost"])
        iters.append(d["iterations"] or 0)
    return np.array(costs), np.array(iters)


def main():
    insts = ["pr01", "pr05", "pr06", "pr20"]
    print(f"{'实例':<6}{'BKS':>9}{'base均值':>10}{'prune均值':>10}{'Δgap(pp)':>9}"
          f"{'base iter':>10}{'prune iter':>11}")
    for name in insts:
        b = collect(name, "base")
        p = collect(name, "prune")
        if b is None or p is None:
            print(f"{name}: 数据不齐 (base={b is not None}, prune={p is not None})")
            continue
        ref = BKS[name]["reference"]
        gap_b = (b[0].mean() - ref) / ref * 100
        gap_p = (p[0].mean() - ref) / ref * 100
        print(f"{name:<6}{ref:>9.2f}{b[0].mean():>10.2f}{p[0].mean():>10.2f}"
              f"{gap_p - gap_b:>9.2f}{int(b[1].mean()):>10}{int(p[1].mean()):>11}")
        # 配对比较
        if len(b[0]) == len(p[0]):
            from scipy import stats
            t, pv = stats.ttest_rel(b[0], p[0])
            print(f"        配对 t={t:.3f} p={pv:.4f} "
                  f"(base min {b[0].min():.2f} vs prune min {p[0].min():.2f})")


if __name__ == "__main__":
    main()
