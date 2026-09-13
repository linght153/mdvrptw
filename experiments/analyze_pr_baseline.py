"""汇总 MDVRPTW 引擎基线: results/baseline_pr_120s/*.json → gap vs 现代 BKS 表。

用法: python experiments/analyze_pr_baseline.py
输出: 控制台表 + results/baseline_pr_120s/summary.json
分组: 窄窗 (pr01-10) / 宽窗 (pr11-20) / 规模档。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BKS = json.load(open(Path(__file__).resolve().parent.parent / "data/mdvrptw_bks_2026.json",
                     encoding="utf-8"))
RES = Path(__file__).resolve().parent.parent / "results" / "baseline_pr_120s"


def main():
    rows = []
    for i in range(1, 21):
        name = f"pr{i:02d}"
        costs = []
        ok = True
        for seed in (42, 43, 44):
            p = RES / f"{name}_s{seed}.json"
            if not p.exists():
                ok = False
                continue
            d = json.load(open(p, encoding="utf-8"))
            c = d["best_feasible_cost"]
            if c is None or d["tw_violations"] != 0:
                ok = False
            if c is not None:
                costs.append(c)
        if not costs:
            rows.append((name, None, None, None, None, not ok))
            continue
        arr = np.array(costs)
        ref = BKS[name]["reference"]
        gap = (arr.mean() - ref) / ref * 100
        rows.append((name, arr.mean(), arr.std(), arr.min(), gap, ok))

    print(f"{'实例':<6}{'3seed均值':>10}{'±std':>8}{'min':>10}{'gap%':>8}  可行")
    g_all, g_narrow, g_wide = [], [], []
    for name, mean, std, mn, gap, ok in rows:
        if mean is None:
            print(f"{name:<6}{'—':>10}")
            continue
        print(f"{name:<6}{mean:>10.2f}{std:>8.2f}{mn:>10.2f}{gap:>8.2f}  {ok}")
        g_all.append(gap)
        (g_narrow if name <= "pr10" else g_wide).append(gap)
    if g_all:
        print(f"\n平均 gap: 全 {np.mean(g_all):.2f}% | 窄窗 pr01-10 {np.mean(g_narrow):.2f}% "
              f"| 宽窗 pr11-20 {np.mean(g_wide):.2f}%  (参照: 现代 BKS 2026-05)")
        summ = {"by_instance": {r[0]: {"mean": r[1], "std": r[2], "min": r[3],
                                       "gap_pct": r[4], "feasible": r[5]} for r in rows},
                "mean_gap_all": float(np.mean(g_all)),
                "mean_gap_narrow": float(np.mean(g_narrow)),
                "mean_gap_wide": float(np.mean(g_wide))}
        (RES / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False),
                                          encoding="utf-8")


if __name__ == "__main__":
    main()
