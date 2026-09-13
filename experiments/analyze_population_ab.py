"""种群 A/B 汇总 (探针 2026-09-04): 读 results/population_ab/*.json 出判定表。

判定口径 (docs/population-ab-prereg-20260904.md): pr15/16/20 × 3 seeds 均值
gap (参照 BKS); cx−base ≤ −1.5pp → H5 成立; −1.5~−0.5 弱正; >−0.5 否定。
机制有效性: div/cx 档案去重列数 ≥ 2× base。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BKS = {"pr06": 3588.78, "pr15": 2433.15, "pr16": 2836.67, "pr20": 2983.78}
ARMS = ["base", "div", "cx"]


def load_results():
    data = {}
    for path in sorted((ROOT / "results" / "population_ab").glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        key = (d["instance"], d["seed"], d["arm"])
        data[key] = d
    return data


def main():
    data = load_results()
    if not data:
        print("无结果文件 (先跑 scripts/rerun_population_ab.sh)")
        return 1
    print(f"{'实例':5s} {'臂':4s} | {'3seed 均值':>10s} {'±std':>7s} "
          f"{'gap%':>7s} | {'去重列':>5s} {'Jaccard':>8s} {'cx次数':>6s}")
    summary = {}
    for inst in ["pr15", "pr16", "pr20", "pr06"]:
        for arm in ARMS:
            rows = [data[(inst, s, arm)] for s in (42, 43, 44)
                    if (inst, s, arm) in data]
            if not rows:
                continue
            costs = np.array([r["best_feasible_cost"] for r in rows])
            mean = costs.mean()
            gap = (mean / BKS[inst] - 1) * 100
            distinct = np.mean([r["archive_distinct_routes"] for r in rows])
            jac = np.mean([r["archive_mean_jaccard"] for r in rows
                           if r["archive_mean_jaccard"] is not None])
            cx = sum(r["cx_iterations"] for r in rows)
            summary[(inst, arm)] = {"mean": mean, "std": costs.std(), "gap": gap,
                                    "distinct": distinct, "jac": jac, "cx": cx,
                                    "n": len(rows), "costs": costs.tolist()}
            print(f"{inst:5s} {arm:4s} | {mean:10.2f} {costs.std():7.2f} "
                  f"{gap:7.2f} | {distinct:5.0f} {jac:8.3f} {cx:6d}")

    print("\n== 主判定 (pr15/16/20 均值) ==")
    for inst in ["pr15", "pr16", "pr20"]:
        if (inst, "base") not in summary:
            continue
        b = summary[(inst, "base")]["gap"]
        line = f"  {inst}: base {b:+.2f}%"
        for arm in ("div", "cx"):
            if (inst, arm) in summary:
                d = summary[(inst, arm)]["gap"] - b
                line += f" | {arm} {d:+.2f}pp"
        print(line)
    if ("pr15", "base") in summary and ("pr20", "base") in summary:
        g = {a: np.mean([summary[(i, a)]["gap"] for i in ("pr15", "pr16", "pr20")])
             for a in ARMS if (("pr15", a) in summary and ("pr16", a) in summary
                               and ("pr20", a) in summary)}
        if len(g) == 3:
            print(f"  主实例均值 gap: base {g['base']:.2f}% | "
                  f"div {g['div']:.2f}% ({g['div']-g['base']:+.2f}pp) | "
                  f"cx {g['cx']:.2f}% ({g['cx']-g['base']:+.2f}pp)")
    if ("pr06", "base") in summary:
        b6 = summary[("pr06", "base")]["gap"]
        print(f"  pr06 对照: base {b6:.2f}% | "
              f"div {summary[('pr06','div')]['gap']:+.2f}pp | "
              f"cx {summary[('pr06','cx')]['gap']-b6:+.2f}pp")

    print("\n== 机制有效性 (档案多样性, 去重列数) ==")
    for inst in ["pr15", "pr16", "pr20"]:
        if (inst, "base") not in summary:
            continue
        b = summary[(inst, "base")]["distinct"]
        d = summary[(inst, "div")]["distinct"]
        c = summary[(inst, "cx")]["distinct"]
        print(f"  {inst}: base {b:.0f} | div {d:.0f} (×{d/b:.1f}) | "
              f"cx {c:.0f} (×{c/b:.1f})")


if __name__ == "__main__":
    sys.exit(main())
