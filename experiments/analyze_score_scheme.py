"""score 解耦探针矩阵分析 (2026-09-05)。用法: python experiments/analyze_score_scheme.py"""
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results" / "population_ab"

REFS = {  # 历史参照 (既有文件, s42)
    "pr15": {"base": 2674.8794, "admit": 2580.0975},
    "pr20": {"base": 3307.9305, "admit": 3185.8899},
    "pr06": {"base": 3601.7952, "admit": 3672.5057},
}

rows = []
for f in sorted(OUT.glob("*.json")):
    name = f.stem
    inst, rest = name.split("_s", 1)
    seed, arm = rest.split("_", 1)
    if arm not in ("base", "admit", "act", "qual", "div", "cx"):
        continue
    d = json.load(open(f))
    rows.append({
        "inst": inst, "seed": int(seed), "arm": arm,
        "best": d["best_feasible_cost"],
        "adds": d["probe_stats"]["adds"] if "probe_stats" in d else None,
        "peak": d["probe_stats"]["stagnation_peak"] if "probe_stats" in d else None,
        "n_imp": len(d.get("improvements", [])),
        "w": d.get("selector_weights"),
    })

# 对比表: 同 (inst, seed) 的臂间差异 (pp vs base)
print(f"{'inst':5} {'seed':4} {'arm':6} {'best':>10} {'ΔvsBase%':>9} {'adds':>6} {'peak':>5} {'n_imp':>5}")
by = {}
for r in rows:
    by.setdefault((r["inst"], r["seed"]), {})[r["arm"]] = r
for (inst, seed), arms in sorted(by.items()):
    b = arms.get("base")
    if b is None:
        continue
    for arm in ("admit", "act", "qual", "div", "cx"):
        a = arms.get(arm)
        if a is None:
            continue
        dp = (a["best"] - b["best"]) / b["best"] * 100
        print(f"{inst:5} {seed:<4} {arm:6} {a['best']:>10.2f} {dp:>+9.2f} "
              f"{str(a['adds']):>6} {str(a['peak']):>5} {a['n_imp']:>5}")

# 每臂改进算子构成 (宽窗主实例 pr15/pr20 汇总)
print("\n改进事件算子构成 (宽窗):")
for arm in ("base", "admit", "act", "qual"):
    cnt = Counter()
    tot = 0
    for r in rows:
        if r["arm"] != arm or r["inst"] not in ("pr15", "pr20"):
            continue
        # improvements 不在 rows 里 — 直接从文件读
        f = OUT / f"{r['inst']}_s{r['seed']}_{arm}.json"
        d = json.load(open(f))
        for i in d.get("improvements", []):
            cnt[i["destroy"]] += 1
            tot += 1
    if tot:
        print(f"  {arm:6}: " + ", ".join(f"{k} {v/tot*100:.0f}%" for k, v in cnt.most_common()))
