"""A2 小规模实验分析: a2 κ 扫描 vs 既有 base/div (population_ab 同口径)。

用法: python experiments/analyze_a2_small.py
输入: results/population_ab/{inst}_s42_{base,div,a2_k*}.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(__file__).resolve().parent.parent / "results" / "population_ab"
BKS = {"pr06": 3588.78, "pr15": 2433.15, "pr20": 2983.78}
INSTS = ["pr15", "pr20", "pr06"]
KAPPAS = [0.02, 0.05, 0.10]


def load(inst, seed, tag):
    p = OUT / f"{inst}_s{seed}_{tag}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    rows = []
    for inst in INSTS:
        base = load(inst, 42, "base")
        div = load(inst, 42, "div")
        bks = BKS[inst]
        line = {"inst": inst}
        if base:
            line["base"] = {"cost": base["best_feasible_cost"],
                            "gap": (base["best_feasible_cost"] / bks - 1) * 100,
                            "routes": base["archive_distinct_routes"],
                            "jac": base["archive_mean_jaccard"]}
        if div:
            line["div"] = {"cost": div["best_feasible_cost"],
                           "gap": (div["best_feasible_cost"] / bks - 1) * 100,
                           "routes": div["archive_distinct_routes"],
                           "jac": div["archive_mean_jaccard"]}
        for k in KAPPAS:
            tag = f"a2_k{k}"
            a2 = load(inst, 42, tag)
            if a2:
                base_c = line.get("base", {}).get("cost")
                div_c = line.get("div", {}).get("cost")
                line[tag] = {
                    "cost": a2["best_feasible_cost"],
                    "gap": (a2["best_feasible_cost"] / bks - 1) * 100,
                    "routes": a2["archive_distinct_routes"],
                    "jac": a2["archive_mean_jaccard"],
                    "vs_base_pp": ((a2["best_feasible_cost"] / base_c - 1)
                                   * 100 if base_c else None),
                    "vs_div_pp": ((a2["best_feasible_cost"] / div_c - 1)
                                  * 100 if div_c else None),
                }
        rows.append(line)

    for line in rows:
        inst = line["inst"]
        print(f"\n== {inst} (BKS {BKS[inst]}) ==")
        for tag in ["base", "div"] + [f"a2_k{k}" for k in KAPPAS]:
            m = line.get(tag)
            if not m:
                continue
            extra = ""
            if tag.startswith("a2"):
                extra = (f" | vs base {m['vs_base_pp']:+.3f}pp"
                         f" | vs div {m['vs_div_pp']:+.3f}pp")
            print(f"  {tag:8s}: cost {m['cost']:.2f} (gap {m['gap']:+.3f}%) | "
                  f"列 {m['routes']} | Jaccard {m['jac']}{extra}")

    path = Path(__file__).resolve().parent.parent / "results" / "a2_small_summary.json"
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
