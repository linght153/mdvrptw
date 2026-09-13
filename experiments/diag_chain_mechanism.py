"""成链差异机制诊断 (2026-09-06): 同客户集路由对上的链长对比。

pr15/pr20 3-seed: 两方法路由按 Jaccard≥0.8 匹配 (客户集几乎相同) →
比较其 closed 链长 → 分离"客户集划分" 与 "路由内顺序" 对 chain_save
差的贡献。只读 dump, 无随机性。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

ROOT = Path(__file__).resolve().parent.parent


def load(name, seed):
    ours_p = ROOT / "results" / "base600s_routes" / f"{name}_s{seed}_routes.json"
    if not ours_p.exists():
        ours_p = ROOT / "results" / f"connectivity_{name}_s42_routes.json"
    ours = json.loads(ours_p.read_text(encoding="utf-8"))["routes"]
    pyv = json.loads(
        (ROOT / "results" / "pyvrp_wide" / f"{name}_s{seed}_600s.json")
        .read_text(encoding="utf-8"))["routes"]
    return ours, pyv


def as_seq_list(routes):
    if isinstance(routes, dict):
        return [s for di in sorted(routes) for s in routes[di]]
    return [r["seq"] for r in routes]


def closed_len(inst, depot, seq):
    nd = inst.num_depots
    dm = inst.distance_matrix
    c = dm[depot][seq[0]] + dm[seq[-1]][depot]
    for a, b in zip(seq, seq[1:]):
        c += dm[a][b]
    return c


def main():
    for name in ["pr15", "pr20"]:
        inst = instance_from_parsed(
            parse_mdvrptw(open(ROOT / f"data/mdvrptw_raw/{name}.txt",
                               encoding="utf-8").read()), name=name)
        print(f"\n== {name} ==")
        for seed in (42, 43, 44):
            try:
                ours, pyv = load(name, seed)
            except FileNotFoundError:
                continue
            # ours: {depot: [seq,...]}; pyv: [{depot, seq}]
            ours_r = [(int(di), s) for di, rs in ours.items() for s in rs]
            pyv_r = [(int(r["depot"]), r["seq"]) for r in pyv]
            matched, dO = [], []
            used_o, used_p = set(), set()
            for io, (do, so) in enumerate(ours_r):
                so_set = set(so)
                best_j, best_idx = 0.0, None
                for ip, (dp, sp) in enumerate(pyv_r):
                    inter = len(so_set & set(sp))
                    j = inter / max(len(so_set | set(sp)), 1)
                    if j > best_j and j >= 0.8:
                        best_j, best_idx = j, ip
                if best_idx is not None:
                    dp, sp = pyv_r[best_idx]
                    if do == dp:          # 同车场才可比 (depot 端弧同)
                        used_o.add(io); used_p.add(best_idx)
                        co = closed_len(inst, do, so)
                        cp = closed_len(inst, do, sp)
                        matched.append((io, best_idx, len(so), len(sp),
                                        best_j, co, cp, co - cp))
            if not matched:
                print(f"  s{seed}: no matched route pairs (Jaccard≥0.8 same-depot)")
                continue
            import numpy as np
            d = np.array([m[7] for m in matched])
            cov_o = sum(m[2] for m in matched)
            tot = sum(len(s) for _, s in ours_r)
            # 全解 closed 差
            co_all = sum(closed_len(inst, do, so) for do, so in ours_r)
            cp_all = sum(closed_len(inst, dp, sp) for dp, sp in pyv_r)
            mo = sum(m[6] for m in matched)
            mp = sum(m[5] for m in matched)
            print(f"  s{seed}: matched {len(matched)} pairs covering "
                  f"{cov_o}/{tot} cust (ours);")
            print(f"    Δclosed on matched pairs: ours−pyvrp = "
                  f"{mo - mp:+.2f} (mean {d.mean():+.3f}/route, "
                  f"pos {int((d > 0).sum())}/{len(d)}), lengths "
                  f"ours {[m[2] for m in matched[:6]]}...")
            print(f"    total Δclosed {co_all - cp_all:+.2f} → matched-pair "
                  f"share {(mo - mp) / (co_all - cp_all) * 100:.0f}%")


if __name__ == "__main__":
    main()
