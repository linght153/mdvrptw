"""结构性差异分析 (2026-09-06): pyvrp/HGS 600s 解 vs 自产 base 600s 解。

只读 results/pyvrp_wide/*_600s.json + results/base600s_routes/*_routes.json
(+ 既有 connectivity_{name}_s42_routes.json 回退) + 实例文件, 计算五组
结构指标 (宏观形态/车场分配几何/区带紧凑/路由成本形态/TW 顺序结构) +
共享骨架段。输出 results/structure_diff_summary_20260906.json + 终端表。

预注册: docs/structure-diff-prereg-20260906.md。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

ROOT = Path(__file__).resolve().parent.parent
INSTANCES = ["pr15", "pr16", "pr20"]
SEEDS = [42, 43, 44]


def load_instance(name):
    text = open(ROOT / f"data/mdvrptw_raw/{name}.txt", encoding="utf-8").read()
    return instance_from_parsed(parse_mdvrptw(text), name=name)


def load_ours(name, seed):
    p = ROOT / "results" / "base600s_routes" / f"{name}_s{seed}_routes.json"
    if not p.exists():
        p = ROOT / "results" / f"connectivity_{name}_s42_routes.json"
        if seed != 42 or not p.exists():
            return None
    d = json.loads(p.read_text(encoding="utf-8"))
    routes = {int(k): [list(r) for r in v] for k, v in d["routes"].items()}
    return {"cost": d["best_cost"], "routes": routes, "src": p.name}


def load_pyvrp(name, seed):
    p = ROOT / "results" / "pyvrp_wide" / f"{name}_s{seed}_600s.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    routes = {}
    for r in d["routes"]:
        routes.setdefault(int(r["depot"]), []).append(list(r["seq"]))
    return {"cost": d.get("float_cost") or d["pyvrp_int_cost"],
            "routes": routes, "feasible": d.get("feasible"),
            "src": p.name}


def analyze(inst, routes, int_tw=False):
    """routes: {depot: [[全局节点号...]]} → 结构指标 dict。"""
    nd = inst.num_depots
    dm = inst.distance_matrix
    depots = inst.depots
    cust = inst.customers            # 客户号 0..n-1
    n = inst.num_customers
    cap = inst.max_vehicle_capacity
    G = lambda c: c + nd             # 客户号 → 全局节点号
    demand = np.array([c.demand for c in cust])
    ready = np.array([c.ready_time for c in cust])
    due = np.array([c.due_time for c in cust])
    serv = np.array([c.service_time for c in cust])
    x = np.array([c.x for c in cust])
    y = np.array([c.y for c in cust])
    # 客户到各车场距离 (全局号索引)
    d2d = np.zeros((n, nd))
    for di in range(nd):
        for ci in range(n):
            d2d[ci, di] = dm[di][G(ci)]
    nearest = d2d.argmin(axis=1)
    d_near = d2d.min(axis=1)
    windows = due - ready
    narrow_thr = float(np.percentile(windows, 25))

    all_seqs = []
    len_list, util_list, closed_list, radial_list, purity_list = [], [], [], [], []
    route_depots, inv_pairs = [], []
    slack_last, narrow_pos = [], []
    depot_arcs, intra_arcs = [], []   # 成本分解: 车场端弧 vs 路由内弧
    for di in range(nd):
        veh = depots[di].vehicles_available
        for seq_g in routes.get(di, []):
            seq = [g - nd for g in seq_g]          # → 客户号
            all_seqs.append((di, seq))
            k = len(seq)
            if k == 0:
                continue
            len_list.append(k)
            dm_di = [demand[c] for c in seq]
            util_list.append(sum(dm_di) / (veh and cap or 1))
            rad = [d2d[c, di] for c in seq]
            radial_list.append(sum(rad))
            purity_list.append(sum(1 for c in seq if nearest[c] == di) / k)
            da = dm[di][G(seq[0])] + dm[G(seq[-1])][di]
            closed = da
            intra = 0.0
            for a, b in zip(seq, seq[1:]):
                arc = dm[G(a)][G(b)]
                closed += arc
                intra += arc
            closed_list.append(closed)
            depot_arcs.append(da)
            intra_arcs.append(intra)
            route_depots.append(di)
            # TW 顺序结构 (float 传播; pr16 pyvrp int 解时仅形态参考)
            t = dm[di][G(seq[0])]
            for ci in range(k):
                c = seq[ci]
                if t < ready[c]:
                    t = ready[c]
                t += serv[c]
                if ci + 1 < k:
                    t += dm[G(c)][G(seq[ci + 1])]
            slack_last.append(due[seq[-1]] - (t - serv[seq[-1]]))
            # due 逆序对
            dues = [due[c] for c in seq]
            inv = sum(1 for a in range(k) for b in range(a + 1, k)
                      if dues[a] > dues[b])
            inv_pairs.append(inv / (k * (k - 1) / 2))
            # 窄窗位置
            np_ = [i / max(k - 1, 1) for i, c in enumerate(seq)
                   if windows[c] <= narrow_thr]
            narrow_pos.extend(np_)

    # 车场分配几何 (每客户)
    actual = np.full(n, -1, dtype=int)
    for di, seq in all_seqs:
        for c in seq:
            actual[c] = di
    assert (actual >= 0).all(), "覆盖检查失败"
    rank = np.array([(d2d[c] < d2d[c, actual[c]]).sum() for c in range(n)])
    ratio_an = np.array([d2d[c, actual[c]] / d_near[c] for c in range(n)])
    extra = np.array([d2d[c, actual[c]] - d_near[c] for c in range(n)])
    mm = rank > 0
    depot_load = np.zeros(nd)
    for di, seq in all_seqs:
        depot_load[di] += sum(demand[c] for c in seq)

    return {
        "n_routes": len(len_list),
        "route_len_mean": float(np.mean(len_list)) if len_list else 0,
        "route_len_std": float(np.std(len_list)) if len_list else 0,
        "route_util_mean": float(np.mean(util_list)) if util_list else 0,
        "route_util_std": float(np.std(util_list)) if util_list else 0,
        "depot_arc_mean": float(np.mean(depot_arcs)) if depot_arcs else 0,
        "intra_arc_mean": float(np.mean(intra_arcs)) if intra_arcs else 0,
        # 成本加性分解: 参照 = 每客户独立往返 (2×d_actual); 路由化节省 =
        # 共享弧段收益; 分配惩罚 = 挂非最近车场的额外往返
        "sum_closed": float(np.sum(closed_list)) if closed_list else 0,
        "sum_depot_arcs": float(np.sum(depot_arcs)) if depot_arcs else 0,
        "sum_intra_arcs": float(np.sum(intra_arcs)) if intra_arcs else 0,
        "sum_twoway_actual": float(sum(2.0 * d2d[c, actual[c]]
                                       for c in range(n))),
        "chain_save_frac": float(
            1.0 - np.sum(closed_list) / max(
                sum(2.0 * d2d[c, actual[c]] for c in range(n)), 1e-9))
        if closed_list else 0,
        "sum_mismatch_penalty": float(np.sum(2.0 * extra)),
        "depot_load_frac": [round(float(depot_load[di] /
                              (depots[di].vehicles_available * cap)), 4)
                            for di in range(nd)],
        "routes_per_depot": [len(routes.get(di, [])) for di in range(nd)],
        # 车场分配几何一致性
        "geo_consistent_frac": float((rank == 0).mean()),
        "rank_ge2_frac": float((rank >= 2).mean()),
        "mean_d_ratio_act_near": float(ratio_an.mean()),
        "mean_extra_dist_mismatch": float(extra[mm].mean()) if mm.any() else 0.0,
        "n_mismatch": int(mm.sum()),
        # 区带紧凑性
        "mean_cust_depot_dist": float(np.mean([d2d[c, actual[c]]
                                               for c in range(n)])),
        "mean_radial_sum": float(np.mean(radial_list)) if radial_list else 0,
        "mean_closed_loop": float(np.mean(closed_list)) if closed_list else 0,
        "mean_ratio_closed_radial": float(
            np.mean([c / max(r, 1e-9) for c, r in zip(closed_list, radial_list)]))
        if closed_list else 0,
        "purity_mean": float(np.mean(purity_list)) if purity_list else 0,
        # TW 顺序结构
        "tw_inversion_ratio": float(np.mean(inv_pairs)) if inv_pairs else 0,
        "mean_slack_last": float(np.mean(slack_last)) if slack_last else 0,
        "narrow_pos_mean": float(np.mean(narrow_pos)) if narrow_pos else np.nan,
        "narrow_frac": float((windows <= narrow_thr).mean()),
    }


def shared_segments(routes_a, routes_b, min_len=4):
    """两解间共享最长连续路由段数 (允许反向)。返回 dict: {min_len: (对数, 覆盖客户数)}。"""
    seqs_a = [(s, set(s)) for di in sorted(routes_a) for s in routes_a[di]]
    seqs_b = [(s, set(s)) for di in sorted(routes_b) for s in routes_b[di]]

    def lcs_sub(x, y):
        best = 0
        for i in range(len(x)):
            for j in range(len(y)):
                k = 0
                while i + k < len(x) and j + k < len(y) and x[i + k] == y[j + k]:
                    k += 1
                best = max(best, k)
        return best

    out = {ml: (0, set()) for ml in (3, 4, 5, 6)}
    for a, sa in seqs_a:
        for b, sb in seqs_b:
            m = lcs_sub(a, b)
            mb = lcs_sub(a, tuple(reversed(b)))
            L = max(m, mb)
            for ml in out:
                if L >= ml:
                    out[ml] = (out[ml][0] + 1, out[ml][1] | (sa & sb))
    return {ml: (v[0], len(v[1])) for ml, v in out.items()}


def main():
    summary = {}
    for name in INSTANCES:
        inst = load_instance(name)
        inst_row = {}
        for seed in SEEDS:
            ours = load_ours(name, seed)
            pyv = load_pyvrp(name, seed)
            if ours is None or pyv is None:
                print(f"[skip] {name} s{seed}: ours={ours is not None} "
                      f"pyvrp={pyv is not None}")
                continue
            s_ours = analyze(inst, ours["routes"])
            s_pyv = analyze(inst, pyv["routes"])
            inst_row[f"s{seed}"] = {
                "ours_cost": ours["cost"], "pyvrp_cost": pyv["cost"],
                "ours": s_ours, "pyvrp": s_pyv,
                "pyvrp_feasible": pyv.get("feasible"),
                "shared_long_pairs": shared_segments(ours["routes"],
                                                     pyv["routes"]),
            }
        summary[name] = inst_row

    out = ROOT / "results" / "structure_diff_summary_20260906.json"
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                   encoding="utf-8")

    KEYS = ["n_routes", "route_len_mean", "route_len_std", "route_util_mean",
            "route_util_std", "depot_arc_mean", "intra_arc_mean",
            "geo_consistent_frac", "rank_ge2_frac",
            "mean_d_ratio_act_near", "mean_extra_dist_mismatch", "purity_mean",
            "mean_closed_loop", "mean_ratio_closed_radial",
            "tw_inversion_ratio", "mean_slack_last", "narrow_pos_mean"]
    for name in INSTANCES:
        row = summary.get(name, {})
        if not row:
            continue
        print(f"\n== {name} ==")
        print(f"{'metric':32s} {'seed':>5s} | {'ours':>10s} {'pyvrp':>10s} "
              f"{'Δ(o-p)':>10s}")
        for k in KEYS:
            vals = []
            for sd in ("s42", "s43", "s44"):
                if sd in row:
                    vals.append((row[sd]["ours"][k], row[sd]["pyvrp"][k]))
            if not vals:
                continue
            o = np.mean([v[0] for v in vals])
            p = np.mean([v[1] for v in vals])
            signs = {np.sign(v[0] - v[1]) for v in vals if v[0] != v[1]}
            mark = " *" if len(signs) == 1 and abs(o - p) > 1e-9 else ""
            print(f"{k:32s} {'mean':>5s} | {o:10.4f} {p:10.4f} "
                  f"{o - p:10.4f}{mark}")
        # 车场负载与共享段 (每 seed)
        for sd in ("s42", "s43", "s44"):
            if sd in row:
                r = row[sd]
                o, p = r["ours"], r["pyvrp"]
                print(f"== {name} {sd}: ours {o['sum_closed']:.2f} vs pyvrp "
                      f"{p['sum_closed']:.2f} (Δ {o['sum_closed']-p['sum_closed']:+.2f}) ==")
                print(f"  cost decomp  ours: closed={o['sum_closed']:.1f} = "
                      f"depot {o['sum_depot_arcs']:.1f} + intra {o['sum_intra_arcs']:.1f} | "
                      f"twoway={o['sum_twoway_actual']:.1f} chain_save={o['chain_save_frac']:.4f} "
                      f"mismatch_pen={o['sum_mismatch_penalty']:.1f}")
                print(f"               pyvrp: closed={p['sum_closed']:.1f} = "
                      f"depot {p['sum_depot_arcs']:.1f} + intra {p['sum_intra_arcs']:.1f} | "
                      f"twoway={p['sum_twoway_actual']:.1f} chain_save={p['chain_save_frac']:.4f} "
                      f"mismatch_pen={p['sum_mismatch_penalty']:.1f}")
                print(f"{'depot_load_frac ' + sd:32s} {'':>5s} | "
                      f"{str([round(v,3) for v in o['depot_load_frac']]):>10s} "
                      f"{str([round(v,3) for v in p['depot_load_frac']]):>10s}")
                print(f"{'n_mismatch ' + sd:32s} {'':>5s} | "
                      f"{o['n_mismatch']:>10d} {p['n_mismatch']:>10d}")
                sh = r['shared_long_pairs']
                print(f"{'shared_seg ≥3/4/5/6 ' + sd:30s} {'':>5s} | "
                      f"{str({k: v for k, v in sh.items()}):>45s}")


if __name__ == "__main__":
    main()
