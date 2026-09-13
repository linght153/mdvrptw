"""P0 探针: MDVRPTW 时间可达弧剪枝潜力 + TW 白试率测量 (2026-09-02)。

测量(行为与引擎完全一致, 仅插桩计数):
1. repair (_insert_cheapest_cost_tw): 每个客户的位置枚举中,
   - 新弧 (prev→ci)/(ci→next) 硬不可达占比 (= 无损弧剪枝潜力下界)
   - TW 整路由检查拒绝率 (白试)
2. LS 四邻域 (two_opt/relocate/swap/2-opt*): 移动候选枚举中弧硬不可达占比
   + 成本改进候选的 TW 检查拒绝率
3. 对照: 窄窗 (pr01/pr07) vs 宽窗 (pr11) — 验证剪枝率 = 窗宽的确定函数

用法: python experiments/probe_tw_candidates.py [pr01|pr07|pr11] [iters] [seed]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.core.instance import instance_from_parsed
from src.core.objectives import calculate_objectives
from src.alns.selection import SegmentRewardSelection
from src.alns.engine import ALNSEngine
from src.operators.local_search import (
    _move_tw_ok, _route_cost, _route_load, _demand, _two_opt_delta,
)
from src.operators.repair import _route_tw_excess
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

C = {}   # 计数器
REACH = None  # 客户对硬可达矩阵 (客户 index 空间)
BASE = 0


def reach(a, b) -> bool:
    return REACH[a - BASE][b - BASE]


def reset(inst):
    global REACH, BASE
    C.clear()
    keys = ["rep_pos", "rep_arc", "rep_tw_rej", "rep_newroute",
            "to_pair", "to_arc", "to_tw_check", "to_tw_rej",
            "rel_pos", "rel_arc", "rel_tw_check", "rel_tw_rej",
            "sw_pair", "sw_arc", "sw_tw_check", "sw_tw_rej",
            "os_pair", "os_arc", "os_tw_check", "os_tw_rej"]
    C.update({k: 0 for k in keys})
    n = inst.num_customers
    base = inst.num_depots
    BASE = base
    cs = inst.customers
    R = np.ones((n, n), dtype=bool)
    for a in cs:
        for b in cs:
            if a.index == b.index:
                continue
            # 硬可达: 最早到 b = e_a + s_a + t_ab <= l_b (等待不可救)
            if (a.ready_time + a.service_time
                    + inst.distance_matrix[a.index][b.index] > b.due_time):
                R[a.index - base][b.index - base] = False
    REACH = R


# ── repair 插桩版 (复制 _insert_cheapest_cost_tw, 只加计数) ──────────
def _probe_insert_tw(solution, ci):
    inst = solution.instance
    c = inst.customers[ci - inst.num_depots]

    best_cost = float("inf")
    best = None
    for di, rlist in solution.routes.items():
        for ri, route in enumerate(rlist):
            route_demand = sum(
                inst.customers[x - inst.num_depots].demand for x in route)
            if route_demand + c.demand > inst.max_vehicle_capacity:
                continue
            for pos in range(len(route) + 1):
                C["rep_pos"] += 1
                prev = route[pos - 1] if pos > 0 else None
                nxt = route[pos] if pos < len(route) else None
                if (prev is not None and not reach(prev, ci)) or \
                   (nxt is not None and not reach(ci, nxt)):
                    C["rep_arc"] += 1
                new_route = list(route)
                new_route.insert(pos, ci)
                if _route_tw_excess(inst, di, new_route, ri) > 1e-9:
                    C["rep_tw_rej"] += 1
                    continue
                delta = _tw_delta_probe(solution, di, route, pos, ci, ri)
                if delta < best_cost:
                    best_cost = delta
                    best = (di, ri, pos)
    if best is not None:
        di, ri, pos = best
        solution.routes[di][ri].insert(pos, ci)
        c.assigned_depot_index = di
    else:
        C["rep_newroute"] += 1
        from src.operators.repair import _nearest_depot_with_capacity
        di = _nearest_depot_with_capacity(solution, ci)
        solution.routes[di].append([ci])
        c.assigned_depot_index = di


def _tw_delta_probe(solution, depot_idx, route, pos, ci, route_index=0):
    """与 repair._tw_insertion_delta 同逻辑 (TW 权重回退)。"""
    inst = solution.instance
    weight = inst.tw_penalty_weight
    if not inst.has_time_windows:
        return _route_cost(inst, depot_idx, route[:pos] + [ci] + route[pos:]) \
            - _route_cost(inst, depot_idx, route)
    if weight <= 0.0:
        weight = 1000.0
    dist_delta = (_route_cost(inst, depot_idx, route[:pos] + [ci] + route[pos:])
                  - _route_cost(inst, depot_idx, route))
    new_route = list(route)
    new_route.insert(pos, ci)
    old_excess = _route_tw_excess(inst, depot_idx, route, route_index)
    new_excess = _route_tw_excess(inst, depot_idx, new_route, route_index)
    return dist_delta + weight * (new_excess - old_excess)


def probe_greedy_cost_tw(partial, removed, rng):
    sol = partial.copy()
    for ci in removed:
        _probe_insert_tw(sol, ci)
    return sol


# ── LS 插桩版 (复制 4 邻域 + RVND, 只加计数) ────────────────────────
def probe_two_opt(solution, max_iterations=50, rng=None):
    if max_iterations <= 0:
        return solution
    sol = solution.copy()
    inst = sol.instance
    for _ in range(max_iterations):
        improved = False
        for depot_idx, route_list in sol.routes.items():
            depot_node = inst.depots[depot_idx]
            for ri, route in enumerate(route_list):
                if len(route) < 2:
                    continue
                best_i, best_j, best_delta = -1, -1, 0.0
                for i in range(len(route) - 1):
                    for j in range(i + 1, min(i + 15, len(route))):
                        C["to_pair"] += 1
                        # 反转新弧: (route[i-1]→route[j]) 与 (route[i]→route[j+1])
                        bad = False
                        if i > 0 and not reach(route[i - 1], route[j]):
                            bad = True
                        if j < len(route) - 1 and not reach(route[i], route[j + 1]):
                            bad = True
                        if bad:
                            C["to_arc"] += 1
                        delta = _two_opt_delta(inst, depot_node, route, i, j)
                        if delta < best_delta:
                            C["to_tw_check"] += 1
                            rev = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
                            if _move_tw_ok(inst, depot_idx, rev, ri):
                                best_delta, best_i, best_j = delta, i, j
                            else:
                                C["to_tw_rej"] += 1
                if best_i >= 0 and best_delta < -1e-6:
                    route[best_i:best_j + 1] = reversed(route[best_i:best_j + 1])
                    improved = True
        if not improved:
            break
    return sol


def probe_relocate(solution, max_iterations=50, rng=None):
    if max_iterations <= 0:
        return solution
    sol = solution.copy()
    inst = sol.instance
    for _ in range(max_iterations):
        improved = False
        route_loads = {(di, ri): _route_load(inst, r)
                       for di, rl in sol.routes.items()
                       for ri, r in enumerate(rl)}
        for from_di, from_routes in list(sol.routes.items()):
            for ri, route in list(enumerate(from_routes)):
                if len(route) < 1:
                    continue
                for pos, ci in enumerate(route):
                    base_removed = _route_cost(inst, from_di, route)
                    new_from = route[:pos] + route[pos + 1:]
                    delta_from = _route_cost(inst, from_di, new_from) - base_removed
                    best_delta, best_tgt = 0.0, None
                    for to_di, to_routes in sol.routes.items():
                        for rj, t_route in enumerate(to_routes):
                            if to_di == from_di and rj == ri:
                                continue
                            if (route_loads[(to_di, rj)] + _demand(inst, ci)
                                    > inst.max_vehicle_capacity + 1e-9):
                                continue
                            for ip in range(len(t_route) + 1):
                                C["rel_pos"] += 1
                                prev = t_route[ip - 1] if ip > 0 else None
                                nxt = t_route[ip] if ip < len(t_route) else None
                                if (prev is not None and not reach(prev, ci)) or \
                                   (nxt is not None and not reach(ci, nxt)):
                                    C["rel_arc"] += 1
                                new_to = t_route[:ip] + [ci] + t_route[ip:]
                                delta_to = (_route_cost(inst, to_di, new_to)
                                            - _route_cost(inst, to_di, t_route))
                                d = delta_from + delta_to
                                if d < best_delta - 1e-9:
                                    C["rel_tw_check"] += 1
                                    if _move_tw_ok(inst, to_di, new_to, rj):
                                        best_delta, best_tgt = d, (to_di, rj, ip)
                                    else:
                                        C["rel_tw_rej"] += 1
                    if best_tgt is not None:
                        to_di, rj, ip = best_tgt
                        sol.routes[to_di][rj].insert(ip, ci)
                        if new_from:
                            sol.routes[from_di][ri] = new_from
                        else:
                            del sol.routes[from_di][ri]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return sol


def probe_swap(solution, max_iterations=50, rng=None):
    if max_iterations <= 0:
        return solution
    sol = solution.copy()
    inst = sol.instance
    for _ in range(max_iterations):
        improved = False
        for di_a, routes_a in list(sol.routes.items()):
            for ri_a, route_a in list(enumerate(routes_a)):
                load_a = _route_load(inst, route_a)
                for pi_a, ci_a in enumerate(route_a):
                    d_a = _demand(inst, ci_a)
                    for di_b, routes_b in list(sol.routes.items()):
                        for ri_b, route_b in list(enumerate(routes_b)):
                            if (di_a, ri_a) == (di_b, ri_b):
                                continue
                            load_b = _route_load(inst, route_b)
                            for pi_b, ci_b in enumerate(route_b):
                                d_b = _demand(inst, ci_b)
                                if (load_a - d_a + d_b > inst.max_vehicle_capacity + 1e-9
                                        or load_b - d_b + d_a > inst.max_vehicle_capacity + 1e-9):
                                    continue
                                C["sw_pair"] += 1
                                ra2 = route_a[:]
                                rb2 = route_b[:]
                                ra2[pi_a], rb2[pi_b] = ci_b, ci_a
                                bad = False
                                for arr, p, moved in ((ra2, pi_a, ci_b), (rb2, pi_b, ci_a)):
                                    if p > 0 and not reach(arr[p - 1], moved):
                                        bad = True
                                    if p < len(arr) - 1 and not reach(moved, arr[p + 1]):
                                        bad = True
                                if bad:
                                    C["sw_arc"] += 1
                                d = (_route_cost(inst, di_a, ra2) - _route_cost(inst, di_a, route_a)
                                     + _route_cost(inst, di_b, rb2) - _route_cost(inst, di_b, route_b))
                                if d < -1e-9:
                                    C["sw_tw_check"] += 1
                                    if (_move_tw_ok(inst, di_a, ra2, ri_a)
                                            and _move_tw_ok(inst, di_b, rb2, ri_b)):
                                        sol.routes[di_a][ri_a] = ra2
                                        sol.routes[di_b][ri_b] = rb2
                                        improved = True
                                        break
                                    else:
                                        C["sw_tw_rej"] += 1
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return sol


def probe_two_opt_star(solution, max_iterations=50, rng=None):
    if max_iterations <= 0:
        return solution
    sol = solution.copy()
    inst = sol.instance
    for _ in range(max_iterations):
        improved = False
        for di_a, routes_a in list(sol.routes.items()):
            for ri_a, route_a in list(enumerate(routes_a)):
                load_a = _route_load(inst, route_a)
                cap = inst.max_vehicle_capacity
                for di_b, routes_b in list(sol.routes.items()):
                    for ri_b, route_b in list(enumerate(routes_b)):
                        if (di_a, ri_a) == (di_b, ri_b):
                            continue
                        load_b = _route_load(inst, route_b)
                        prefix_a = 0.0
                        for cut_a in range(1, len(route_a)):
                            prefix_a += _demand(inst, route_a[cut_a - 1])
                            tail_a = load_a - prefix_a
                            prefix_b = 0.0
                            for cut_b in range(1, len(route_b)):
                                prefix_b += _demand(inst, route_b[cut_b - 1])
                                tail_b = load_b - prefix_b
                                if (prefix_a + tail_b > cap + 1e-9
                                        or prefix_b + tail_a > cap + 1e-9):
                                    continue
                                C["os_pair"] += 1
                                # 接合新弧: a[cut_a-1]→b[cut_b], b[cut_b-1]→a[cut_a]
                                if (not reach(route_a[cut_a - 1], route_b[cut_b])
                                        or not reach(route_b[cut_b - 1], route_a[cut_a])):
                                    C["os_arc"] += 1
                                ra2 = route_a[:cut_a] + route_b[cut_b:]
                                rb2 = route_b[:cut_b] + route_a[cut_a:]
                                d = (_route_cost(inst, di_a, ra2) - _route_cost(inst, di_a, route_a)
                                     + _route_cost(inst, di_b, rb2) - _route_cost(inst, di_b, route_b))
                                if d < -1e-9:
                                    C["os_tw_check"] += 1
                                    if (_move_tw_ok(inst, di_a, ra2, ri_a)
                                            and _move_tw_ok(inst, di_b, rb2, ri_b)):
                                        sol.routes[di_a][ri_a] = ra2
                                        sol.routes[di_b][ri_b] = rb2
                                        improved = True
                                        break
                                    else:
                                        C["os_tw_rej"] += 1
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return sol


def probe_rvnd(solution, max_iterations=50, rng=None):
    """与 rvnd_improve 相同, 但邻域集 = 插桩版。"""
    if max_iterations <= 0:
        return solution
    if rng is None:
        rng = np.random.default_rng()
    sol = solution.copy()
    neighborhoods = [probe_two_opt, probe_relocate, probe_swap,
                     probe_two_opt_star]
    budget = max_iterations
    while budget > 0:
        order = rng.permutation(len(neighborhoods))
        improved_any = False
        for idx in order:
            before_cost = calculate_objectives(sol)[0]
            candidate = neighborhoods[idx](sol, max_iterations=1)
            cand_cost = calculate_objectives(candidate)[0]
            if cand_cost < before_cost - 1e-9 and candidate.is_feasible():
                sol = candidate
                improved_any = True
                budget -= 1
                break
        if not improved_any:
            break
    return sol


def run(inst_name, iters, seed):
    text = open(Path(__file__).resolve().parent.parent / f"data/mdvrptw_raw/{inst_name}.txt",
                encoding="utf-8").read()
    inst = instance_from_parsed(parse_mdvrptw(text), name=inst_name)
    from dataclasses import replace as dc_replace
    inst = dc_replace(inst, customers=[dc_replace(c) for c in inst.customers])
    reset(inst)

    config = {"max_iterations": iters, "max_time_seconds": 1e9,
              "stagnation_limit": 300, "segment_size": 100,
              "reaction_factor": 0.1, "decay_factor": 1.0,
              "min_selection_prob": 0.005, "initial_temperature": None,
              "cooling_rate": None, "archive_capacity": 20,
              "parent_selection": "crowding", "initial_solution": "nearest",
              "k_min_ratio": 0.10, "k_max_ratio": 0.40, "sigma": 3.0,
              "local_search_max_iter": 10, "local_search_freq": 5,
              "local_search_on_accept": True, "local_search_final": False}
    rng = np.random.default_rng(seed)
    destroy_names = list(__import__("src.operators.destroy",
                                    fromlist=["DESTROY_OPERATORS"]).DESTROY_OPERATORS.keys())
    repair_names = ["greedy_cost_tw"]
    selector = SegmentRewardSelection(
        destroy_names, repair_names, segment_size=100, reaction_factor=0.1,
        decay_factor=1.0, min_selection_prob=0.005)
    engine = ALNSEngine(instance=inst, config=config, rng=rng,
                        selector=selector,
                        repair_ops={"greedy_cost_tw": probe_greedy_cost_tw},
                        local_search=probe_rvnd)
    import time
    t0 = time.time()
    archive = engine.run()
    wall = time.time() - t0
    best = min(e[0][0] for e in archive.entries) if archive.entries else float("inf")
    feasible = all(e[1]["solution"].is_feasible() for e in archive.entries)
    return inst, wall, best, feasible


def report(inst_name, inst, wall, best, feasible):
    def pct(a, b):
        return f"{a / b * 100:6.1f}%" if b else "   n/a"
    print(f"\n===== {inst_name}: n={inst.num_customers} m={inst.num_depots} "
          f"窗宽中位={int(np.median([c.due_time - c.ready_time for c in inst.customers]))} "
          f"wall={wall:.1f}s best={best:.2f} feasible={feasible}")
    rp, ra, rt = C["rep_pos"], C["rep_arc"], C["rep_tw_rej"]
    print(f"[repair] 位置枚举 {rp}: 弧硬不可达(无损可剪) {ra} {pct(ra, rp)}; "
          f"TW整路由拒绝 {rt} {pct(rt, rp)}; 开新车 {C['rep_newroute']}")
    print(f"         → 无损剪枝已覆盖 TW 白试的 {pct(min(ra, rt), rt) if rt else 'n/a'}")
    for name, pk, ak, ck, rk in [
            ("two_opt ", "to_pair", "to_arc", "to_tw_check", "to_tw_rej"),
            ("relocate", "rel_pos", "rel_arc", "rel_tw_check", "rel_tw_rej"),
            ("swap    ", "sw_pair", "sw_arc", "sw_tw_check", "sw_tw_rej"),
            ("2-opt*  ", "os_pair", "os_arc", "os_tw_check", "os_tw_rej")]:
        p, a, ck_, r = C[pk], C[ak], C[ck], C[rk]
        if not p:
            continue
        print(f"[LS {name}] 候选 {p}: 弧硬不可达 {a} {pct(a, p)}; "
              f"成本改进后查TW {ck_} 次, 被拒(白试) {r} {pct(r, ck_) if ck_ else 'n/a'}")


def main():
    names = sys.argv[1:] or ["pr01"]
    for inst_name in names:
        iters = 300 if len(sys.argv) > len(names) + 1 else 300
        inst, wall, best, feasible = run(inst_name, iters, 42)
        report(inst_name, inst, wall, best, feasible)


if __name__ == "__main__":
    main()
