"""多邻域局部搜索算子: 2-opt (路由内) + relocate/swap/2-opt* (跨路由)
+ RVND 组合。LS 是 VRP 元启发式的核心引擎。"""

import numpy as np

from src.core.solution import Solution
from src.core.objectives import calculate_objectives


def two_opt_improve(solution: Solution, max_iterations: int = 50,
                    rng: np.random.Generator | None = None) -> Solution:
    """对所有路由应用 2-opt 路由内改进。

    反转路由段以降低总成本。
    仅接受严格改进综合目标 + TW 可行的移动 (TW-aware)。
    rng: 兼容 engine 统一签名 (two_opt 确定性, 不使用)。

    当 max_iterations <= 0 时直接返回原解（短路，避免拷贝开销）。
    """
    if max_iterations <= 0:
        return solution
    sol = solution.copy()
    inst = sol.instance
    best_cost, best_emit = calculate_objectives(sol)
    best_combined = best_cost + best_emit

    for _ in range(max_iterations):
        improved = False
        for depot_idx, route_list in sol.routes.items():
            depot_node = inst.depots[depot_idx]
            for ri, route in enumerate(route_list):
                if len(route) < 2:
                    continue
                best_i, best_j = -1, -1
                best_delta = 0.0

                for i in range(len(route) - 1):
                    for j in range(i + 1, min(i + 15, len(route))):
                        # 计算反转 route[i..j] 的成本增量
                        delta = _two_opt_delta(inst, depot_node, route, i, j)
                        if delta < best_delta:
                            # TW-aware: 反转后路由必须 TW 可行
                            rev = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
                            if _move_tw_ok(inst, depot_idx, rev, ri):
                                best_delta = delta
                                best_i, best_j = i, j

                if best_i >= 0 and best_delta < -1e-6:
                    route[best_i:best_j + 1] = reversed(route[best_i:best_j + 1])
                    improved = True

        if not improved:
            break

    return sol


def _two_opt_delta(inst, depot_node, route: list[int], i: int, j: int) -> float:
    """计算反转 route[i..j] 的成本增量。负值 = 改进。"""
    dm = inst.distance_matrix
    n = len(route)

    if i == 0 and j == n - 1:
        return 0.0  # 完全反转，无向 TSP 无变化

    # 移除的边
    if i == 0:
        removed = dm[depot_node.index][route[0]]
    else:
        removed = dm[route[i - 1]][route[i]]

    removed += dm[route[j]][route[(j + 1) % n]] if j < n - 1 else dm[route[j]][depot_node.index]

    # 新增的边（反转后端点连接方式不变）
    if i == 0:
        added = dm[depot_node.index][route[j]]
    else:
        added = dm[route[i - 1]][route[j]]

    added += dm[route[i]][route[j + 1]] if j < n - 1 else dm[route[i]][depot_node.index]

    return (added - removed) * inst.emission_model.unit_cost_per_km


# ─────────────────────────────────────────────────────────────
# 多邻域局部搜索 (RVND 风格) — LS 是 VRP 元启发式的核心引擎
# (Vidal 2013 HGSADC 强在多邻域 LS; two_opt 仅路由内, 无法跨路由
#  改进)。以下算子全部 TW-aware: 只接受"目标不劣化 + TW 不违规"的移动。
# ─────────────────────────────────────────────────────────────

def _route_cost(inst, depot_idx: int, route: list[int]) -> float:
    """路由总距离成本 (含车场往返)。"""
    if not route:
        return 0.0
    dm = inst.distance_matrix
    depot_node = inst.depots[depot_idx].index
    total = dm[depot_node][route[0]] + dm[route[-1]][depot_node]
    for a, b in zip(route, route[1:]):
        total += dm[a][b]
    return total * inst.emission_model.unit_cost_per_km


def _arc_delta_insert(inst, depot_idx: int, route: list[int], pos: int,
                      new_cust: int) -> float:
    """在 route 的 pos 处插入 new_cust 的距离增量 (O(1) 弧差; 含车场边界)。

    与 _route_cost(new) - _route_cost(old) 代数恒等 (浮点末位可差, 阈值
    1e-9 余量 ≥1e3 倍)。LS 候选热路径替代全路由重算 (曾 425 万次)。
    """
    dm = inst.distance_matrix
    depot_node = inst.depots[depot_idx].index
    if not route:
        new_d = dm[depot_node][new_cust] + dm[new_cust][depot_node]
        return new_d * inst.emission_model.unit_cost_per_km
    if pos == 0:
        old_d = dm[depot_node][route[0]]
        new_d = dm[depot_node][new_cust] + dm[new_cust][route[0]]
    elif pos == len(route):
        old_d = dm[route[-1]][depot_node]
        new_d = dm[route[-1]][new_cust] + dm[new_cust][depot_node]
    else:
        old_d = dm[route[pos - 1]][route[pos]]
        new_d = dm[route[pos - 1]][new_cust] + dm[new_cust][route[pos]]
    return (new_d - old_d) * inst.emission_model.unit_cost_per_km


def _arc_delta_remove(inst, depot_idx: int, route: list[int],
                      pos: int) -> float:
    """从 route 移除 pos 处客户的距离增量 (O(1) 弧差; 路由变空含车场往返)。"""
    dm = inst.distance_matrix
    depot_node = inst.depots[depot_idx].index
    if len(route) == 1:
        new_d = 0.0
        old_d = dm[depot_node][route[0]] + dm[route[0]][depot_node]
    elif pos == 0:
        old_d = dm[depot_node][route[0]] + dm[route[0]][route[1]]
        new_d = dm[depot_node][route[1]]
    elif pos == len(route) - 1:
        old_d = dm[route[pos - 1]][route[pos]] + dm[route[pos]][depot_node]
        new_d = dm[route[pos - 1]][depot_node]
    else:
        old_d = dm[route[pos - 1]][route[pos]] + dm[route[pos]][route[pos + 1]]
        new_d = dm[route[pos - 1]][route[pos + 1]]
    return (new_d - old_d) * inst.emission_model.unit_cost_per_km


def _arc_delta_replace(inst, depot_idx: int, route: list[int], pos: int,
                       new_cust: int) -> float:
    """把 route[pos] 替换为 new_cust 的距离增量 (O(1) 弧差; 含车场边界)。"""
    dm = inst.distance_matrix
    depot_node = inst.depots[depot_idx].index
    old_cust = route[pos]
    if len(route) == 1:
        old_d = dm[depot_node][old_cust] + dm[old_cust][depot_node]
        new_d = dm[depot_node][new_cust] + dm[new_cust][depot_node]
    elif pos == 0:
        old_d = dm[depot_node][old_cust] + dm[old_cust][route[1]]
        new_d = dm[depot_node][new_cust] + dm[new_cust][route[1]]
    elif pos == len(route) - 1:
        old_d = dm[route[pos - 1]][old_cust] + dm[old_cust][depot_node]
        new_d = dm[route[pos - 1]][new_cust] + dm[new_cust][depot_node]
    else:
        old_d = dm[route[pos - 1]][old_cust] + dm[old_cust][route[pos + 1]]
        new_d = dm[route[pos - 1]][new_cust] + dm[new_cust][route[pos + 1]]
    return (new_d - old_d) * inst.emission_model.unit_cost_per_km


def _arc_delta_stitch(inst, depot_a: int, depot_b: int,
                      route_a: list[int], cut_a: int,
                      route_b: list[int], cut_b: int) -> float:
    """2-opt* 重连 (A[:cut_a]+B[cut_b:], B[:cut_b]+A[cut_a:]) 距离增量 O(1)。

    变化弧 8 条: 两条被剪弧 (cut-1→cut) + 两条接合弧 + 四条回程弧
    (尾段换车场后 B[-1]→depot_a、A[-1]→depot_b; 同车场时成对抵消)。
    """
    dm = inst.distance_matrix
    u = inst.emission_model.unit_cost_per_km
    na = inst.depots[depot_a].index
    nb = inst.depots[depot_b].index
    last_a, last_b = route_a[-1], route_b[-1]
    delta = (dm[route_a[cut_a - 1]][route_b[cut_b]]
             + dm[route_b[cut_b - 1]][route_a[cut_a]]
             + dm[last_b][na] + dm[last_a][nb]
             - dm[route_a[cut_a - 1]][route_a[cut_a]]
             - dm[route_b[cut_b - 1]][route_b[cut_b]]
             - dm[last_a][na] - dm[last_b][nb])
    return delta * u


def _move_dur_ok(inst, depot_idx: int, route: list[int],
                 route_index: int = 0) -> bool:
    """检查路由时长 ≤ D (无 D 恒 True; 单一事实源 = solution.route_return_duration)。"""
    if inst.route_duration_limit is None:
        return True
    from src.core.solution import route_return_duration

    return route_return_duration(inst, depot_idx, route,
                                 route_index)[1] <= inst.route_duration_limit + 1e-9


def _move_tw_ok(inst, depot_idx: int, route: list[int],
                route_index: int = 0) -> bool:
    """检查路由 TW + 时长双重可行性 (增量, 不构造 Solution; 分车起始时刻感知)。"""
    if inst.has_time_windows:
        from src.operators.repair import _route_tw_excess

        if _route_tw_excess(inst, depot_idx, route, route_index) > 1e-9:
            return False
    return _move_dur_ok(inst, depot_idx, route, route_index)


def _route_load(inst, route: list[int]) -> float:
    """路由总需求 (跨路由移动的容量可行性检查)。"""
    return sum(inst.customers[i - inst.num_depots].demand for i in route)


def _demand(inst, customer_idx: int) -> float:
    """单个客户的需求量。"""
    return inst.customers[customer_idx - inst.num_depots].demand


def relocate_improve(solution: Solution, max_iterations: int = 50,
                     rng: np.random.Generator | None = None) -> Solution:
    """跨路由 relocate: 将一个客户从路由 A 移到路由 B 的任一位置。

    接受标准: 综合目标 (成本) 严格改进 + 两条路由 TW 均可行。
    rng: 兼容 engine 统一签名。
    """
    if max_iterations <= 0:
        return solution
    sol = solution.copy()
    inst = sol.instance
    dm = inst.distance_matrix
    R = inst.reach_matrix   # 方向 A: 局部可达矩阵 (None = 不剪枝)
    nd = inst.num_depots
    tw = inst.has_time_windows
    from src.operators.repair import _insert_incremental_excess, _route_timeline

    for _ in range(max_iterations):
        improved = False
        # 缓存所有路由 load (单次 full pass 内路由不变, 直到 relocate 应用 break)
        route_loads = {
            (di, ri): _route_load(inst, r)
            for di, rl in sol.routes.items()
            for ri, r in enumerate(rl)
        }
        for from_di, from_routes in list(sol.routes.items()):
            for ri, route in list(enumerate(from_routes)):
                if len(route) < 1:
                    continue
                for pos, ci in enumerate(route):
                    # 计算移除 ci 的增量 (O(1) 弧差)
                    delta_from = _arc_delta_remove(inst, from_di, route, pos)
                    new_from = route[:pos] + route[pos + 1:]
                    # 时长: from 侧移除后必须仍 ≤ D (移除首客可改变 wait_c1)
                    if not _move_dur_ok(inst, from_di, new_from, ri):
                        continue

                    # 尝试插入到所有路由的所有位置
                    best_delta, best_tgt = 0.0, None
                    for to_di, to_routes in sol.routes.items():
                        for rj, t_route in enumerate(to_routes):
                            if to_di == from_di and rj == ri:
                                continue
                            # 容量检查: 插入后目标路由不得超容 (load 已缓存)
                            if (route_loads[(to_di, rj)] + _demand(inst, ci)
                                    > inst.max_vehicle_capacity + 1e-9):
                                continue
                            # TW 增量: 目标路由时间线一次构建 (to 侧检查;
                            # from 侧移除不破坏可行性: 欧氏三角 → 移除后
                            # 到达不晚于原到达, excess 单调不增)
                            arr = done = cum_exc = None
                            if tw and t_route:
                                arr, done, cum_exc = _route_timeline(
                                    inst, to_di, t_route, rj)
                            for ip in range(len(t_route) + 1):
                                # 方向 A: 新弧 (prev→ci)/(ci→next) 硬不可达 → 无损剪
                                if R is not None:
                                    if ip > 0 and not R[t_route[ip - 1] - nd][ci - nd]:
                                        continue
                                    if ip < len(t_route) and not R[ci - nd][t_route[ip] - nd]:
                                        continue
                                # 距离增量 O(1) (可行位置 TW 项为 0)
                                delta_to = _arc_delta_insert(
                                    inst, to_di, t_route, ip, ci)
                                d = delta_from + delta_to
                                if d < best_delta - 1e-9:
                                    if tw and t_route and _insert_incremental_excess(
                                            inst, to_di, t_route, rj, ip, ci,
                                            arr, done, cum_exc) > 1e-9:
                                        continue
                                    if not _move_dur_ok(
                                            inst, to_di,
                                            t_route[:ip] + [ci] + t_route[ip:], rj):
                                        continue
                                    best_delta, best_tgt = d, (to_di, rj, ip)
                    if best_tgt is not None:
                        to_di, rj, ip = best_tgt
                        sol.routes[to_di][rj].insert(ip, ci)
                        new_from = route[:pos] + route[pos + 1:]
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


def swap_improve(solution: Solution, max_iterations: int = 50,
                 rng: np.random.Generator | None = None) -> Solution:
    """跨路由 swap: 交换两个不同路由中的客户。

    接受标准: 综合目标严格改进 + 两条路由 TW 均可行。
    rng: 兼容 engine 统一签名。
    """
    if max_iterations <= 0:
        return solution
    sol = solution.copy()
    inst = sol.instance
    R = inst.reach_matrix   # 方向 A
    nd = inst.num_depots
    tw = inst.has_time_windows
    from src.operators.repair import _replace_incremental_excess, _route_timeline

    for _ in range(max_iterations):
        improved = False
        for di_a, routes_a in list(sol.routes.items()):
            for ri_a, route_a in list(enumerate(routes_a)):
                load_a = _route_load(inst, route_a)
                arr_a = done_a = cum_a = None
                if tw:
                    arr_a, done_a, cum_a = _route_timeline(inst, di_a, route_a, ri_a)
                for pi_a, ci_a in enumerate(route_a):
                    d_a = _demand(inst, ci_a)
                    for di_b, routes_b in list(sol.routes.items()):
                        for ri_b, route_b in list(enumerate(routes_b)):
                            if (di_a, ri_a) == (di_b, ri_b):
                                continue
                            load_b = _route_load(inst, route_b)
                            arr_b = done_b = cum_b = None
                            if tw:
                                arr_b, done_b, cum_b = _route_timeline(
                                    inst, di_b, route_b, ri_b)
                            for pi_b, ci_b in enumerate(route_b):
                                # 容量检查: 交换后两条路由均不得超容
                                d_b = _demand(inst, ci_b)
                                if (load_a - d_a + d_b
                                        > inst.max_vehicle_capacity + 1e-9):
                                    continue
                                if (load_b - d_b + d_a
                                        > inst.max_vehicle_capacity + 1e-9):
                                    continue
                                # 方向 A: 交换后 4 条新弧任一硬不可达 → 无损剪
                                if R is not None:
                                    if ((pi_a > 0 and not R[route_a[pi_a - 1] - nd][ci_b - nd])
                                            or (pi_a < len(route_a) - 1 and not R[ci_b - nd][route_a[pi_a + 1] - nd])
                                            or (pi_b > 0 and not R[route_b[pi_b - 1] - nd][ci_a - nd])
                                            or (pi_b < len(route_b) - 1 and not R[ci_a - nd][route_b[pi_b + 1] - nd])):
                                        continue
                                # 距离增量 O(1) 弧差 (d<0 才做 TW 白试, 同原语义)
                                d = (_arc_delta_replace(inst, di_a, route_a, pi_a, ci_b)
                                     + _arc_delta_replace(inst, di_b, route_b, pi_b, ci_a))
                                if d < -1e-9:
                                    # TW 增量: 各自路由内替换点传播 (判定与全重算一致)
                                    if tw and (_replace_incremental_excess(
                                            inst, di_a, route_a, ri_a, pi_a, ci_b,
                                            arr_a, done_a, cum_a) > 1e-9
                                            or _replace_incremental_excess(
                                            inst, di_b, route_b, ri_b, pi_b, ci_a,
                                            arr_b, done_b, cum_b) > 1e-9):
                                        continue
                                    ra2 = route_a[:]
                                    rb2 = route_b[:]
                                    ra2[pi_a], rb2[pi_b] = ci_b, ci_a
                                    if (not _move_dur_ok(inst, di_a, ra2, ri_a)
                                            or not _move_dur_ok(inst, di_b, rb2, ri_b)):
                                        continue
                                    sol.routes[di_a][ri_a] = ra2
                                    sol.routes[di_b][ri_b] = rb2
                                    improved = True
                                    break
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


def two_opt_star_improve(solution: Solution, max_iterations: int = 50,
                         rng: np.random.Generator | None = None) -> Solution:
    """2-opt*: 跨路由重连 — 交换两条路由的尾部段。

    rng: 兼容 engine 统一签名。
    """
    if max_iterations <= 0:
        return solution
    sol = solution.copy()
    inst = sol.instance
    R = inst.reach_matrix   # 方向 A
    nd = inst.num_depots
    tw = inst.has_time_windows
    from src.operators.repair import _cross_tail_excess, _route_timeline

    for _ in range(max_iterations):
        improved = False
        for di_a, routes_a in list(sol.routes.items()):
            for ri_a, route_a in list(enumerate(routes_a)):
                load_a = _route_load(inst, route_a)
                cap = inst.max_vehicle_capacity
                arr_a = done_a = cum_a = None
                if tw:
                    arr_a, done_a, cum_a = _route_timeline(inst, di_a, route_a, ri_a)
                for di_b, routes_b in list(sol.routes.items()):
                    for ri_b, route_b in list(enumerate(routes_b)):
                        if (di_a, ri_a) == (di_b, ri_b):
                            continue
                        load_b = _route_load(inst, route_b)
                        arr_b = done_b = cum_b = None
                        if tw:
                            arr_b, done_b, cum_b = _route_timeline(
                                inst, di_b, route_b, ri_b)
                        prefix_a = 0.0
                        for cut_a in range(1, len(route_a)):
                            prefix_a += _demand(inst, route_a[cut_a - 1])
                            tail_a = load_a - prefix_a
                            prefix_b = 0.0
                            for cut_b in range(1, len(route_b)):
                                prefix_b += _demand(inst, route_b[cut_b - 1])
                                tail_b = load_b - prefix_b
                                # 容量检查: 尾段交换后两条路由均不得超容
                                if prefix_a + tail_b > cap + 1e-9:
                                    continue
                                if prefix_b + tail_a > cap + 1e-9:
                                    continue
                                # 方向 A: 接合新弧硬不可达 → 无损剪
                                if R is not None:
                                    if (not R[route_a[cut_a - 1] - nd][route_b[cut_b] - nd]
                                            or not R[route_b[cut_b - 1] - nd][route_a[cut_a] - nd]):
                                        continue
                                # 距离增量 O(1) 接合弧差
                                d = _arc_delta_stitch(
                                    inst, di_a, di_b, route_a, cut_a,
                                    route_b, cut_b)
                                if d < -1e-9:
                                    # TW 增量: 尾段以新前缀传播, 吸收基准 =
                                    # 原路由时间线 (原 excess=0 → 吸收即零)
                                    if tw:
                                        # A' = A[:cut_a] + B[cut_b:]
                                        sd_a = done_a[cut_a - 1]
                                        sn_a = route_a[cut_a - 1]
                                        if _cross_tail_excess(
                                                inst, di_b, route_b, cut_b,
                                                sd_a, sn_a, arr_b, done_b,
                                                cum_b) > 1e-9:
                                            continue
                                        # B' = B[:cut_b] + A[cut_a:]
                                        sd_b = done_b[cut_b - 1]
                                        sn_b = route_b[cut_b - 1]
                                        if _cross_tail_excess(
                                                inst, di_a, route_a, cut_a,
                                                sd_b, sn_b, arr_a, done_a,
                                                cum_a) > 1e-9:
                                            continue
                                    ra2 = route_a[:cut_a] + route_b[cut_b:]
                                    rb2 = route_b[:cut_b] + route_a[cut_a:]
                                    if (not _move_dur_ok(inst, di_a, ra2, ri_a)
                                            or not _move_dur_ok(inst, di_b, rb2, ri_b)):
                                        continue
                                    sol.routes[di_a][ri_a] = ra2
                                    sol.routes[di_b][ri_b] = rb2
                                    improved = True
                                    break
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


def rvnd_improve(solution: Solution, max_iterations: int = 50,
                 rng: np.random.Generator | None = None) -> Solution:
    """RVND: 随机可变邻域下降 — 随机打乱邻域顺序, 依次尝试各邻域。

    邻域集: [two_opt, relocate, swap, two_opt_star]。
    任一邻域改进后立即重启邻域集; 全部无改进则终止。
    """
    if max_iterations <= 0:
        return solution
    if rng is None:
        rng = np.random.default_rng()
    sol = solution.copy()
    neighborhoods = [two_opt_improve, relocate_improve, swap_improve,
                     two_opt_star_improve]

    budget = max_iterations
    while budget > 0:
        # 随机打乱邻域顺序 (RVND 关键)
        order = rng.permutation(len(neighborhoods))
        improved_any = False
        for idx in order:
            before_cost = calculate_objectives(sol)[0]
            # 每个邻域只做一轮改进 (内部 full pass), RVND 层控制总预算
            candidate = neighborhoods[idx](sol, max_iterations=1)
            cand_cost = calculate_objectives(candidate)[0]
            # 只接受严格改进 + 完全可行 (容量+TW+覆盖) 的候选 (保证单调)
            if cand_cost < before_cost - 1e-9 and candidate.is_feasible():
                sol = candidate
                improved_any = True
                budget -= 1
                break
        if not improved_any:
            break
    return sol
