"""ALNS 修复算子 — 将移除的客户重新插入部分解。"""

import numpy as np

from src.core.solution import Solution, route_return_duration
from src.core.objectives import calculate_cost, calculate_emission


def greedy_cost_insertion(
    partial: Solution, removed: list[int], rng: np.random.Generator
) -> Solution:
    """将每个客户插入使成本增加最小的位置。"""
    sol = partial.copy()
    for ci in removed:
        _insert_cheapest_cost(sol, ci)
    return sol


def greedy_emission_insertion(
    partial: Solution, removed: list[int], rng: np.random.Generator
) -> Solution:
    """将每个客户插入使排放增加最小的位置。"""
    sol = partial.copy()
    for ci in removed:
        _insert_cheapest_emission(sol, ci)
    return sol


def regret_cost_insertion(
    partial: Solution, removed: list[int], rng: np.random.Generator
) -> Solution:
    """
    基于成本的 Regret-k 插入（k=3）。
    优先处理最优与第三优插入成本差值最大的客户，
    避免短视决策。
    """
    sol = partial.copy()
    inst = sol.instance
    k_regret = min(3, sol.total_vehicles() + 1)

    remaining = list(removed)
    while remaining:
        best_regret = -float("inf")
        best_ci = None
        best_pos = None
        best_depot = None

        for ci in remaining:
            costs_per_depot = _all_insertion_costs_cost(sol, ci)
            if not costs_per_depot:
                continue

            # 展平所有 (车场, 路由, 位置, 成本) 选项
            all_options = []
            for di, options in costs_per_depot.items():
                for route_idx, pos, cost in options:
                    all_options.append((di, route_idx, pos, cost))

            if len(all_options) == 0:
                # 必须创建新路由 (最近且未达车辆上限的车场)
                depot_idx = _nearest_depot_with_capacity(sol, ci)
                all_options.append((depot_idx, -1, 0, 0.0))

            all_options.sort(key=lambda x: x[3])
            if len(all_options) >= k_regret:
                regret = all_options[k_regret - 1][3] - all_options[0][3]
            elif len(all_options) >= 2:
                regret = all_options[-1][3] - all_options[0][3]
            else:
                regret = all_options[0][3]

            if regret > best_regret:
                best_regret = regret
                best_ci = ci
                best_depot = all_options[0][0]
                best_pos = (all_options[0][1], all_options[0][2])

        if best_ci is None:
            break

        _place_customer(sol, best_ci, best_depot, best_pos[0], best_pos[1])
        remaining.remove(best_ci)

    return sol


def regret_emission_insertion(
    partial: Solution, removed: list[int], rng: np.random.Generator
) -> Solution:
    """基于排放增加的 Regret-k 插入。"""
    sol = partial.copy()
    inst = sol.instance
    k_regret = min(3, sol.total_vehicles() + 1)

    remaining = list(removed)
    while remaining:
        best_regret = -float("inf")
        best_ci = None
        best_pos = None
        best_depot = None

        for ci in remaining:
            emissions_per_depot = _all_insertion_costs_emission(sol, ci)
            all_options = []
            for di, options in emissions_per_depot.items():
                for route_idx, pos, cost in options:
                    all_options.append((di, route_idx, pos, cost))

            if not all_options:
                depot_idx = _nearest_depot_with_capacity(sol, ci)
                all_options.append((depot_idx, -1, 0, 0.0))

            all_options.sort(key=lambda x: x[3])
            if len(all_options) >= k_regret:
                regret = all_options[k_regret - 1][3] - all_options[0][3]
            elif len(all_options) >= 2:
                regret = all_options[-1][3] - all_options[0][3]
            else:
                regret = all_options[0][3]

            if regret > best_regret:
                best_regret = regret
                best_ci = ci
                best_depot = all_options[0][0]
                best_pos = (all_options[0][1], all_options[0][2])

        if best_ci is None:
            break

        _place_customer(sol, best_ci, best_depot, best_pos[0], best_pos[1])
        remaining.remove(best_ci)

    return sol


def random_insertion(
    partial: Solution, removed: list[int], rng: np.random.Generator
) -> Solution:
    """将每个客户插入随机可行位置 (跨车场遍历所有车场)。"""
    sol = partial.copy()
    inst = sol.instance
    cap = inst.max_vehicle_capacity

    shuffled = list(removed)
    rng.shuffle(shuffled)

    for ci in shuffled:
        c_demand = inst.customers[ci - inst.num_depots].demand
        feasible = []  # (di, ri, pos), ri=-1 表示开新路由
        for di, routes in sol.routes.items():
            for ri, route in enumerate(routes):
                route_demand = sum(
                    inst.customers[x - inst.num_depots].demand for x in route
                )
                if (route_demand + c_demand <= cap
                        or getattr(inst, "soft_capacity", False)):
                    for pos in range(len(route) + 1):
                        feasible.append((di, ri, pos))
            if len(routes) < inst.depots[di].vehicles_available:
                feasible.append((di, -1, 0))

        if feasible:
            di, ri, pos = feasible[int(rng.integers(0, len(feasible)))]
            if ri < 0:
                sol.routes[di].append([ci])
            else:
                sol.routes[di][ri].insert(pos, ci)
            inst.customers[ci - inst.num_depots].assigned_depot_index = di
        else:
            di = _nearest_depot_with_capacity(sol, ci)
            sol.routes[di].append([ci])
            inst.customers[ci - inst.num_depots].assigned_depot_index = di

    return sol


# ============================================================
# 内部辅助函数
# ============================================================

def _insertion_costs(
    solution: Solution, customer_idx: int, cost_fn
) -> dict[int, list[tuple[int, int, float]]]:
    """
    计算客户在每个可行位置的插入成本。

    返回: {车场索引: [(路由索引, 位置, 成本), ...]}
    """
    inst = solution.instance
    c = inst.customers[customer_idx - inst.num_depots]
    cap = inst.max_vehicle_capacity
    results = {}

    for di, routes in solution.routes.items():
        options = []
        for ri, route in enumerate(routes):
            route_demand = sum(
                inst.customers[x - inst.num_depots].demand for x in route
            )
            if (route_demand + c.demand > cap
                    and not getattr(inst, "soft_capacity", False)):
                continue

            if not route:
                # 空路由：唯一有效位置为 pos=0
                old_dist = 0
                new_dist = inst.distance_matrix[di][customer_idx] + inst.distance_matrix[customer_idx][di]
                cost = cost_fn(old_dist, new_dist, customer_idx, 0, route)
                options.append((ri, 0, cost))
                continue

            for pos in range(len(route) + 1):
                # 计算在 pos 插入的成本增量
                if pos == 0:
                    old_dist = inst.distance_matrix[di][route[0]]
                    new_dist = inst.distance_matrix[di][customer_idx] + inst.distance_matrix[customer_idx][route[0]]
                elif pos == len(route):
                    old_dist = inst.distance_matrix[route[-1]][di]
                    new_dist = inst.distance_matrix[route[-1]][customer_idx] + inst.distance_matrix[customer_idx][di]
                else:
                    old_dist = inst.distance_matrix[route[pos - 1]][route[pos]]
                    new_dist = (
                        inst.distance_matrix[route[pos - 1]][customer_idx]
                        + inst.distance_matrix[customer_idx][route[pos]]
                    )

                cost = cost_fn(old_dist, new_dist, customer_idx, pos, route)
                options.append((ri, pos, cost))

        if options:
            results[di] = options

    return results


def _all_insertion_costs_cost(solution, ci):
    """获取客户 ci 基于成本的所有插入成本。"""
    inst = solution.instance
    def cost_fn(old_d, new_d, ci, pos, route):
        return new_d - old_d
    return _insertion_costs(solution, ci, cost_fn)


def _all_insertion_costs_emission(solution, ci):
    """获取客户 ci 基于排放的所有插入成本。"""
    inst = solution.instance
    def cost_fn(old_d, new_d, ci, pos, route):
        # 近似排放增量（为速度忽略负载变化）
        return new_d - old_d
    return _insertion_costs(solution, ci, cost_fn)


def _nearest_depot_with_capacity(solution, ci):
    """开新车时的车场选择: 距离最近且未达车辆上限的车场 (权威解"多用车"策略)。

    车辆上限按**每个车场自己的 vehicles_available** (曾用 depots[0] 当全局
    上限 — 车场间车辆数不同时会误判, 2026-09-01 修复)。
    """
    inst = solution.instance
    best_depot = None
    best_dist = float("inf")
    for di in range(inst.num_depots):
        n_veh = len(solution.routes.get(di, []))
        if n_veh >= inst.depots[di].vehicles_available:
            continue
        d = inst.distance_matrix[di][ci]
        if d < best_dist:
            best_dist = d
            best_depot = di
    if best_depot is None:
        best_depot = min(range(inst.num_depots),
                         key=lambda di: inst.distance_matrix[di][ci])
    return best_depot


def _lightest_route_insert(solution, ci):
    """兜底插入: 所有车场车辆满且无容量可行位时, 插入负载最轻路由末尾。

    绝不增加车辆数 (旧"就近开新车"与车辆数硬约束冲突 → _trim_to_vehicle_limit
    死循环)。可能超容, 由调用方 (is_feasible/_enforce_capacity) 判定处置。
    """
    inst = solution.instance
    c = inst.customers[ci - inst.num_depots]
    best = None  # (di, ri, load)
    for di, routes in solution.routes.items():
        for ri, route in enumerate(routes):
            load = sum(inst.customers[x - inst.num_depots].demand for x in route)
            if best is None or load < best[2]:
                best = (di, ri, load)
    if best is None:
        # 无任何现有路由 (极端): 只能开新车
        di = min(range(inst.num_depots), key=lambda d: inst.distance_matrix[d][ci])
        solution.routes[di].append([ci])
        c.assigned_depot_index = di
        return
    di, ri, _ = best
    solution.routes[di][ri].append(ci)
    c.assigned_depot_index = di


def _insert_cheapest_cost(solution, ci):
    """将客户插入成本最低的位置 (跨车场遍历 + 开新路由候选)。

    修复前只按 assigned_depot_index 插入单一路由 → 非 TW 实例 (assigned
    全 0) 只用车场 0。现在遍历所有车场所有位置 + 每个未满车场的开新路由
    候选, 选全局最低成本 (多车场重分配核心)。
    """
    inst = solution.instance
    c = inst.customers[ci - inst.num_depots]
    cap = inst.max_vehicle_capacity

    best_cost = float("inf")
    best = None  # (di, ri, pos), ri=-1 表示开新路由

    for di, routes in solution.routes.items():
        # 已有路由的插入位置
        for ri, route in enumerate(routes):
            route_demand = sum(
                inst.customers[x - inst.num_depots].demand for x in route
            )
            if (route_demand + c.demand > cap
                    and not getattr(inst, "soft_capacity", False)):
                continue
            for pos in range(len(route) + 1):
                delta = _insertion_delta(inst, route, pos, ci, di)
                if delta < best_cost:
                    best_cost = delta
                    best = (di, ri, pos)
        # 开新路由候选 (该车场车辆未达上限)
        if len(routes) < inst.depots[di].vehicles_available:
            new_cost = (inst.distance_matrix[di][ci]
                        + inst.distance_matrix[ci][di])
            if new_cost < best_cost:
                best_cost = new_cost
                best = (di, -1, None)

    if best is None:
        # 无容量可行位 (紧绑定): 优先在未满车场开新车 (合法操作);
        # 仅当**所有车场都满**时兜底插负载最轻路由 — 绝不增加车辆数
        # (旧实现无条件"就近开新车"会让 _trim_to_vehicle_limit 陷入
        # 移除→重插→开新车→超限→再移除 死循环: 实测 p07 子问题 22 客户
        # 需求=4×100 恰好满载, 500 迭代 >200s 跑不完)。
        # 兜底插入可能超容, 由调用方 (is_feasible/_enforce_capacity) 判定处置。
        di = _nearest_depot_with_capacity(solution, ci)
        if len(solution.routes[di]) < solution.instance.depots[di].vehicles_available:
            solution.routes[di].append([ci])
            c.assigned_depot_index = di
        else:
            _lightest_route_insert(solution, ci)
    else:
        di, ri, pos = best
        if ri < 0:
            solution.routes[di].append([ci])
        else:
            solution.routes[di][ri].insert(pos, ci)
        c.assigned_depot_index = di


"""TW-aware 贪婪修复算子 (greedy_cost_tw_insertion)。

TW 权重回退值: TW 实例且实例未显式配置 tw_penalty_weight (默认 0) 时,
用此权重让 TW 违规增量主导插入选择, 保证产出 TW 可行解。
"""

TW_WEIGHT_FALLBACK = 1000.0


def _route_start(inst, depot_idx: int, route_index: int):
    """该路由对应车辆的 (起始时刻, 起始节点) (与 Solution._route_start 同逻辑)。"""
    depot_node = inst.depots[depot_idx].index
    if inst.vehicle_start_times is None:
        return 0.0, depot_node
    st = inst.vehicle_start_times[depot_idx]
    sn = (inst.vehicle_start_nodes or [None] * len(inst.depots))[depot_idx]
    t0 = st[route_index] if route_index < len(st) else 0.0
    n0 = sn[route_index] if sn and route_index < len(sn) else depot_node
    return float(t0), int(n0)


def _tw_insertion_delta(solution, depot_idx, route, pos, ci,
                        route_index: int = 0) -> float:
    """TW-aware 插入增量 = 距离增量 + 惩罚权重 × TW 违规增量。

    通过比较插入前后整条 route 的 TW 违规总量计算 TW 增量。

    注意: TW 实例下即使 tw_penalty_weight=0 (默认) 也强制考虑 TW —
    用默认权重 TW_WEIGHT_FALLBACK, 否则 greedy_cost_tw 会退化为纯距离
    插入导致 ALNS 产出 TW 违规解 (曾实测 pr01 861.32 含 31 违规)。
    """
    inst = solution.instance
    weight = inst.tw_penalty_weight
    if not inst.has_time_windows:
        # 无 TW: 退化为纯距离增量
        return _insertion_delta(inst, route, pos, ci, depot_idx)
    if weight <= 0.0:
        # TW 实例但权重 0 (默认): 用回退权重保证 TW 优先
        weight = TW_WEIGHT_FALLBACK

    def _tw_excess_of(route_with_ci):
        """计算一条路由的 TW 违规总量 (excess 之和, 分车起始时刻感知)。"""
        return _route_tw_excess(inst, depot_idx, route_with_ci, route_index)

    dist_delta = _insertion_delta(inst, route, pos, ci, depot_idx)

    if not route:
        new_route = [ci]
        old_excess = 0.0
    else:
        new_route = list(route)
        new_route.insert(pos, ci)
        old_excess = _route_tw_excess(inst, depot_idx, route, route_index)

    new_excess = _tw_excess_of(new_route)
    return dist_delta + weight * (new_excess - old_excess)


def _route_tw_excess(inst, depot_idx: int, route: list[int],
                     route_index: int = 0) -> float:
    """增量计算单条路由的 TW 违规总量 (不构造 Solution)。

    O(len(route)) 时间; 用于插入时快速判断 TW 可行性。
    时间基线 = 该车起始时刻, 起点 = 该车当前位置 (动态重优化)。
    """
    if not inst.has_time_windows or not route:
        return 0.0
    t0, n0 = _route_start(inst, depot_idx, route_index)
    excess = 0.0
    t = t0
    prev = n0
    for c_idx in route:
        cust = inst.customers[c_idx - inst.num_depots]
        t += inst.distance_matrix[prev][c_idx]
        if t - cust.due_time > 1e-9:
            excess += t - cust.due_time
        if t < cust.ready_time:
            t = cust.ready_time
        t += cust.service_time
        prev = c_idx
    return excess


def _route_timeline(inst, depot_idx: int, route: list[int],
                    route_index: int = 0):
    """路由时间线: (arr, done, cum_exc) — 每客户的到达/完成时刻 + 违规累计。

    O(len(route)) 一次构建, 供增量插入检查复用 (替代每位置整路由重算)。
    arr[k]: 到达 route[k] 的时刻; done[k]: 完成 (等待+服务后);
    cum_exc[k]: 前 k 个客户的违规累计 (cum_exc[len] = 总违规)。
    """
    t0, n0 = _route_start(inst, depot_idx, route_index)
    nd = inst.num_depots
    arr, done, cum_exc = [], [], [0.0]
    t, prev, acc = t0, n0, 0.0
    for c_idx in route:
        cust = inst.customers[c_idx - nd]
        t += inst.distance_matrix[prev][c_idx]
        arr.append(t)
        if t - cust.due_time > 1e-9:
            acc += t - cust.due_time
        if t < cust.ready_time:
            t = cust.ready_time
        t += cust.service_time
        done.append(t)
        cum_exc.append(acc)
        prev = c_idx
    return arr, done, cum_exc


def _route_timeline_metrics(inst, depot_idx: int, route: list[int],
                            route_index: int = 0):
    """路由时间线 + (return_time, duration) — 一次前向, 供 duration 增量复用。

    返回 (arr, done, cum_exc, return_time, duration)。duration 口径 =
    return_time − wait_{c1} (单一事实源 = solution.route_return_duration)。
    """
    t0, n0 = _route_start(inst, depot_idx, route_index)
    nd = inst.num_depots
    arr, done, cum_exc = [], [], [0.0]
    t, prev, acc = t0, n0, 0.0
    first_wait = None
    for c_idx in route:
        cust = inst.customers[c_idx - nd]
        t += inst.distance_matrix[prev][c_idx]
        arr.append(t)
        if t - cust.due_time > 1e-9:
            acc += t - cust.due_time
        if t < cust.ready_time:
            if first_wait is None:
                first_wait = cust.ready_time - t
            t = cust.ready_time
        elif first_wait is None:
            first_wait = 0.0
        t += cust.service_time
        done.append(t)
        cum_exc.append(acc)
        prev = c_idx
    if not route:
        return arr, done, cum_exc, 0.0, 0.0
    return_time = t + inst.distance_matrix[prev][depot_idx]
    duration = return_time - (first_wait if first_wait is not None else 0.0)
    return arr, done, cum_exc, float(return_time), float(duration)


def _insert_incremental_return(inst, depot_idx: int, route: list[int],
                               pos: int, ci: int, done: list[float],
                               route_index: int = 0) -> float:
    """插入 ci@pos (pos>0) 后的新回场时刻 (从 pos 向前传播 O(len−pos))。

    首客不变 → 其等待不变; 只需新回场时刻即可得 duration 增量。pos=0 由
    调用方走 route_return_duration 全量重算 (低频)。
    """
    t0, n0 = _route_start(inst, depot_idx, route_index)
    nd = inst.num_depots
    prev_done = done[pos - 1] if pos > 0 else t0
    prev_node = route[pos - 1] if pos > 0 else n0
    t = prev_done + inst.distance_matrix[prev_node][ci]
    cust_ci = inst.customers[ci - nd]
    if t < cust_ci.ready_time:
        t = cust_ci.ready_time
    t += cust_ci.service_time
    prev_node = ci
    for k in range(pos, len(route)):
        c_idx = route[k]
        cust = inst.customers[c_idx - nd]
        t += inst.distance_matrix[prev_node][c_idx]
        if t < cust.ready_time:
            t = cust.ready_time
        t += cust.service_time
        prev_node = c_idx
    return float(t + inst.distance_matrix[prev_node][depot_idx])


def _append_duration_ok(inst, depot_idx: int, route: list[int], ci: int,
                        route_index: int = 0) -> bool:
    """在 route 末尾追加 ci 后是否仍满足 duration ≤ D (无 D 恒 True)。"""
    if inst.route_duration_limit is None:
        return True
    _, dur = route_return_duration(inst, depot_idx, route + [ci], route_index)
    return dur <= inst.route_duration_limit + 1e-9


def _route_combined_excess(inst, depot_idx: int, route: list[int],
                           route_index: int = 0, include_tw: bool = True) -> float:
    """路由级联合超额 = TW 违规量 + max(0, duration − D)。

    duration 项无 D 实例恒 0; 供可行化爬坡统一度量两类时间违规 —
    时长通道的兜底放置可能再引入 TW 违规 (实测 pr11 紧车队下互相干扰),
    故对两者合并求解而非各自为政。include_tw=False 时只度量时长
    (软 TW 模式: 违规流量是机制本体, 不得在时长修复中被顺手抹掉)。
    """
    if not route:
        return 0.0
    e = 0.0
    if include_tw:
        e += _route_tw_excess(inst, depot_idx, route, route_index)
    D = inst.route_duration_limit
    if D is not None:
        _, dur = route_return_duration(inst, depot_idx, route, route_index)
        if dur > D + 1e-9:
            e += dur - D
    return e


def combined_excess_repair(sol, rng, max_rounds: int = 20, kick_size: int = 3,
                           max_iterations: int = 40, include_tw: bool = True,
                           work_budget: int = 40000, step_cap: int = 3000):
    """联合超额爬坡修复 (最大路线时长违规的兜底通道)。

    以 relocate/swap 逐步降低"联合超额"(TW + duration; include_tw=False
    则只算 duration), 局部最优处以随机扰动 (从违规路线随机清退客户盲插)
    重启; 至超额 0 或 max_rounds 轮耗尽返回。只移动客户: 车辆数不变、
    客户守恒; 迁入受容量约束 (soft_capacity 模式除外), 车场归属末尾统一
    _sync_assigned_depots。随机数只用于扰动 (engine.rng 链, 每次扰动固定
    4 次 draw → 同轨迹同随机流); 无 D 实例直接返回 (行为不变)。

    成本封顶 (2026-09-11 profile: 未封顶时大实例全局违规单次调用 >20s):
    work_budget = 单次调用候选评估上限; step_cap = 单步枚举上限 (达到即
    用当前 best 落子)。均为确定性截断; 小残差场景远达不到上限不受影响。
    负载用增量字典维护 (原实现每个候选重算 sum → 10^7 次 O(L) 累加)。
    """
    inst = sol.instance
    D = inst.route_duration_limit
    if D is None:
        return sol
    cap = inst.max_vehicle_capacity
    soft_cap = bool(getattr(inst, "soft_capacity", False))
    nd = inst.num_depots
    evals = 0

    def r_exc(route, di, ri):
        return _route_combined_excess(inst, di, route, ri, include_tw)

    rl_all = [(di, ri) for di, rl in sol.routes.items()
              for ri in range(len(rl))]
    loads = {k: sum(inst.customers[x - nd].demand
                    for x in sol.routes[k[0]][k[1]]) for k in rl_all}

    def total():
        return sum(r_exc(sol.routes[di][ri], di, ri)
                   for di, rl in sol.routes.items() for ri in range(len(rl)))

    for _ in range(max_rounds):
        if total() <= 1e-9 or evals >= work_budget:
            break
        # ── 爬坡: 每步选全局增益最大的 relocate/swap ──
        for _ in range(max_iterations):
            if total() <= 1e-9 or evals >= work_budget:
                break
            base_e = {k: r_exc(sol.routes[k[0]][k[1]], *k) for k in rl_all}
            sources = [k for k in rl_all if base_e[k] > 1e-9]
            best = None
            best_gain = 1e-9
            step_evals = 0
            stop = False
            for (di, ri) in sources:
                R = sol.routes[di][ri]
                eR = base_e[(di, ri)]
                for pos in range(len(R)):
                    x = R[pos]
                    cx = inst.customers[x - nd]
                    R2 = R[:pos] + R[pos + 1:]
                    eR2 = r_exc(R2, di, ri)
                    for (di2, ri2) in rl_all:
                        if (di2, ri2) == (di, ri):
                            continue
                        if (not soft_cap
                                and loads[(di2, ri2)] + cx.demand > cap):
                            continue
                        Rr = sol.routes[di2][ri2]
                        eRr = base_e[(di2, ri2)]
                        for p2 in range(len(Rr) + 1):
                            T = Rr[:p2] + [x] + Rr[p2:]
                            gain = (eR + eRr) - (eR2 + r_exc(T, di2, ri2))
                            evals += 1
                            step_evals += 1
                            if gain > best_gain:
                                best_gain = gain
                                best = ('reloc',
                                        (di, ri, pos, di2, ri2, p2))
                            if step_evals >= step_cap:
                                stop = True
                                break
                        if stop:
                            break
                    if stop:
                        break
                if stop:
                    break
            if not stop:
                for (di, ri) in sources:
                    R = sol.routes[di][ri]
                    eR = base_e[(di, ri)]
                    for pos in range(len(R)):
                        x = R[pos]
                        cx = inst.customers[x - nd]
                        for (di2, ri2) in rl_all:
                            if (di2, ri2) == (di, ri):
                                continue
                            R2 = sol.routes[di2][ri2]
                            eR2 = base_e[(di2, ri2)]
                            for p2 in range(len(R2)):
                                y = R2[p2]
                                cy = inst.customers[y - nd]
                                if not soft_cap:
                                    if (loads[(di, ri)] - cx.demand
                                            + cy.demand > cap):
                                        continue
                                    if (loads[(di2, ri2)] - cy.demand
                                            + cx.demand > cap):
                                        continue
                                T1 = R[:pos] + [y] + R[pos + 1:]
                                T2 = R2[:p2] + [x] + R2[p2 + 1:]
                                gain = eR + eR2 - (r_exc(T1, di, ri)
                                                   + r_exc(T2, di2, ri2))
                                evals += 1
                                step_evals += 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best = ('swap',
                                            (di, ri, pos, di2, ri2, p2))
                                if step_evals >= step_cap:
                                    stop = True
                                    break
                            if stop:
                                break
                        if stop:
                            break
                    if stop:
                        break
            if best is None:
                break  # 局部最优 → 进入扰动
            kind, (di, ri, pos, di2, ri2, p2) = best
            if kind == 'reloc':
                x = sol.routes[di][ri].pop(pos)
                sol.routes[di2][ri2].insert(p2, x)
                loads[(di, ri)] -= inst.customers[x - nd].demand
                loads[(di2, ri2)] += inst.customers[x - nd].demand
            else:
                x = sol.routes[di][ri][pos]
                y = sol.routes[di2][ri2][p2]
                sol.routes[di][ri][pos], sol.routes[di2][ri2][p2] = y, x
                dx = inst.customers[x - nd].demand
                dy = inst.customers[y - nd].demand
                loads[(di, ri)] += dy - dx
                loads[(di2, ri2)] += dx - dy
        if total() <= 1e-9 or evals >= work_budget:
            break
        # ── 扰动重启: 从违规路线随机清退客户盲插 (固定 4 次 draw/扰动) ──
        viol = [k for k in rl_all
                if r_exc(sol.routes[k[0]][k[1]], *k) > 1e-9]
        if not viol:
            break
        for _ in range(kick_size):
            i1 = int(rng.integers(len(viol)))
            i2 = int(rng.integers(len(rl_all)))
            (d1, r1) = viol[i1]
            (d2, r2) = rl_all[i2]
            L1 = len(sol.routes[d1][r1])
            L2 = len(sol.routes[d2][r2])
            pos = int(rng.integers(max(L1, 1)))
            pos2 = int(rng.integers(L2 + 1))
            if L1 == 0 or (d2, r2) == (d1, r1):
                continue
            x = sol.routes[d1][r1][pos]
            xd = inst.customers[x - nd].demand
            if not soft_cap and loads[(d2, r2)] + xd > cap:
                continue
            sol.routes[d1][r1].pop(pos)
            sol.routes[d2][r2].insert(
                min(pos2, len(sol.routes[d2][r2])), x)
            loads[(d1, r1)] -= xd
            loads[(d2, r2)] += xd

    _sync_assigned_depots(sol)
    return sol


# 临时诊断计数 (验证后移除)
_INC_STATS = {"absorb": 0, "truncate": 0, "full": 0, "total_pos": 0, "len_sum": 0}


def _replace_incremental_excess(inst, depot_idx, route, route_index,
                                pos, new_ci, arr, done, cum_exc) -> float:
    """把 route[pos] 替换为 new_ci 后的路由 TW 违规 (增量, 与全重算判定一致)。

    LS swap 用: 前缀 (0..pos-1) 不变 (违规 = cum_exc[pos]), pos 处换客户,
    后缀 (pos+1..) 原客户续接 — 从 pos+1 起传播, 吸收基准 = 原时间线。
    """
    t0, n0 = _route_start(inst, depot_idx, route_index)
    nd = inst.num_depots
    old_total = cum_exc[-1]
    prefix_old = cum_exc[pos]
    prev_done = done[pos - 1] if pos > 0 else t0
    prev_node = route[pos - 1] if pos > 0 else n0
    t = prev_done + inst.distance_matrix[prev_node][new_ci]
    cust = inst.customers[new_ci - nd]
    exc = 0.0
    if t - cust.due_time > 1e-9:
        exc += t - cust.due_time
    if t < cust.ready_time:
        t = cust.ready_time
    t += cust.service_time
    prev_node = new_ci
    for k in range(pos + 1, len(route)):
        c_idx = route[k]
        cust = inst.customers[c_idx - nd]
        t += inst.distance_matrix[prev_node][c_idx]
        if t <= arr[k]:
            return prefix_old + exc + (old_total - cum_exc[k])
        if t - cust.due_time > 1e-9:
            exc += t - cust.due_time
            if old_total <= 1e-9:
                return prefix_old + exc
        if t < cust.ready_time:
            t = cust.ready_time
        t += cust.service_time
        prev_node = c_idx
    return prefix_old + exc


def _cross_tail_excess(inst, depot_idx, route, cut, start_done, start_node,
                       arr, done, cum_exc) -> float:
    """把 route[cut:] 尾段接到 (start_done, start_node) 后的路由 TW 违规增量。

    2-opt* 用: 前缀换成另一路由的尾 (时刻 = start_done), 尾段客户不变 —
    吸收基准 = 原路由时间线 (原路由 excess=0 时吸收即零违规继承)。
    """
    nd = inst.num_depots
    old_total = cum_exc[-1]
    t = start_done
    prev = start_node
    exc = 0.0
    for k in range(cut, len(route)):
        c_idx = route[k]
        cust = inst.customers[c_idx - nd]
        t += inst.distance_matrix[prev][c_idx]
        if t <= arr[k]:
            # 与原路由同步: 剩余违规继承 (原路由可行时 = 0)
            return exc + (old_total - cum_exc[k])
        if t - cust.due_time > 1e-9:
            exc += t - cust.due_time
            if old_total <= 1e-9:
                return exc
        if t < cust.ready_time:
            t = cust.ready_time
        t += cust.service_time
        prev = c_idx
    return exc


def _insert_incremental_excess(inst, depot_idx, route, route_index,
                               pos, ci, arr, done, cum_exc) -> float:
    """增量: 插入 ci@pos 后的路由 TW 违规量截断界 (判定等价性已实证: 原路由
    可行时与 _route_tw_excess 全重算 0 误拒/0 误收; 数值仅为界, 非精确值 —
    2026-09-08 验收修正注释, 勿当精确违规量使用)。

    吸收: 若新到达 ≤ 原到达, 则等待/服务后完成时刻亦 ≤ 原值, 后续全部
    与原路由同步 → 原路由该点之后的违规直接继承 (cum_exc 差), 无需重算。
    前提: 原路由 excess=0 时违规累计即停 (返回值已 >1e-9, 终值单调不减);
    原路由已有违规时保守传播到吸收点/末尾 (精确值)。
    """
    t0, n0 = _route_start(inst, depot_idx, route_index)
    nd = inst.num_depots
    old_total = cum_exc[-1]
    prefix_old = cum_exc[pos]  # 插入点之前的原违规, 不受插入影响
    _INC_STATS["total_pos"] += 1
    _INC_STATS["len_sum"] += len(route) - pos
    # 到达 ci
    prev_done = done[pos - 1] if pos > 0 else t0
    prev_node = route[pos - 1] if pos > 0 else n0
    t = prev_done + inst.distance_matrix[prev_node][ci]
    cust_ci = inst.customers[ci - nd]
    exc = 0.0
    if t - cust_ci.due_time > 1e-9:
        exc += t - cust_ci.due_time
    if t < cust_ci.ready_time:
        t = cust_ci.ready_time
    t += cust_ci.service_time
    # 从原位置 pos 向后传播
    prev_node = ci
    for k in range(pos, len(route)):
        c_idx = route[k]
        cust = inst.customers[c_idx - nd]
        t += inst.distance_matrix[prev_node][c_idx]
        if t <= arr[k]:
            # 吸收: 后续与原路由同步, 继承其剩余违规
            _INC_STATS["absorb"] += 1
            return prefix_old + exc + (old_total - cum_exc[k])
        if t - cust.due_time > 1e-9:
            exc += t - cust.due_time
            if old_total <= 1e-9:
                # 违规只增不减 (原路由可行): 终值必 >1e-9, 截断返回已足判弃
                _INC_STATS["truncate"] += 1
                return prefix_old + exc
        if t < cust.ready_time:
            t = cust.ready_time
        t += cust.service_time
        prev_node = c_idx
    _INC_STATS["full"] += 1
    return prefix_old + exc


def _insert_cheapest_cost_tw(solution, ci):
    """将客户插入 (距离增量 + TW 惩罚增量) 最低的位置。

    策略:
    1. 跨车场遍历所有路由的所有位置, 仅接受"插入后该路由 TW 可行"的候选;
    2. 选 TW 可行候选中增量最低者;
    3. 若无任何 TW 可行位置: 开新车 (单客户路由, 权威解策略),
       车场选择 = 距离该客户最近且未达车辆上限的车场;
    4. 插入后更新客户的 assigned_depot_index (多车场分配核心)。
    """
    inst = solution.instance
    c = inst.customers[ci - inst.num_depots]
    tw = inst.has_time_windows
    R = inst.reach_matrix if tw else None
    nd = inst.num_depots
    D = inst.route_duration_limit

    best_cost = float("inf")
    best_di = None
    best_ri = None
    best_pos = None

    for di, rlist in solution.routes.items():
        for ri, route in enumerate(rlist):
            route_demand = sum(
                inst.customers[x - nd].demand for x in route
            )
            overload_pen = 0.0
            if getattr(inst, "soft_capacity", False):
                over = route_demand + c.demand - inst.max_vehicle_capacity
                if over > 0:
                    # 软容量: 超载候选按 λ×超载量 计罚 (与接受准则同 λ) —
                    # 距离节省超过罚金时才选超载位, 穿越受控而非放任
                    overload_pen = getattr(
                        inst, "soft_capacity_lambda", 0.0
                    ) * over
            if (route_demand + c.demand > inst.max_vehicle_capacity
                    and not getattr(inst, "soft_capacity", False)):
                continue
            if tw or D is not None:
                # 增量时间线一次构建 (含 duration 口径), 全部位置复用
                arr, done, cum_exc, return_old, dur_old = \
                    _route_timeline_metrics(inst, di, route, ri)
            else:
                arr = done = cum_exc = return_old = dur_old = None
            for pos in range(len(route) + 1):
                # 方向 A: 新弧 (prev→ci)/(ci→next) 硬不可达 → 无损剪
                if R is not None:
                    prev = route[pos - 1] if pos > 0 else None
                    nxt = route[pos] if pos < len(route) else None
                    if ((prev is not None and not R[prev - nd][ci - nd])
                            or (nxt is not None and not R[ci - nd][nxt - nd])):
                        continue
                if tw:
                    # 增量传播检查插入后 TW 可行性 (吸收点提前终止,
                    # 替代整路由 _route_tw_excess 全重算 O(len)/位置)
                    if _insert_incremental_excess(
                            inst, di, route, ri, pos, ci, arr, done,
                            cum_exc) > 1e-9:
                        continue  # 该位置会引入 TW 违规, 跳过
                if D is not None:
                    # 最大路线时长: pos=0 (新首客/空路由) 全量重算 (低频);
                    # pos>0 首客不变 → dur 增量 = 新回场 − 旧回场 (增量传播)
                    if not route or pos == 0:
                        _, dur_new = route_return_duration(
                            inst, di, [ci] + route, ri)
                    else:
                        return_new = _insert_incremental_return(
                            inst, di, route, pos, ci, done, ri)
                        dur_new = dur_old + (return_new - return_old)
                    if dur_new > D + 1e-9:
                        continue  # 该位置会超长, 跳过
                # 可行位置 TW 增量 = weight×(0−0) = 0 → 选纯距离增量最低
                delta = _insertion_delta(inst, route, pos, ci, di) + overload_pen
                if delta < best_cost:
                    best_cost = delta
                    best_di = di
                    best_ri = ri
                    best_pos = pos

    if best_di is not None:
        solution.routes[best_di][best_ri].insert(best_pos, ci)
        c.assigned_depot_index = best_di
    else:
        # 无任何 TW 可行位置: 优先在未满车场开新车 (权威解策略);
        # 所有车场车辆已满时兜底插负载最轻路由 (不增车) —
        # 旧实现就近车场无条件 append 新路由 → 车辆数违规
        # (实测 pr11: 每车场 1 辆上限, depot3 出现 2 条路由,
        # 非 LS 迭代漏进档案)。
        di = _nearest_depot_with_capacity(solution, ci)
        if len(solution.routes[di]) < solution.instance.depots[di].vehicles_available:
            solution.routes[di].append([ci])
            c.assigned_depot_index = di
        else:
            _lightest_route_insert(solution, ci)


def greedy_cost_tw_insertion(
    partial: Solution, removed: list[int], rng: np.random.Generator
) -> Solution:
    """TW-aware 贪婪插入: 成本 = 距离增量 + 惩罚权重 × TW 违规增量。"""
    sol = partial.copy()
    for ci in removed:
        _insert_cheapest_cost_tw(sol, ci)
    _sync_assigned_depots(sol)
    return sol


def _sync_assigned_depots(sol: Solution):
    """同步所有客户的车场分配与当前路由一致 (多车场不变量)。"""
    inst = sol.instance
    # 先清空再按路由设置
    for c in inst.customers:
        c.assigned_depot_index = -1
    for di, rl in sol.routes.items():
        for route in rl:
            for ci in route:
                inst.customers[ci - inst.num_depots].assigned_depot_index = di


def _insert_cheapest_emission(solution, ci):
    """将客户插入排放最低位置（基于距离的模型中与成本相同）。"""
    _insert_cheapest_cost(solution, ci)


def _insertion_delta(inst, route, pos, ci, depot_idx):
    """计算在路由 pos 插入 ci 的距离增量。"""
    if not route:
        # 空路由：插入成本为 车场→客户→车场
        return inst.distance_matrix[depot_idx][ci] + inst.distance_matrix[ci][depot_idx]
    if pos == 0:
        old_d = inst.distance_matrix[depot_idx][route[0]]
        new_d = inst.distance_matrix[depot_idx][ci] + inst.distance_matrix[ci][route[0]]
    elif pos == len(route):
        old_d = inst.distance_matrix[route[-1]][depot_idx]
        new_d = inst.distance_matrix[route[-1]][ci] + inst.distance_matrix[ci][depot_idx]
    else:
        old_d = inst.distance_matrix[route[pos - 1]][route[pos]]
        new_d = inst.distance_matrix[route[pos - 1]][ci] + inst.distance_matrix[ci][route[pos]]
    return new_d - old_d


def _place_customer(solution, ci, depot_idx, route_idx, pos):
    """将客户放置在指定位置, 并同步更新客户的车场分配。

    所有跨车场插入算子 (regret/greedy) 必须经由本函数, 保证
    assigned_depot_index 与路由一致 (多车场分配核心不变量)。
    """
    if route_idx < 0:
        # 新路由
        solution.routes[depot_idx].append([ci])
    else:
        solution.routes[depot_idx][route_idx].insert(pos, ci)
    # 同步车场分配 (客户对象可变; Solution.copy 深拷贝后需重设)
    c = solution.instance.customers[ci - solution.instance.num_depots]
    c.assigned_depot_index = depot_idx


# ============================================================
# 注册表
# ============================================================

REPAIR_OPERATORS = {
    "greedy_cost": greedy_cost_insertion,
    "greedy_emission": greedy_emission_insertion,
    "regret_cost": regret_cost_insertion,
    "regret_emission": regret_emission_insertion,
    "random": random_insertion,
    "greedy_cost_tw": greedy_cost_tw_insertion,
}
