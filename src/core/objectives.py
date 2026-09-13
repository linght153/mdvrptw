"""双目标计算：成本 + 排放。"""

from src.core.instance import Instance
from src.core.solution import Solution


def calculate_cost(solution: Solution) -> float:
    """
    计算总行驶成本 = 单位成本 × 总距离。
    """
    inst = solution.instance
    total_distance = 0.0

    for depot_idx, routes in solution.routes.items():
        depot_node = inst.depots[depot_idx]
        for route in routes:
            if not route:
                continue
            # 车场 → 第一个客户
            total_distance += inst.distance_matrix[depot_node.index][route[0]]
            # 客户之间
            for i in range(len(route) - 1):
                total_distance += inst.distance_matrix[route[i]][route[i + 1]]
            # 最后一个客户 → 车场
            total_distance += inst.distance_matrix[route[-1]][depot_node.index]

    return total_distance * inst.emission_model.unit_cost_per_km


def calculate_emission(solution: Solution) -> float:
    """计算总 CO₂ 排放。

    支持两种模型 (由 inst.emission_model.type 决定):
    - MEET_based (默认): base_rate × distance × (1 + alpha × load_ratio)
    - COPERT: Σ 因子(g/km) × 弧距离, 因子按 Euro 标准 + 平均速度查表,
      支持逐弧速度剖面 speed_profile{(from,to): kmh}; 无剖面用 speed_kmh。
    """
    inst = solution.instance
    model = inst.emission_model
    if model.type.upper() == "COPERT":
        return _calculate_emission_copert(solution)
    return _calculate_emission_meet(solution)


def _calculate_emission_meet(solution: Solution) -> float:
    """MEET 模型: 带负载修正的排放 (旧行为, 完全保留)。"""
    inst = solution.instance
    base_rate = inst.emission_model.base_emission_rate
    alpha = inst.emission_model.load_correction_alpha
    if base_rate == 0.0:
        # 单目标模式 (MDVRPTW pr 实例默认全零): 排放恒 0, 短路省全遍历
        return 0.0
    cap = inst.max_vehicle_capacity
    total_emission = 0.0

    for depot_idx, routes in solution.routes.items():
        depot_node = inst.depots[depot_idx]
        for route in routes:
            if not route:
                continue

            # 该路由的总需求量
            route_demands = []
            for c_idx in route:
                c = inst.customers[c_idx - inst.num_depots]
                route_demands.append(c.demand)

            total_route_demand = sum(route_demands)
            remaining = total_route_demand

            # 车场 → 第一个客户（满载）
            load_ratio = remaining / cap if cap > 0 else 0
            dist = inst.distance_matrix[depot_node.index][route[0]]
            total_emission += base_rate * dist * (1 + alpha * load_ratio)

            # 客户之间
            for i in range(len(route) - 1):
                remaining -= route_demands[i]
                load_ratio = remaining / cap if cap > 0 else 0
                dist = inst.distance_matrix[route[i]][route[i + 1]]
                total_emission += base_rate * dist * (1 + alpha * load_ratio)

            # 最后一个客户 → 车场（空载返回）
            remaining -= route_demands[-1]
            load_ratio = remaining / cap if cap > 0 else 0
            dist = inst.distance_matrix[route[-1]][depot_node.index]
            total_emission += base_rate * dist * (1 + alpha * load_ratio)

    return total_emission


def _arcs_of(solution: Solution) -> list[tuple[int, int, float]]:
    """列出解的所有弧: (from_idx, to_idx, distance)。"""
    inst = solution.instance
    arcs = []
    for depot_idx, routes in solution.routes.items():
        depot_node = inst.depots[depot_idx]
        for route in routes:
            if not route:
                continue
            prev = depot_node.index
            for c_idx in route:
                arcs.append((prev, c_idx, inst.distance_matrix[prev][c_idx]))
                prev = c_idx
            arcs.append((prev, depot_node.index, inst.distance_matrix[prev][depot_node.index]))
    return arcs


def _speed_mult_at(t: float, profile: list) -> float:
    """查询 t 时刻的速度倍率 (分时段分段常数)。"""
    for t0, t1, mult in profile:
        if t0 <= t < t1:
            return float(mult)
    return float(profile[-1][2])  # 超出范围用最后一段


def _calculate_emission_copert(solution: Solution) -> float:
    """COPERT 模型: 排放 = Σ 因子(g/km) × 弧距离。

    有 time_speed_profile 时走时变分支 (因子随 route 累计时刻变化);
    否则逐弧 speed_profile 或固定 speed_kmh。

    (2026-09-08 验收) src/green 已随精简提交 2dc0c97 移除 (归档于 tag
    pre-reorg-20260902), 本分支为残留半清理 — 调用必抛, 错误信息指向归档。
    """
    raise NotImplementedError(
        "COPERT 排放模型已随 src/green 移除 (2dc0c97, 归档于 tag "
        "pre-reorg-20260902 + bundle); 本分支不可用。"
    )


def _calculate_emission_copert_tv(solution: Solution) -> float:
    """时变速度 COPERT (TDPRP 风格): 弧因子随 route 累计时刻变化。

    t 为抽象时间单位 (与 TW 同量级): 车场出发 t = departure_time_offset
    (默认 0), 每弧 travel = dist / mult (mult=1 基准时 travel=dist),
    等待 + 服务推进时刻; 物理速度 = mult × speed_kmh → 查 COPERT 因子表
    → 成本-碳排解耦。

    (2026-09-08 验收) 与 _calculate_emission_copert 同 — src/green 已移除,
    本函数不可用 (保留签名以锁定 API)。
    """
    raise NotImplementedError(
        "COPERT 时变分支已随 src/green 移除 (2dc0c97, 归档于 tag "
        "pre-reorg-20260902); 本分支不可用。"
    )


def calculate_objectives(solution: Solution) -> tuple[float, float]:
    """返回 (成本, 排放) 元组。"""
    return calculate_cost(solution), calculate_emission(solution)


def calculate_objectives_tw(
    solution: Solution,
    tw_penalty_weight: float = 0.0,
) -> tuple[float, float, float]:
    """返回 (成本, 排放, TW 惩罚) 三元组。

    TW 惩罚 = tw_penalty_weight × Σ max(0, arrival - due_time)。
    仅当实例有 TW 且权重 > 0 时计算; 否则惩罚为 0 (兼容无 TW 场景)。
    """
    cost, emission = calculate_objectives(solution)
    inst = solution.instance
    if not inst.has_time_windows or tw_penalty_weight <= 0.0:
        return cost, emission, 0.0
    excess_total = sum(v["excess"] for v in solution.tw_violations())
    return cost, emission, tw_penalty_weight * excess_total


def combined_objective(solution: Solution, weight_cost: float = 0.5) -> float:
    """单目标对比用的成本与排放加权和。"""
    cost, emission = calculate_objectives(solution)
    # 近似归一化：emission ~ cost * base_rate，权重均等
    return weight_cost * cost + (1 - weight_cost) * emission


def removal_cost_delta(
    solution: Solution, customer_idx: int
) -> tuple[float, float]:
    """
    估计移除某客户后的成本和排放减少量。
    用于 Worst Cost/Emission Removal 算子。

    返回 (成本节省, 排放节省)。正值 = 移除有利。
    """
    inst = solution.instance

    # 查找该客户所属的车场和路由
    for depot_idx, routes in solution.routes.items():
        for route_idx, route in enumerate(routes):
            if customer_idx in route:
                pos = route.index(customer_idx)
                # 计算移除该客户节省的距离
                n = len(route)
                if n == 1:
                    # 路由变为空，节省 车场→客户 + 客户→车场
                    dist_saved = (
                        inst.distance_matrix[depot_idx][customer_idx]
                        + inst.distance_matrix[customer_idx][depot_idx]
                    )
                elif pos == 0:
                    dist_saved = (
                        inst.distance_matrix[depot_idx][customer_idx]
                        + inst.distance_matrix[customer_idx][route[1]]
                        - inst.distance_matrix[depot_idx][route[1]]
                    )
                elif pos == n - 1:
                    dist_saved = (
                        inst.distance_matrix[route[pos - 1]][customer_idx]
                        + inst.distance_matrix[customer_idx][depot_idx]
                        - inst.distance_matrix[route[pos - 1]][depot_idx]
                    )
                else:
                    dist_saved = (
                        inst.distance_matrix[route[pos - 1]][customer_idx]
                        + inst.distance_matrix[customer_idx][route[pos + 1]]
                        - inst.distance_matrix[route[pos - 1]][route[pos + 1]]
                    )

                # 移除 delta = 弧长节省 × 单位成本 (emission 分量同构;
                # 旧实现先全解重算再相减, 代数上完全抵消 → 删除全遍历)。
                # (2026-09-08 验收) 抵消仅在 α=0 成立: MEET α>0 时移除客户使
                # 其前方弧载重比下降, 真实排放节省被系统性低估 (实测 ~33-54%);
                # 成本分量 (unit_cost×Δdist) 精确。绿色线已归档 (2dc0c97),
                # 未修正公式以保持历史口径。
                return (inst.emission_model.unit_cost_per_km * dist_saved,
                        inst.emission_model.base_emission_rate * dist_saved)

    return (0.0, 0.0)
