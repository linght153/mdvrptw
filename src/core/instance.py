"""Green MDVRP JSON 格式算例加载器。"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.helpers import project_root


@dataclass
class Depot:
    index: int
    id: int
    x: float
    y: float
    vehicles_available: int
    capacity: float
    tw_late: float | None = None   # 最迟返回时刻 (MDVRPTW 车场 l; None=不检查)


@dataclass
class Customer:
    index: int
    id: int
    x: float
    y: float
    demand: float
    assigned_depot_index: int
    ready_time: float = 0.0      # 最早开始服务时间 (无 TW 时为 0)
    due_time: float = float("inf")  # 最晚开始服务时间 (无 TW 时为 inf)
    service_time: float = 0.0    # 服务时长


@dataclass
class EmissionModel:
    type: str = "MEET_based"
    base_emission_rate: float = 0.155
    load_correction_alpha: float = 0.30
    unit_cost_per_km: float = 1.0
    # COPERT 模式参数
    euro_standard: str = "euro6"        # euro3/4/5/6/6e
    speed_kmh: float = 40.0             # 默认平均速度 (无速度剖面时)
    speed_profile: dict | None = None   # {(from_idx, to_idx): kmh} 逐弧速度
    # 时变速度剖面 (TDPRP 风格): [(t_start, t_end, speed_multiplier), ...]
    # t 为抽象时间单位 (与 TW ready_time/due_time 同量级), multiplier 为
    # 相对基准速度 (speed_kmh) 的倍率; 使成本(距离)-碳排解耦
    time_speed_profile: list | None = None
    departure_time_offset: float = 0.0   # 时变速度: 车场出发时刻偏移 (抽象时间, 默认 0)


@dataclass
class Instance:
    name: str
    source: str
    num_depots: int
    num_customers: int
    vehicle_capacity: float
    depots: list[Depot]
    customers: list[Customer]
    distance_matrix: list[list[float]]
    emission_model: EmissionModel
    total_nodes: int = 0
    tw_penalty_weight: float = 0.0   # TW 违规惩罚权重 (0=纯硬约束)
    route_duration_limit: float | None = None  # 最大路线时长 D (None=不约束)
    # 动态重优化: 分车起始状态 (2026-09-01, 规划-执行时间轴对齐)
    # - vehicle_start_times: [depot][k] = 车场 d 第 k 辆车的当前时刻
    #   (该车新路由的 TW 时间基线; None = 全 0, 静态行为不变)
    # - vehicle_start_nodes: [depot][k] = 该车当前所在节点 (路由起点;
    #   None = 车场, 静态行为不变)
    vehicle_start_times: list[list[float]] | None = None
    vehicle_start_nodes: list[list[int]] | None = None
    # 时间可达弧剪枝 (2026-09-03, 方向 A): TW 实例预计算客户对硬不可达弧
    # (e_a+s_a+t_ab > l_b → 任何可行解都不可能连续访问 a→b), repair/LS
    # 候选枚举时无损跳过。use_reach_prune=False 可关 (逐位等价测试用);
    # 无 TW 实例矩阵为 None, 零开销。
    use_reach_prune: bool = True
    _reach_cache: "np.ndarray | None" = None  # noqa: F821  (惰性, 非数据字段语义)
    _distance_max_cache: float | None = None
    _demand_max_cache: float | None = None
    _has_tw_cache: bool | None = None

    def __post_init__(self):
        self.total_nodes = self.num_depots + self.num_customers

    @property
    def reach_matrix(self) -> "np.ndarray | None":
        """客户对硬可达矩阵 (n×n bool, 客户相对索引); 无 TW 或关闭时 None。

        可达: e_a + s_a + t_ab <= l_b (等待不可救 → 不可达弧永不可行)。
        车场端点恒可达 (不查矩阵)。
        """
        if not self.has_time_windows or not self.use_reach_prune:
            return None
        if self._reach_cache is not None:
            return self._reach_cache
        import numpy as np
        n = self.num_customers
        base = self.num_depots
        R = np.ones((n, n), dtype=bool)
        for a in self.customers:
            for b in self.customers:
                if a.index == b.index:
                    continue
                if (a.ready_time + a.service_time
                        + self.distance_matrix[a.index][b.index] > b.due_time):
                    R[a.index - base][b.index - base] = False
        self._reach_cache = R
        return R

    @property
    def distance_max(self) -> float:
        """距离矩阵全局最大值 (归一化因子, 惰性缓存; 曾每调用重算 O(n²))。"""
        if self._distance_max_cache is None:
            self._distance_max_cache = max(max(row) for row in self.distance_matrix)
        return self._distance_max_cache

    @property
    def demand_max(self) -> float:
        """客户需求最大值 (归一化因子, 惰性缓存)。"""
        if self._demand_max_cache is None:
            self._demand_max_cache = max(c.demand for c in self.customers)
        return self._demand_max_cache

    @property
    def depot_indices(self) -> list[int]:
        return list(range(self.num_depots))

    @property
    def customer_indices(self) -> list[int]:
        return list(range(self.num_depots, self.total_nodes))

    @property
    def total_demand(self) -> float:
        return sum(c.demand for c in self.customers)

    @property
    def max_vehicle_capacity(self) -> float:
        return self.vehicle_capacity

    @property
    def emission_alpha(self) -> float:
        return self.emission_model.load_correction_alpha

    @property
    def emission_base_rate(self) -> float:
        return self.emission_model.base_emission_rate

    @property
    def unit_cost(self) -> float:
        return self.emission_model.unit_cost_per_km

    @property
    def has_time_windows(self) -> bool:
        """是否存在任何客户具有有限时间窗。"""
        if self._has_tw_cache is None:
            self._has_tw_cache = any(c.due_time < float("inf") for c in self.customers)
        return self._has_tw_cache


def load_instance(name: str, data_dir: str | None = None) -> Instance:
    """
    从 JSON 文件加载 Green MDVRP 算例。

    Args:
        name: 算例名称（如 "p01"、"zg_m01"）
        data_dir: 可选的数据目录覆盖路径
    """
    if data_dir is None:
        data_dir = project_root() / "data" / "green_mdvrp"
    filepath = Path(data_dir) / f"{name}.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Instance file not found: {filepath}")

    with open(filepath, "r") as f:
        raw = json.load(f)

    meta = raw["meta"]
    emission_raw = raw["emission_model"]

    depots = [
        Depot(
            index=d["index"],
            id=d["id"],
            x=d["x"],
            y=d["y"],
            vehicles_available=d["vehicles_available"],
            capacity=d["capacity"],
            tw_late=d.get("tw_late"),
        )
        for d in raw["depots"]
    ]

    customers = [
        Customer(
            index=c["index"],
            id=c["id"],
            x=c["x"],
            y=c["y"],
            demand=c["demand"],
            assigned_depot_index=c["assigned_depot_index"],
        )
        for c in raw["customers"]
    ]

    emission = EmissionModel(
        type=emission_raw.get("type", "MEET_based"),
        base_emission_rate=emission_raw.get("base_emission_rate", 0.155),
        load_correction_alpha=emission_raw.get("load_correction_alpha", 0.30),
        unit_cost_per_km=emission_raw.get("unit_cost_per_km", 1.0),
    )

    return Instance(
        name=meta["name"],
        source=meta.get("source", "unknown"),
        num_depots=meta["num_depots"],
        num_customers=meta["num_customers"],
        vehicle_capacity=meta["vehicle_capacity"],
        depots=depots,
        customers=customers,
        distance_matrix=raw["distance_matrix"],
        emission_model=emission,
        route_duration_limit=raw.get("route_duration_limit",
                                     meta.get("route_duration_limit")),
    )


def load_all_instances(data_dir: str | None = None) -> dict[str, Instance]:
    """从清单文件加载所有算例。"""
    if data_dir is None:
        data_dir = project_root() / "data" / "green_mdvrp"
    manifest_path = Path(data_dir) / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    instances = {}
    for group in manifest.values():
        for name in group["instances"]:
            try:
                instances[name] = load_instance(name, data_dir)
            except FileNotFoundError:
                pass
    return instances


def instance_from_parsed(
    parsed,
    name: str,
    unit_cost_per_km: float = 1.0,
    base_emission_rate: float = 0.0,
    load_correction_alpha: float = 0.0,
) -> Instance:
    """从 ParsedInstance (Cordeau MDVRPTW 格式) 构建带 TW 的 Instance。

    节点编号约定: 车场在前 (0..t-1), 客户在后 (t..t+n-1), 与 load_instance 一致。
    距离矩阵: 欧氏距离 (与 Cordeau 基准一致)。

    MDVRPTW 单目标语义 (2026-09-03): 排放默认全零 — green 线遗留的双目标
    默认 (base_rate 0.155 / alpha 0.3) 曾污染全部 pr 实验: engine 的
    best/archive/accept 比较用 cost+emission 混合标量, 而 BKS/gap 按纯
    距离计 → 搜索在错误目标上运行。
    """
    import math

    depots = [
        Depot(
            index=i,
            id=i,
            x=d["x"],
            y=d["y"],
            vehicles_available=parsed.num_vehicles_per_depot,
            capacity=parsed.vehicle_capacity,
            tw_late=d["due_time"],
        )
        for i, d in enumerate(parsed.depots)
    ]
    customers = [
        Customer(
            index=parsed.num_depots + i,
            id=c["index"],
            x=c["x"],
            y=c["y"],
            demand=c["demand"],
            assigned_depot_index=0,  # 初始默认车场, 搜索中可变
            ready_time=c["ready_time"],
            due_time=c["due_time"],
            service_time=c["service_time"],
        )
        for i, c in enumerate(parsed.customers)
    ]

    nodes = [(d.x, d.y) for d in depots] + [(c.x, c.y) for c in customers]
    n = len(nodes)
    distance_matrix = [
        [math.dist(nodes[i], nodes[j]) for j in range(n)]
        for i in range(n)
    ]

    return Instance(
        name=name,
        source="VRP-REP 2017-0012 Cordeau et al. (2001) MDVRPTW",
        num_depots=len(depots),
        num_customers=len(customers),
        vehicle_capacity=parsed.vehicle_capacity,
        depots=depots,
        customers=customers,
        distance_matrix=distance_matrix,
        emission_model=EmissionModel(
            type="MEET_based",
            base_emission_rate=base_emission_rate,
            load_correction_alpha=load_correction_alpha,
            unit_cost_per_km=unit_cost_per_km,
        ),
        route_duration_limit=parsed.route_duration_limit,
    )
