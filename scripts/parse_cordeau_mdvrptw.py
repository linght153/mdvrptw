"""Cordeau MDVRPTW 标准格式解析器 (VRP-REP 2017-0012 数据集, Cordeau et al. 2001)。

格式说明 (见 data/mdvrptw_raw/readme.txt):
- 第一行: type m n t   (type=6 MDVRPTW, m=每车场车辆数, n=客户数, t=车场数)
- 接下来 t 行: D Q      (D=最大路线时长, Q=车辆容量)
- 接下来 n 行客户: i x y d q f a list... e l
    (i=编号, x, y, d=服务时长, q=需求, f=频率, a=组合数, list=a 个组合, e=最早, l=最晚)
- 最后 t 行车场: i x y d q f a list... e l (d=q=0, e=0, l=horizon)

注意: a 是变长的! 读 a 个 list token 后再读 e, l。MDVRPTW 实例中 a=4。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedInstance:
    """解析后的 MDVRPTW 实例 (原始格式, 未转成项目 Instance)。"""

    num_vehicles_per_depot: int
    num_customers: int
    num_depots: int
    route_duration_limit: float
    vehicle_capacity: float
    customers: list[dict[str, Any]] = field(default_factory=list)
    depots: list[dict[str, Any]] = field(default_factory=list)

    @property
    def max_vehicle_capacity(self) -> float:
        return self.vehicle_capacity

    @property
    def vehicles_per_depot(self) -> int:
        return self.num_vehicles_per_depot


def _tokens(line: str) -> list[str]:
    return line.split()


def parse_customer_line(line: str) -> dict[str, Any]:
    """解析客户行: i x y d q f a list... e l (a 变长)。"""
    tok = _tokens(line)
    i = int(tok[0])
    x = float(tok[1])
    y = float(tok[2])
    d = float(tok[3])   # 服务时长
    q = float(tok[4])   # 需求
    f = int(tok[5])     # 频率 (MDVRPTW 恒为 1)
    a = int(tok[6])     # 组合数
    assert a >= 1 and len(tok) >= 7 + a + 2, f"客户行 {i} 字段不足"
    # list 占 a 个 token
    list_tokens = tok[7 : 7 + a]
    e = float(tok[7 + a])
    l = float(tok[7 + a + 1])
    return {
        "index": i,
        "x": x,
        "y": y,
        "service_time": d,
        "demand": q,
        "frequency": f,
        "visit_combinations": [int(t) for t in list_tokens],
        "ready_time": e,
        "due_time": l,
    }


def parse_depot_line(line: str) -> dict[str, Any]:
    """解析车场行: i x y d q f a list... e l (d=q=0, e=0, l=horizon)。"""
    tok = _tokens(line)
    i = int(tok[0])
    x = float(tok[1])
    y = float(tok[2])
    a = int(tok[6]) if len(tok) > 6 else 0
    if a > 0:
        e = float(tok[7 + a])
        l = float(tok[7 + a + 1])
    else:
        e = float(tok[7])
        l = float(tok[8])
    return {
        "index": i,
        "x": x,
        "y": y,
        "ready_time": e,
        "due_time": l,
    }


def parse_mdvrptw(text: str) -> ParsedInstance:
    """解析完整 MDVRPTW 文本 → ParsedInstance。"""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("空输入")

    header = _tokens(lines[0])
    type_, m, n, t = int(header[0]), int(header[1]), int(header[2]), int(header[3])
    if type_ != 6:
        raise ValueError(f"不是 MDVRPTW 实例 (type={type_}, 期望 6)")

    # t 行 D Q
    depot_lines = lines[1 : 1 + t]
    route_duration_limit = float(_tokens(depot_lines[0])[0])
    vehicle_capacity = float(_tokens(depot_lines[0])[1])
    for dl in depot_lines[1:]:
        tok = _tokens(dl)
        assert float(tok[1]) == vehicle_capacity, "车场容量不一致"

    # 接下来 n 行客户
    customer_lines = lines[1 + t : 1 + t + n]
    customers = [parse_customer_line(cl) for cl in customer_lines]

    # 最后 t 行车场
    depot_spec_lines = lines[1 + t + n : 1 + t + n + t]
    depots = [parse_depot_line(dl) for dl in depot_spec_lines]

    return ParsedInstance(
        num_vehicles_per_depot=m,
        num_customers=n,
        num_depots=t,
        route_duration_limit=route_duration_limit,
        vehicle_capacity=vehicle_capacity,
        customers=customers,
        depots=depots,
    )
