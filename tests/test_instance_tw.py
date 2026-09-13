"""Instance 时间窗字段扩展测试。

覆盖:
- Customer TW 字段默认值 (无 TW 实例行为不变)
- has_time_windows 标志
- 从 ParsedInstance (Cordeau MDVRPTW) 构建带 TW 的 Instance
- 距离矩阵自动计算 (欧氏)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import (
    Customer,
    Depot,
    EmissionModel,
    Instance,
    load_instance,
)
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw
from src.core.instance import instance_from_parsed  # noqa: F401  (TDD: 先 RED)


def test_customer_tw_defaults():
    """无 TW 实例: ready=0, due=inf, service=0 (legacy 行为)"""
    c = Customer(index=0, id=1, x=0.0, y=0.0, demand=5.0, assigned_depot_index=0)
    assert c.ready_time == 0.0
    assert c.due_time == float("inf")
    assert c.service_time == 0.0


def test_load_instance_without_tw_keeps_legacy_behavior():
    """现有 p01 实例无 TW 字段 → 不报错且行为不变"""
    inst = load_instance("p01")
    assert inst.has_time_windows is False
    assert all(c.due_time == float("inf") for c in inst.customers)


def test_instance_from_parsed_pr01():
    """从解析的 pr01 构建 Instance, 含 TW 且距离矩阵正确"""
    text = (Path(__file__).resolve().parent.parent / "data/mdvrptw_raw/pr01.txt").read_text(encoding="utf-8")
    parsed = parse_mdvrptw(text)
    inst = instance_from_parsed(parsed, name="pr01")

    assert inst.has_time_windows is True
    assert inst.num_depots == 4
    assert inst.num_customers == 48
    assert inst.vehicle_capacity == 200.0

    # 客户 TW 正确 (第一个客户 ready=399, due=525)
    c0 = inst.customers[0]
    assert c0.ready_time == pytest.approx(399.0)
    assert c0.due_time == pytest.approx(525.0)
    assert c0.service_time == pytest.approx(2.0)

    # 距离矩阵: 对称 + 对角 0 + 有限
    n = inst.total_nodes
    assert len(inst.distance_matrix) == n
    assert len(inst.distance_matrix[0]) == n
    assert inst.distance_matrix[0][0] == 0.0
    assert inst.distance_matrix[0][1] == pytest.approx(inst.distance_matrix[1][0])
    assert all(
        inst.distance_matrix[i][j] > 0
        for i in range(n)
        for j in range(n)
        if i != j
    )


def test_instance_from_parsed_all_20():
    """全部 20 个公开实例可转换为 Instance"""
    import json

    manifest = json.loads(
        (Path(__file__).resolve().parent.parent / "data/mdvrptw_raw/manifest.json").read_text(encoding="utf-8")
    )
    for entry in manifest["instances"]:
        text = (Path(__file__).resolve().parent.parent / "data/mdvrptw_raw" / entry["file"]).read_text(encoding="utf-8")
        parsed = parse_mdvrptw(text)
        inst = instance_from_parsed(parsed, name=entry["name"])
        assert inst.has_time_windows is True
        assert inst.num_customers == entry["customers"]
        assert inst.num_depots == entry["depots"]
