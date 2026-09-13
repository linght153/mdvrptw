"""Cordeau MDVRPTW 标准格式解析器测试 (VRP-REP 2017-0012 数据集)。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.parse_cordeau_mdvrptw import (
    parse_customer_line,
    parse_depot_line,
    parse_mdvrptw,
    ParsedInstance,
)

# pr01 头部 6 行 (VRP-REP 官方格式)
PR01_HEADER = """6 2 48 4
500 200
500 200
500 200
500 200
  1  -29.730   64.136  2 12 1 4 1 2 4 8 399 525
  2  -30.664    5.463  7  8 1 4 1 2 4 8 121 299
"""


def test_parse_customer_line_with_tw():
    """客户行 13 字段: i x y d q f a list(4) e l"""
    c = parse_customer_line("  1  -29.730   64.136  2 12 1 4 1 2 4 8 399 525")
    assert c["index"] == 1
    assert c["x"] == pytest.approx(-29.730)
    assert c["y"] == pytest.approx(64.136)
    assert c["service_time"] == 2.0
    assert c["demand"] == 12.0
    assert c["ready_time"] == 399.0
    assert c["due_time"] == 525.0


def test_parse_depot_line():
    """车场行 9 字段: i x y d q f a list 0 horizon(1000)"""
    d = parse_depot_line(" 49    4.163   13.559  0  0 0 0  0 1000")
    assert d["index"] == 49
    assert d["x"] == pytest.approx(4.163)
    assert d["ready_time"] == 0.0
    assert d["due_time"] == 1000.0


def test_parse_full_pr01():
    """解析完整 pr01 文本 → 结构正确的 ParsedInstance"""
    text = (Path(__file__).resolve().parent.parent / "data/mdvrptw_raw/pr01.txt").read_text(encoding="utf-8")
    inst = parse_mdvrptw(text)
    assert inst.num_depots == 4
    assert inst.num_customers == 48
    assert inst.vehicles_per_depot == 2
    assert len(inst.depots) == 4
    assert len(inst.customers) == 48
    # 所有客户有 TW
    assert all(c["due_time"] > c["ready_time"] for c in inst.customers)
    # 车场容量来自 D Q 行
    assert inst.vehicle_capacity == 200.0
    # 车场 horizon
    assert all(d["due_time"] == 1000.0 for d in inst.depots)
    # 客户 id 连续
    ids = [c["index"] for c in inst.customers]
    assert ids == list(range(1, 49))


def test_parse_all_20_instances():
    """全部 20 个实例可解析且规模符合 Cordeau 2001 规格"""
    import json

    manifest = json.loads(
        (Path(__file__).resolve().parent.parent / "data/mdvrptw_raw/manifest.json").read_text(encoding="utf-8")
    )
    for entry in manifest["instances"]:
        text = (Path(__file__).resolve().parent.parent / "data/mdvrptw_raw" / entry["file"]).read_text(encoding="utf-8")
        inst = parse_mdvrptw(text)
        assert inst.num_customers == entry["customers"]
        assert inst.num_depots == entry["depots"]
        assert inst.vehicles_per_depot == entry["vehicles_per_depot"]


def test_invalid_type_rejected():
    """非 MDVRPTW (type != 6) 应被拒绝"""
    with pytest.raises(ValueError, match="MDVRPTW"):
        parse_mdvrptw("2 1 48 4\n500 200\n")
