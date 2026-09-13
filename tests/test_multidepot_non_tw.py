"""非 TW 实例多车场重分配测试: greedy/random 修复算子应跨车场插入。

背景: 多车场分配修复 (61fe88a) 只修了 TW 路径 (greedy_cost_tw 跨车场遍历),
非 TW 的 _insert_cheapest_cost / random_insertion 仍只按 assigned_depot_index
插入单个车场。green_mdvrp 实例 (p01-p23) 的 assigned 原始值全 0 → 非 TW
ALNS 的 repair 只用车场 0, 靠 nearest_neighbor 初始解才分到多车场, 但搜索
过程中无法跨车场重分配。本测试锁定修复。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import load_instance
from src.core.solution import Solution
from src.operators.repair import (
    greedy_cost_insertion,
    greedy_emission_insertion,
    random_insertion,
)


@pytest.fixture(scope="module")
def p01_non_tw():
    """green_mdvrp p01: 4 车场 × 4 辆, 非 TW, assigned 全 0。"""
    return load_instance("p01")


def _used_depots(sol: Solution) -> list[int]:
    return [d for d, rl in sol.routes.items() if rl]


def test_greedy_cost_uses_multiple_depots_non_tw(p01_non_tw):
    """greedy_cost_insertion 从空解重建应跨车场 (不只车场 0)。"""
    inst = p01_non_tw
    empty = Solution(inst, {d: [] for d in range(inst.num_depots)})
    rebuilt = greedy_cost_insertion(
        empty, [c.index for c in inst.customers], np.random.default_rng(0))
    used = _used_depots(rebuilt)
    assert len(used) >= 2, f"greedy_cost 只用 {used} 个车场 (assigned 全 0 时应跨车场)"
    # 所有客户覆盖
    assert len(rebuilt.all_served_customers()) == inst.num_customers


def test_greedy_emission_uses_multiple_depots_non_tw(p01_non_tw):
    """greedy_emission_insertion (委托 _insert_cheapest_cost) 也应跨车场。"""
    inst = p01_non_tw
    empty = Solution(inst, {d: [] for d in range(inst.num_depots)})
    rebuilt = greedy_emission_insertion(
        empty, [c.index for c in inst.customers], np.random.default_rng(0))
    used = _used_depots(rebuilt)
    assert len(used) >= 2, f"greedy_emission 只用 {used} 个车场"


def test_random_uses_multiple_depots_non_tw(p01_non_tw):
    """random_insertion 从空解重建应跨车场随机分配。"""
    inst = p01_non_tw
    empty = Solution(inst, {d: [] for d in range(inst.num_depots)})
    rebuilt = random_insertion(
        empty, [c.index for c in inst.customers], np.random.default_rng(0))
    used = _used_depots(rebuilt)
    assert len(used) >= 2, f"random 只用 {used} 个车场"


def test_greedy_cost_moves_customer_to_nearest_depot(p01_non_tw):
    """跨车场重分配: 客户 assigned=0 但距离车场 1 更近, greedy 应跨车场插到最近。

    修复前 _insert_cheapest_cost 读 assigned_depot_index(=0) 只插车场 0,
    即使客户物理上离车场 1 更近也无法跨车场。修复后应插到最近车场。
    """
    inst = p01_non_tw
    dm = inst.distance_matrix
    # 找距离车场 1 明显近于车场 0 的客户
    target = None
    for c in inst.customers:
        ci = c.index
        if dm[1][ci] < dm[0][ci] * 0.5:
            target = ci
            break
    assert target is not None, "p01 应有靠近车场 1 的客户"

    # 强制 assigned=0 (green_mdvrp 原始状态), 空解重插 target
    inst.customers[target - inst.num_depots].assigned_depot_index = 0
    empty = Solution(inst, {d: [] for d in range(inst.num_depots)})
    rebuilt = greedy_cost_insertion(empty, [target], np.random.default_rng(0))

    after_depot = inst.customers[target - inst.num_depots].assigned_depot_index
    nearest = min(range(inst.num_depots), key=lambda d: dm[d][target])
    assert after_depot == nearest, (
        f"客户 {target} 重插到车场 {after_depot}, 最近应为 {nearest}")
    assert len(rebuilt.all_served_customers()) == 1

