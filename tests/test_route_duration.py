"""最大路线时长 (D) 约束修复测试 (规格 docs/durfix-spec-20260911.md §7)。

口径 (唯一定义): duration(r) = 回场时刻 − 首客等待; 约束 duration ≤ D;
另 return_time ≤ 车场最迟返回 tw_late。
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import (
    Customer,
    Depot,
    EmissionModel,
    Instance,
    instance_from_parsed,
    load_instance,
)
from src.core.solution import Solution
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

REPO = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────
# 合成实例 (1 车场 + 3 客户共线)
#
# 车场 (0,0); A(10,0) ready0 s5, B(20,0) ready30 s5, C(30,0) ready0 s5。
# 距离: depot-A=10, depot-B=20, depot-C=30, A-B=10, B-C=10, A-C=20。
# 手工值 (duration = 回场 − 首客等待):
#   [A]=25   [B]=45   [C]=65
#   [A,B]=55 [A,C]=70 [B,A]=50 [B,C]=70 [C,A]=70 [C,B]=70
#   [A,B,C]=80  [B,A,C]=95  [C,B,A]=75
# ─────────────────────────────────────────────────────────────
def _line_instance(route_duration_limit=None, depot_tw_late=None):
    depots = [Depot(index=0, id=0, x=0.0, y=0.0, vehicles_available=3,
                    capacity=100.0, tw_late=depot_tw_late)]
    customers = [
        Customer(index=1, id=1, x=10.0, y=0.0, demand=1.0, assigned_depot_index=0,
                 ready_time=0.0, due_time=1000.0, service_time=5.0),   # A
        Customer(index=2, id=2, x=20.0, y=0.0, demand=1.0, assigned_depot_index=0,
                 ready_time=30.0, due_time=1000.0, service_time=5.0),  # B
        Customer(index=3, id=3, x=30.0, y=0.0, demand=1.0, assigned_depot_index=0,
                 ready_time=0.0, due_time=1000.0, service_time=5.0),   # C
    ]
    nodes = [(d.x, d.y) for d in depots] + [(c.x, c.y) for c in customers]
    n = len(nodes)
    dm = [[math.dist(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]
    return Instance(
        name="line", source="test", num_depots=1, num_customers=3,
        vehicle_capacity=100.0, depots=depots, customers=customers,
        distance_matrix=dm, emission_model=EmissionModel(),
        route_duration_limit=route_duration_limit,
    )


@pytest.fixture(scope="module")
def pr01_instance():
    text = (REPO / "data/mdvrptw_raw/pr01.txt").read_text(encoding="utf-8")
    return instance_from_parsed(parse_mdvrptw(text), name="pr01")


# ─────────────────────────────────────────────────────────────
# 1. duration 公式
# ─────────────────────────────────────────────────────────────
def test_duration_formula_three_wait_types():
    """首客等待 / 中间等待 / 零等待 三型与手工值一致。"""
    inst = _line_instance()
    sol = Solution(inst, {0: [[1, 2, 3]]})
    # [A,B,C]: 首客 A 零等待; 中间 B 等待 5 计入 → dur=80
    assert sol.route_duration(0, [1, 2, 3]) == pytest.approx(80.0)

    # [B,A,C]: 首客 B 等待 10 不计入 → dur=95 (弧80+服务15)
    assert sol.route_duration(0, [2, 1, 3]) == pytest.approx(95.0)

    # [A,C]: 全零等待 → dur = 弧50 + 服务10 = 60? 复核: 弧10+20+30=60, 服务10 → 70
    assert sol.route_duration(0, [1, 3]) == pytest.approx(70.0)


def test_duration_empty_and_single():
    inst = _line_instance()
    sol = Solution(inst, {0: [[], [1]]})
    assert sol.route_duration(0, []) == 0.0
    assert sol.route_duration(0, [1]) == pytest.approx(25.0)


def test_duration_no_limit_returns_zero_violations():
    inst = _line_instance()  # route_duration_limit=None
    sol = Solution(inst, {0: [[1, 2, 3]]})
    assert sol.duration_violations() == []
    assert sol.is_feasible()


def test_duration_pr01_res_matches(pr01_instance):
    """pr01 官方 .res 8 路线 duration 逐位一致 (|diff| < 0.01)。"""
    res_path = REPO / "data/mdvrptw_sol/pr01.res"
    if not res_path.exists():
        pytest.skip("data/mdvrptw_sol/pr01.res 不存在")
    inst = pr01_instance
    nd = inst.num_depots
    sol = Solution(inst, {})
    lines = [ln for ln in res_path.read_text(encoding="utf-8").splitlines()
             if ln.strip()][1:]  # 跳过成本行
    assert len(lines) == 8
    for ln in lines:
        tok = ln.split()
        depot_id = int(tok[0])
        dur_field = float(tok[2])
        # 节点序列: tok[4:] 成对 (node, (time))
        raw_nodes = [int(tok[k]) for k in range(4, len(tok), 2)]
        cust_raw = raw_nodes[1:-1]
        depot_idx = depot_id - 1
        route = [nd + (r - 1) for r in cust_raw]
        got = sol.route_duration(depot_idx, route, depot_id - 1)
        assert abs(got - dur_field) < 0.01, (
            f"depot {depot_id} route dur {got} vs res {dur_field}")


# ─────────────────────────────────────────────────────────────
# 2. is_feasible 超长 → False; 恰好 == D → True
# ─────────────────────────────────────────────────────────────
def test_is_feasible_over_duration_false():
    inst = _line_instance(route_duration_limit=50.0)
    sol = Solution(inst, {0: [[1, 2, 3]]})  # duration 80 > 50
    assert not sol.is_feasible()
    viol = sol.duration_violations()
    assert len(viol) == 1
    assert viol[0]["excess"] == pytest.approx(30.0)


def test_is_feasible_exactly_D_true():
    inst = _line_instance(route_duration_limit=80.0)
    sol = Solution(inst, {0: [[1, 2, 3]]})  # duration == 80
    assert sol.is_feasible()


def test_is_feasible_depot_return_late():
    """车场最迟返回: return_time > tw_late → False。"""
    inst = _line_instance(depot_tw_late=70.0)
    # [1,2,3] return_time=80 > 70 → 不可行
    assert not Solution(inst, {0: [[1, 2, 3]]}).is_feasible()
    # [1,2] return=55, [3] return=65 均 ≤ 70 → 可行 (全客户覆盖)
    assert Solution(inst, {0: [[1, 2], [3]]}).is_feasible()


# ─────────────────────────────────────────────────────────────
# 3. repair: 超长位置不被选; 开新车候选可用
# ─────────────────────────────────────────────────────────────
def test_repair_rejects_overlong_position():
    """只剩超长插入位时改开新车 (D=65; [1,2]=55 可行, 插 3 的任一位置超长)。"""
    inst = _line_instance(route_duration_limit=65.0)
    from src.operators.repair import greedy_cost_tw_insertion

    partial = Solution(inst, {0: [[1, 2]]})
    rng = np.random.default_rng(0)
    sol = greedy_cost_tw_insertion(partial, [3], rng)
    assert sol.all_served_customers() == {1, 2, 3}
    for di, rl in sol.routes.items():
        for ri, r in enumerate(rl):
            assert sol.route_duration(di, r, ri) <= 65.0 + 1e-9
    # 客户 3 必须开自己的新车 (不能并入 [1,2])
    assert all(3 not in r for r in sol.routes[0] if 1 in r or 2 in r)


def test_repair_open_new_route_when_overlong():
    """无 duration 可行插入位时开单车新路线 (D=45; 插 2 进 [1] 仅新车可行)。"""
    inst = _line_instance(route_duration_limit=45.0)
    from src.operators.repair import greedy_cost_tw_insertion

    partial = Solution(inst, {0: [[1]]})
    rng = np.random.default_rng(1)
    sol = greedy_cost_tw_insertion(partial, [2], rng)
    assert sol.all_served_customers() == {1, 2}
    for di, rl in sol.routes.items():
        for ri, r in enumerate(rl):
            assert sol.route_duration(di, r, ri) <= 45.0 + 1e-9
    # 2 必须与 1 分开 (新车)
    assert any(1 in r for r in sol.routes[0]) and any(2 in r for r in sol.routes[0])
    assert not any(1 in r and 2 in r for r in sol.routes[0])


# ─────────────────────────────────────────────────────────────
# 4. LS: 超长候选移动被拒
# ─────────────────────────────────────────────────────────────
def test_ls_rejects_overlong_move():
    """relocate 把 3 并入 [1,2] 会超长 → 拒绝 (D=65; 起点 [1,2]+[3] 均可行)。"""
    inst = _line_instance(route_duration_limit=65.0)
    from src.operators.local_search import relocate_improve

    sol = Solution(inst, {0: [[1, 2], [3]]})
    assert sol.is_feasible()
    out = relocate_improve(sol, max_iterations=5, rng=np.random.default_rng(0))
    for di, rl in out.routes.items():
        for ri, r in enumerate(rl):
            assert out.route_duration(di, r, ri) <= 65.0 + 1e-9
    assert out.all_served_customers() == {1, 2, 3}
    # 3 保持独立 (并入 [1,2] → 80 超长, 必须被拒)
    assert not any(3 in r and (1 in r or 2 in r) for r in out.routes[0])


# ─────────────────────────────────────────────────────────────
# 5. _enforce_duration 客户守恒
# ─────────────────────────────────────────────────────────────
def test_enforce_duration_keeps_conservation():
    inst = _line_instance(route_duration_limit=65.0)
    from src.alns.engine import ALNSEngine

    engine = ALNSEngine(inst, {"max_iterations": 1}, np.random.default_rng(0))
    sol = Solution(inst, {0: [[1, 2, 3]]})  # 超长 80
    out = engine._enforce_duration(sol)
    # 守恒: 修复不丢客户 (可行性由上层判定)
    assert out.all_served_customers() == {1, 2, 3}


def test_enforce_feasibility_duration(pr01_instance):
    """注入超长路线 → _enforce_feasibility 后守恒 (客户不丢)。"""
    inst = pr01_instance
    from src.alns.engine import ALNSEngine

    engine = ALNSEngine(inst, {"max_iterations": 1}, np.random.default_rng(0))
    all_c = [c.index for c in inst.customers]
    # 把前 12 个客户全塞进 depot 0 一条路由 (必超长 D=500)
    sol = Solution(inst, {0: [all_c[:12]]})
    out = engine._enforce_feasibility(sol)
    assert out.all_served_customers() >= sol.all_served_customers()


# ─────────────────────────────────────────────────────────────
# 6. 无 D 实例行为逐位不变
# ─────────────────────────────────────────────────────────────
def test_no_D_instance_legacy_unchanged():
    inst = load_instance("p01")  # green_mdvrp 无 D
    assert inst.route_duration_limit is None
    assert all(d.tw_late is None for d in inst.depots)
    from src.core.initial_solution import nearest_neighbor

    sol = nearest_neighbor(inst, np.random.default_rng(42))
    assert sol.duration_violations() == []
    assert sol.is_feasible()


# ─────────────────────────────────────────────────────────────
# 7. 确定性: 同 seed 两次逐位一致
# ─────────────────────────────────────────────────────────────
def test_deterministic_same_seed(pr01_instance):
    from src.alns.engine import ALNSEngine
    from src.operators.repair import greedy_cost_tw_insertion
    from src.operators.destroy import DESTROY_OPERATORS
    from src.alns.selection import SegmentRewardSelection

    def run(seed):
        inst = instance_from_parsed(
            parse_mdvrptw((REPO / "data/mdvrptw_raw/pr01.txt")
                          .read_text(encoding="utf-8")), name="pr01")
        sel = SegmentRewardSelection(
            list(DESTROY_OPERATORS.keys()), ["greedy_cost_tw"],
            segment_size=100, reaction_factor=0.1, decay_factor=1.0,
            min_selection_prob=0.005)
        engine = ALNSEngine(inst, {
            "max_iterations": 30, "stagnation_limit": 30,
            "local_search_final": False,
        }, np.random.default_rng(seed), selector=sel,
            repair_ops={"greedy_cost_tw": greedy_cost_tw_insertion})
        arch = engine.run()
        feas = [(e[0][0], e[1]["solution"]) for e in arch.entries
                if e[1]["solution"].is_feasible()]
        if not feas:
            return None
        best = min(feas, key=lambda t: t[0])
        return (round(best[0], 10), tuple(sorted(
            (di, tuple(r)) for di, rl in best[1].routes.items()
            for r in rl)))

    a = run(7)
    b = run(7)
    assert a is not None and a == b


# ─────────────────────────────────────────────────────────────
# 8. stress: 人为收紧 D 修复链有界不挂死
# ─────────────────────────────────────────────────────────────
def test_stress_tight_D_bounded():
    """极紧 D → _enforce_duration 迭代有界返回 (不挂死), 客户守恒。"""
    inst = _line_instance(route_duration_limit=0.001)
    from src.alns.engine import ALNSEngine

    engine = ALNSEngine(inst, {"max_iterations": 1}, np.random.default_rng(0))
    sol = Solution(inst, {0: [[1, 2, 3]]})
    out = engine._enforce_duration(sol)
    assert out.all_served_customers() == {1, 2, 3}
