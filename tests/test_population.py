"""种群壳 (M1, 2026-09-07) 回归测试 — 轴2 Memetic-ALNS 里程碑 1。

机制 = config 键 population_mode (默认 False): μ 成员池 + 锦标赛选择
(fitness = cost − κ×池内最优成本×dc) + 池级停滞刷新最劣成员。M1 无重组/
教育/罚域。默认关 → 行为逐位不变 (既有全套测试回归锁存; test_default_off
另加缺省断言)。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace as dc_replace

from src.core.instance import Customer, Depot, EmissionModel, Instance
from src.core.objectives import calculate_objectives
from src.core.solution import Solution
from src.alns.engine import (
    ALNSEngine,
    pool_accept_replace,
    pool_gate_improve,
    pool_penalty_gate,
    pool_penalized_cost,
    _member_refresh_threshold,
    _pool_member_dc,
    _pool_select_member,
)
from src.alns.archive import ParetoArchive
from src.alns.selection import SegmentRewardSelection
from src.operators.local_search import rvnd_improve
from src.operators.destroy import DESTROY_OPERATORS, random_removal
from src.operators.repair import greedy_cost_tw_insertion
from src.operators.crossover import partition_crossover

from tests.test_soft_capacity import LOCK_CONFIG


# ── 合成 TW 样本: 2 车场 × 3 车 (容量 100), 6 客户 (两簇各 3, 宽窗 due=500)
#    全局客户号 2..7 (客户 i = i + num_depots)。任意路由 TW 可行 (宽窗),
#    多车 + 宽容 → 随机初始解产生多种路由集互异结构。
def _tw_inst():
    depots = [Depot(0, 0, 0.0, 0.0, 3, 100.0), Depot(1, 1, 60.0, 0.0, 3, 100.0)]
    custs = []
    # 簇 A (近 depot0): x = 4, 8, 12
    for i, dx in enumerate((4.0, 8.0, 12.0)):
        custs.append(Customer(2 + i, 2 + i, dx, 2.0, 10.0, 0,
                              ready_time=0.0, due_time=500.0, service_time=1.0))
    # 簇 B (近 depot1): x = 52, 56, 60
    for i, dx in enumerate((52.0, 56.0, 60.0)):
        custs.append(Customer(5 + i, 5 + i, dx, 2.0, 10.0, 0,
                              ready_time=0.0, due_time=500.0, service_time=1.0))
    nodes = [(0.0, 0.0), (60.0, 0.0)] + [(c.x, c.y) for c in custs]
    n = len(nodes)
    dist = [[__import__("math").dist(nodes[i], nodes[j]) for j in range(n)]
            for i in range(n)]
    return Instance("pop_tw", "test", 2, 6, 100.0, depots, custs, dist,
                    EmissionModel(base_emission_rate=0.0,
                                  load_correction_alpha=0.0))


@pytest.fixture(scope="module")
def tw_instance():
    return _tw_inst()


def _fresh(inst):
    """深拷贝客户列表 — 引擎会改 assigned_depot_index (共享可变状态)。"""
    return dc_replace(inst, customers=[dc_replace(c) for c in inst.customers])


def _make_engine(inst, config, seed, acceptor=None):
    """装配 TW 实例引擎 (与 solve_standard_alns / 既有机制测试同构), 不 run。"""
    inst = _fresh(inst)
    destroy_names = list(DESTROY_OPERATORS.keys())
    selector = SegmentRewardSelection(
        destroy_names, ["greedy_cost_tw"],
        segment_size=config.get("segment_size", 100),
        reaction_factor=config.get("reaction_factor", 0.1),
        decay_factor=config.get("decay_factor", 1.0),
        min_selection_prob=config.get("min_selection_prob", 0.005),
    )
    return ALNSEngine(
        instance=inst, config=config, rng=np.random.default_rng(seed),
        selector=selector, acceptor=acceptor,
        repair_ops={"greedy_cost_tw": greedy_cost_tw_insertion},
        local_search=rvnd_improve,
    )


def _run_engine(inst, config, seed, acceptor=None):
    engine = _make_engine(inst, config, seed, acceptor=acceptor)
    archive = engine.run()
    return archive, engine


def _pop_cfg(**overrides):
    """种群臂基线配置: 随机初始池 (成员互异可验证) + population 键 μ=4。"""
    cfg = dict(LOCK_CONFIG, initial_solution="random",
               population_mode=True, population_size=4,
               population_tournament=2, population_dc_kappa=0.0)
    cfg.update(overrides)
    return cfg


def _routes(sol):
    return {d: [list(r) for r in rl] for d, rl in sol.routes.items()}


class _RejectAcceptor:
    """全拒 acceptor (无 _auto_calibrate): accept 恒 False, 池永不因接受而替换。"""
    def accept(self, current_obj, new_obj, rng):
        return False

    def cool(self):
        pass

    def reset(self):
        pass


# ── ① config gating: 默认关 ──────────────────────────────────

def test_default_off_attributes(tw_instance):
    """config 无 population 键 → population_mode False / stats None / final None。

    键集断言: 默认路径不动 population_stats / population_final (既有全套
    pytest 回归 = gated 生效证据)。
    """
    engine = _make_engine(tw_instance, dict(LOCK_CONFIG), seed=42)
    assert engine.population_mode is False
    assert engine.population_size == 8
    assert engine.population_tournament == 2
    assert engine.population_dc_kappa == 0.0
    assert engine.population_stats is None
    assert engine.population_final is None
    engine.run()
    assert engine.population_stats is None
    assert engine.population_final is None


# ── ② 非法组合守卫 ──────────────────────────────────────────

_FORBIDDEN = [
    {"population_crossover": True},
    {"educate_mode": "deep"},
    {"rebalance_mode": "removal"},
    {"soft_capacity": True},
    {"soft_tw": True},
    {"stagnation_source": "current"},  # 裁定 3: 池级停滞只实现 archive 语义
]


def test_forbidden_combos_raise_when_on(tw_instance):
    """population_mode=True 叠加 M1 未测组合 → ValueError。"""
    for ov in _FORBIDDEN:
        cfg = dict(LOCK_CONFIG, population_mode=True, **ov)
        with pytest.raises(ValueError, match="population_mode M1 forbids"):
            _make_engine(tw_instance, cfg, 42)


def test_forbidden_combos_do_not_raise_when_off(tw_instance):
    """默认关路径 (无 population_mode) 不触发守卫。"""
    for ov in _FORBIDDEN:
        cfg = dict(LOCK_CONFIG, **ov)
        _make_engine(tw_instance, cfg, 42)  # 不抛


# ── 裁定 1 纯函数: 两级接受 ──────────────────────────────────

def test_pool_accept_replace_locks_two_level_acceptance():
    """裁定 1: 成员替换 = SA 接受 AND 子代可行 (池成员只允许可行解)。

    (T,T)=True / (T,F)=False (SA 接受但不可行 → 按拒绝记账) /
    (F,T)=False (SA 拒绝不改池)。决策点禁内联绕过 — 集成行为由该单测锁定。
    """
    assert pool_accept_replace(True, True) is True
    assert pool_accept_replace(True, False) is False
    assert pool_accept_replace(False, True) is False


# ── ③④ 锦标赛选择纯函数 (单元级) ────────────────────────────

def test_tournament_pure_cost_selects_lower_of_sampled():
    """κ=0 → fitness=成本; 先抽 min(tournament,μ) 子集再取其中成本低者。

    抽中集合由固定 seed 复现 (rng.choice 同构), 断言胜者 = 抽中子集成本
    最小 (而非全池最小) — 锁存"先抽子集再比 fitness"的抽样语义。
    """
    costs = [10.0, 20.0, 30.0]
    keys = [frozenset(), frozenset(), frozenset()]  # κ=0 → dc 不参与
    rng = np.random.default_rng(0)
    winner = _pool_select_member(costs, keys, best_pool_cost=10.0,
                                 kappa=0.0, tournament=2, rng=rng)
    drawn = np.random.default_rng(0).choice(3, size=2, replace=False)
    assert winner in [int(i) for i in drawn]
    assert costs[winner] == min(costs[int(i)] for i in drawn)


def test_tournament_dc_prefers_diverse_over_clone():
    """两成员路由完全相同 (低成本) + 一成员路由不同 (成本略高)。

    κ=2 → fitness = cost − κ×best×dc; 不同路由成员 dc=1 → fitness 最低 → 选中;
    κ=0 → 纯成本并列取序号小者 → 低成本成员 0 选中。
    """
    inst = _tw_inst()
    m0 = Solution(inst, {0: [[2, 3, 4]], 1: [[5, 6, 7]]})
    m1 = Solution(inst, {0: [[2, 3, 4]], 1: [[5, 6, 7]]})
    m2 = Solution(inst, {0: [[5, 6, 7]], 1: [[2, 3, 4]]})
    keys = [ParetoArchive._route_keys({"solution": s}) for s in (m0, m1, m2)]
    costs = [10.0, 10.0, 11.0]
    # dc 语义锚定: m0/m1 互为克隆 dc=0; m2 与二者路由集不相交 dc=1
    assert _pool_member_dc(0, keys) == pytest.approx(0.0)
    assert _pool_member_dc(2, keys) == pytest.approx(1.0)
    # κ=0 → 纯成本
    assert _pool_select_member(costs, keys, 10.0, 0.0, 3,
                               np.random.default_rng(0)) == 0
    # κ=2 → 多样性贡献进 fitness, 选不同路由成员
    assert _pool_select_member(costs, keys, 10.0, 2.0, 3,
                               np.random.default_rng(0)) == 2


# ── ⑤ 池构建: μ=4 全部可行 / 路由集互异 / 档案初始播种 1 条 ──

def test_pool_build_feasible_distinct_seeded(tw_instance):
    """max_iterations=0: 池建好后即收尾 → population_final = 初始池。

    全部可行; 固定 seed 下路由集互异 (随机初始池); 档案只含初始播种 1 条
    (type=population_init, 可行)。
    """
    archive, engine = _run_engine(
        tw_instance, _pop_cfg(max_iterations=0), seed=42)
    finals = engine.population_final
    assert finals is not None and len(finals) == 4
    assert engine.population_stats["init_members"] == 4
    keysets = []
    for obj, sol in finals:
        assert obj[0] > 0
        assert sol.is_feasible()
        assert len(sol.tw_violations()) == 0
        assert len(sol.all_served_customers()) == 6
        keysets.append(ParetoArchive._route_keys({"solution": sol}))
    for i in range(4):
        for j in range(i + 1, 4):
            assert keysets[i] != keysets[j], "池成员路由集必须互异"
    assert archive.size == 1
    for e in archive.entries:
        assert e[1]["type"] == "population_init"
        assert e[1]["solution"].is_feasible()


# ── ⑥ 集成小跑: 完整 / 档案纯可行 / 轨迹=iter / 确定性 ──────

def test_population_integration_runs_deterministic(tw_instance):
    """μ=4, max_iterations=40: 运行完整 + 档案纯可行 + 轨迹长度=40 +
    population_stats 键齐 + last_solution 可行 + 同 seed 重跑逐位相同。"""
    cfg = _pop_cfg(max_iterations=40)
    results = []
    for _ in range(2):
        archive, engine = _run_engine(tw_instance, dict(cfg), seed=42)
        s = engine.population_stats
        assert s["init_members"] == 4
        assert s["refresh_events"] == 0
        assert s["member_refresh_events"] == 0
        assert set(s) == {"init_members", "refresh_events",
                          "member_refresh_events", "member_advances",
                          "dc_min_mean",
                          "timefix_bootstrap_attempts",
                          "timefix_bootstrap_success"}  # timefix 键 (2026-09-12)
        assert set(s["member_advances"]) == {0, 1, 2, 3}
        assert all(v >= 0 for v in s["member_advances"].values())
        assert 0.0 <= s["dc_min_mean"] <= 1.0
        assert len(engine.trajectory) == 40
        assert engine.last_solution is not None
        assert engine.last_solution.is_feasible()
        assert len(engine.population_final) == 4
        for e in archive.entries:
            sol = e[1]["solution"]
            assert sol.is_feasible()
            assert len(sol.tw_violations()) == 0
        results.append((archive.best_cost(),
                        [r["outcome"]["cost"] for r in engine.trajectory]))
    assert results[0][0] == pytest.approx(results[1][0], abs=1e-9)
    assert results[0][1] == results[1][1]


# ── ⑦ 拒绝不改池 (destroy/repair 链无原地修改) ───────────────

def test_reject_leaves_pool_unchanged(tw_instance):
    """全拒 acceptor + 40 iter: 池成员与初始池逐位相同 (解/obj 均不变),
    population_final 同 — 证明 destroy/repair/LS 链对成员无原地修改。"""
    _, eng_init = _run_engine(tw_instance, _pop_cfg(max_iterations=0), seed=7)
    initial = eng_init.population_final
    assert initial is not None

    cfg = _pop_cfg(stagnation_limit=6000, max_iterations=40)
    archive, engine = _run_engine(tw_instance, cfg, seed=7,
                                  acceptor=_RejectAcceptor())
    final = engine.population_final
    assert engine.population_stats["refresh_events"] == 0
    assert engine.population_stats["member_advances"] == {0: 0, 1: 0, 2: 0, 3: 0}
    assert len(final) == len(initial)
    for (o0, s0), (o1, s1) in zip(initial, final):
        assert o0 == o1
        assert _routes(s0) == _routes(s1)
    assert archive.size >= 1  # 初始播种在


# ── ⑧ 刷新路径 ──────────────────────────────────────────────

def test_pool_refresh_path_triggered(tw_instance):
    """全拒 acceptor + 小 stagnation_limit + 300 iter → refresh_events ≥ 1。

    档案容量 1 (饱和后劣解不入档 → 池级停滞单调累计到 recovery 阈值),
    运行完整不崩, 档案非空 (初始播种在)。
    """
    cfg = _pop_cfg(archive_capacity=1, stagnation_limit=200,
                   max_iterations=300)
    archive, engine = _run_engine(tw_instance, cfg, seed=42,
                                  acceptor=_RejectAcceptor())
    assert engine.population_stats["refresh_events"] >= 1
    assert engine.population_stats["init_members"] == 4
    assert len(engine.trajectory) == 300
    assert len(engine.population_final) == 4
    assert archive.size >= 1
    for e in archive.entries:
        assert e[1]["solution"].is_feasible()


# ── ⑨ Solution.apply_destroy/apply_repair 原地性确认 ─────────

def test_destroy_repair_do_not_mutate_input(tw_instance):
    """destroy/repair 链作用于副本: 输入解路由在两次调用后逐位不变。"""
    inst = tw_instance
    sol = Solution(inst, {0: [[2, 3, 4]], 1: [[5, 6, 7]]})
    before = _routes(sol)
    rng = np.random.default_rng(3)
    partial, removed = sol.apply_destroy(random_removal, 2, rng)
    assert _routes(sol) == before, "destroy 不得原地修改输入解"
    assert len(removed) == 2
    assert all(2 <= ci < 2 + inst.num_customers for ci in removed)
    repaired = partial.apply_repair(greedy_cost_tw_insertion, removed, rng)
    assert _routes(sol) == before, "repair 不得原地修改输入解"
    assert repaired.is_feasible()


# ── 裁定 2 (2026-09-07): 成员级停滞刷新 ──────────────────────

def test_member_refresh_threshold_formula():
    """member_refresh_threshold = max(50, stagnation_limit // 30) 公式三值。

    6000iter BASE (stagnation_limit=6000) → 200; 临界 1500 → 50; 短预算
    (stagnation_limit=90) → max(50, 3) = 50。公式写死, 不加 config 键。
    """
    assert _member_refresh_threshold(6000) == 200
    assert _member_refresh_threshold(1500) == 50
    assert _member_refresh_threshold(90) == 50


def test_member_refresh_evicts_stale_member_from_archive(tw_instance):
    """驱逐集成测: archive ≥2 可行条目, 白盒把某成员 member_stagnation 拨到
    ≥ 阈值 → 跑 1 迭代 → 该成员被档案精英替换 (obj = 某档案条目 obj)、
    member_stagnation=0、member_refresh_events=1。

    全拒 acceptor 隔离迭代体内可行子代替换 (成员刷新后不被再次替换), 使断言
    精确锁定驱逐路径。
    """
    cfg = _pop_cfg(stagnation_limit=6000, max_iterations=1)
    engine = _make_engine(tw_instance, cfg, seed=42, acceptor=_RejectAcceptor())
    inst = engine.instance  # 引擎深拷贝后的实例 (与池成员同一引用域)
    # 预置档案 ≥2 可行条目 (成本不同; single_objective 拒等成本重复)
    for routes in ({0: [[2, 3, 4]], 1: [[5, 6, 7]]},
                   {0: [[2, 3], [4]], 1: [[5, 6, 7]]}):
        sol = Solution(inst, routes)
        assert sol.is_feasible()
        engine.archive.add(calculate_objectives(sol),
                           {"type": "test", "solution": sol.copy()})
    assert engine.archive.size >= 2
    thr = _member_refresh_threshold(cfg["stagnation_limit"])  # 6000 → 200
    assert thr == 200
    engine._member_stagnation_override = [thr, 0, 0, 0]
    engine.run()
    assert engine.population_stats["member_refresh_events"] == 1
    assert engine._member_stagnation[0] == 0
    final0_obj, final0_sol = engine.population_final[0]
    assert final0_sol.is_feasible()
    # 成员 0 被档案精英 (某档案条目) 替换
    assert any(abs(final0_obj[0] - e[0][0]) < 1e-9 for e in engine.archive.entries)


def test_member_refresh_disable_switch(tw_instance):
    """C4 对照臂开关 (2026-09-09): population_member_refresh=False 关闭成员级
    停滞驱逐 (member_refresh_events=0, 停滞成员不被档案精英替换); 缺省/True =
    旧行为逐位不变 (驱逐发生, 与 test_member_refresh_evicts_stale_member_
    from_archive 同构)。"""
    thr = 200

    def run(member_refresh):
        cfg = _pop_cfg(stagnation_limit=6000, max_iterations=1)
        if member_refresh is not None:
            cfg["population_member_refresh"] = member_refresh
        engine = _make_engine(tw_instance, cfg, seed=42,
                              acceptor=_RejectAcceptor())
        inst = engine.instance
        for routes in ({0: [[2, 3, 4]], 1: [[5, 6, 7]]},
                       {0: [[2, 3], [4]], 1: [[5, 6, 7]]}):
            sol = Solution(inst, routes)
            assert sol.is_feasible()
            engine.archive.add(calculate_objectives(sol),
                               {"type": "test", "solution": sol.copy()})
        engine._member_stagnation_override = [thr, 0, 0, 0]
        engine.run()
        return engine

    off = run(False)
    assert off.population_stats["member_refresh_events"] == 0
    # 停滞照常推进且未被清零 (未驱逐): override 200 → 200+1
    assert off._member_stagnation[0] == thr + 1

    assert run(None).population_stats["member_refresh_events"] == 1
    assert run(True).population_stats["member_refresh_events"] == 1


def test_member_refresh_archive_empty_falls_back(tw_instance):
    """档案空兜底: archive 清空 → 成员刷新走 _build_pool_member 不崩。

    白盒直驱刷新源 _member_refresh_replacement (生产成员级扫描同源逻辑):
    合成样本初始池恒可行 → run 内初始播种总会给 archive 补 ≥1 条, 空 archive
    的扫描分支在该 fixture 上不可达; 此处锁定空档案分支 = _build_pool_member。
    """
    engine = _make_engine(tw_instance, _pop_cfg(max_iterations=1), seed=42)
    engine.archive.clear()
    assert engine._select_parent() is None  # 档案空 → 精英路径 None
    sol, obj = engine._member_refresh_replacement()
    assert isinstance(sol, Solution)
    assert obj[0] > 0
    assert len(sol.all_served_customers()) == tw_instance.num_customers


# ══════════════════════════════════════════════════════════════
# M2 池内重组 (2026-09-07): population_cx_mode route|partition
# 规格: docs/ma-alns-m2-spec-20260907.md。A 臂 = 路由级继承
# (route_copy_crossover); B 臂 = 车场簇继承 (partition_crossover)。
# 默认 none → M1 行为逐位不变。全部随机走 engine.rng; 硬币序固定。
# ══════════════════════════════════════════════════════════════

_CX_M1_KEYS = {"init_members", "refresh_events", "member_refresh_events",
               "member_advances", "dc_min_mean",
               "timefix_bootstrap_attempts", "timefix_bootstrap_success"}  # timefix 键 (2026-09-12)
_CX_KEYS = {"cx_attempts", "cx_children", "cx_duplicates",
            "cx_novel_route_keys"}
_CX_FORBIDDEN = [
    {"population_crossover": True},
    {"educate_mode": "deep"},
    {"rebalance_mode": "removal"},
    {"soft_capacity": True},
    {"soft_tw": True},
    {"stagnation_source": "current"},
]


class _SeqRng:
    """硬币序列注入 (partition 只消费 rng.random(), 逐次返回预设值)。"""
    def __init__(self, seq):
        self._seq = list(seq)
        self._i = 0

    def random(self):
        v = self._seq[self._i]
        self._i += 1
        return v


def _all_at_depot(inst, d):
    """全客户布局: 全部客户集中在单个车场 d 的一条路由 (TW 宽窗 → 可行)。"""
    custs = sorted(c.index for c in inst.customers)
    return Solution(inst, {d: [custs], 1 - d: []})


def _cx_cfg(**overrides):
    cfg = _pop_cfg(max_iterations=40, population_cx_rate=0.2)
    cfg.update(overrides)
    return cfg


# ── M2 config gating: 默认 none 与键默认值 ──────────────────

def test_population_cx_default_attrs(tw_instance):
    """config 无 cx 键 → mode none / rate 0.2 / inherit_prob 0.6。"""
    engine = _make_engine(tw_instance, dict(LOCK_CONFIG), seed=42)
    assert engine.population_cx_mode == "none"
    assert engine.population_cx_rate == pytest.approx(0.2)
    assert engine.population_cx_inherit_prob == pytest.approx(0.6)


def test_population_cx_default_none_no_cx_keys(tw_instance):
    """mode 缺省 none: population_stats 只含 M1 键, 无 cx_* 键。"""
    _, engine = _run_engine(tw_instance, _pop_cfg(max_iterations=0), seed=42)
    assert set(engine.population_stats) == _CX_M1_KEYS
    assert not (_CX_KEYS & set(engine.population_stats))


# ── M2 守卫 ────────────────────────────────────────────────

def test_population_cx_requires_population_mode(tw_instance):
    """mode ≠ none 且 population_mode=False → ValueError。"""
    for mode in ("route", "partition"):
        cfg = dict(LOCK_CONFIG, population_cx_mode=mode)
        with pytest.raises(ValueError):
            _make_engine(tw_instance, cfg, 42)


def test_population_cx_mode_invalid_value(tw_instance):
    """非法 mode 值 → ValueError (消息含 population_cx_mode)。"""
    cfg = dict(LOCK_CONFIG, population_mode=True, population_cx_mode="bogus")
    with pytest.raises(ValueError, match="population_cx_mode"):
        _make_engine(tw_instance, cfg, 42)


def test_population_cx_mode_forbids_legacy_combos(tw_instance):
    """mode ≠ none 叠加 M2 未测禁开的 legacy 机制 → ValueError。"""
    for mode in ("route", "partition"):
        for ov in _CX_FORBIDDEN:
            cfg = dict(LOCK_CONFIG, population_mode=True,
                       population_cx_mode=mode, **ov)
            with pytest.raises(ValueError):
                _make_engine(tw_instance, cfg, 42)


def test_population_cx_legacy_ok_when_mode_none(tw_instance):
    """默认 none 路径不触发 M2 守卫 (与 M1 各自的守卫正交)。"""
    for ov in _CX_FORBIDDEN:
        cfg = dict(LOCK_CONFIG, **ov)
        _make_engine(tw_instance, cfg, 42)  # 不抛


# ── B 臂算子单元: partition_crossover ──────────────────────

def test_partition_crossover_unit_feasible_deterministic():
    """B 臂单元: A 全 depot0 + B 全 depot1 → 子代覆盖全集且可行;
    同 seed 重跑路由逐位一致 (确定性)。"""
    inst = _tw_inst()
    a = _all_at_depot(inst, 0)
    b = _all_at_depot(inst, 1)
    assert a.is_feasible() and b.is_feasible()
    outs = []
    for _ in range(2):
        child = partition_crossover(a, b, np.random.default_rng(7))
        assert child.is_feasible()
        assert len(child.all_served_customers()) == inst.num_customers
        outs.append(_routes(child))
    assert outs[0] == outs[1]


def test_partition_crossover_duplicates_count_semantics():
    """修复池计数 (= 跨父本重复客户数): A 全 d0 + B 全 d1 且硬币 A@d0/B@d1
    → 全部客户跨父本重复 (duplicates = 6); 硬币全 A → 0。算子层经
    child._cx_duplicates 暴露。"""
    inst = _tw_inst()
    a = _all_at_depot(inst, 0)
    b = _all_at_depot(inst, 1)
    child = partition_crossover(a, b, _SeqRng([0.4, 0.7]))  # A@d0, B@d1
    assert child._cx_duplicates == inst.num_customers
    assert child.is_feasible()
    child2 = partition_crossover(a, b, _SeqRng([0.4, 0.4]))  # 全偏 A
    assert child2._cx_duplicates == 0
    assert _routes(child2) == _routes(a)


def test_partition_crossover_inherit_purity_all_a():
    """继承纯度: 硬币全偏 A → 子代 depot0 路由 = A 的 depot0 路由逐条同序。"""
    inst = _tw_inst()
    a = _all_at_depot(inst, 0)
    b = _all_at_depot(inst, 1)
    child = partition_crossover(a, b, _SeqRng([0.4, 0.4]))
    assert child.routes[0] == [list(r) for r in a.routes[0]]
    assert child.routes[1] == []
    assert child.is_feasible()


def test_partition_crossover_light_bound_mixed_coins():
    """硬币随机 (四种组合全扫): 子代可行 + 每车场路由数 ≤ 两父本该车场
    路由数之和 (轻断言, 不经修复池重插验证结构继承)。"""
    inst = _tw_inst()
    a = _all_at_depot(inst, 0)
    b = _all_at_depot(inst, 1)
    for seq in ([0.4, 0.4], [0.4, 0.7], [0.7, 0.4], [0.7, 0.7]):
        child = partition_crossover(a, b, _SeqRng(list(seq)))
        assert child.is_feasible()
        assert len(child.all_served_customers()) == inst.num_customers
        for d in range(inst.num_depots):
            assert len(child.routes[d]) <= len(a.routes[d]) + len(b.routes[d])


# ── 集成: pop + route / pop + partition ────────────────────

def _assert_cx_integration(tw_instance, cfg, route_duplicates_zero):
    results = []
    for _ in range(2):
        archive, engine = _run_engine(tw_instance, dict(cfg), seed=42)
        s = engine.population_stats
        assert _CX_M1_KEYS <= set(s)
        assert _CX_KEYS <= set(s)
        assert s["cx_attempts"] >= 1, "rate 命中 + 双亲齐备必须发生"
        assert s["cx_children"] >= 1
        assert isinstance(s["cx_attempts"], int)
        assert isinstance(s["cx_children"], int)
        assert isinstance(s["cx_duplicates"], int)
        assert isinstance(s["cx_novel_route_keys"], int)
        if route_duplicates_zero:
            assert s["cx_duplicates"] == 0, "route 臂恒 0"
        assert 0 <= s["cx_duplicates"]
        assert s["cx_novel_route_keys"] >= 0
        assert engine.last_solution is not None
        assert engine.last_solution.is_feasible()
        assert len(engine.population_final) == 4
        for e in archive.entries:
            sol = e[1]["solution"]
            assert sol.is_feasible()
            assert len(sol.tw_violations()) == 0
        results.append((archive.best_cost(),
                        [r["outcome"]["cost"] for r in engine.trajectory],
                        dict(s)))
    assert results[0][0] == pytest.approx(results[1][0], abs=1e-9)
    assert results[0][1] == results[1][1]
    assert results[0][2] == results[1][2]


def test_population_cx_route_integration_runs_deterministic(tw_instance):
    """pop + route 40 iter (μ=4): 完成 / 档案可行 / last_solution 可行 /
    cx_attempts ≥ 1 / cx 键齐 / 同 seed 重跑逐位同 (含 population_stats)。"""
    _assert_cx_integration(
        tw_instance,
        _cx_cfg(population_cx_mode="route", population_cx_inherit_prob=0.6),
        route_duplicates_zero=True)


def test_population_cx_partition_integration_runs_deterministic(tw_instance):
    """pop + partition 40 iter (μ=4): 同上 (partition 臂 cx_duplicates 累计
    修复池客户, 不设死值)。"""
    _assert_cx_integration(
        tw_instance,
        _cx_cfg(population_cx_mode="partition"),
        route_duplicates_zero=False)


# ── cx 子代路径无原地修改 ──────────────────────────────────

def test_cx_rejected_child_leaves_pool_unchanged(tw_instance):
    """cx 触发 (rate=1) + 全拒 acceptor → 池成员与初始池逐位相同 (解/obj
    均不变) — cx 子代生成 (route 与 partition) 对池成员无原地修改。"""
    _, eng_init = _run_engine(tw_instance, _pop_cfg(max_iterations=0), seed=7)
    initial = eng_init.population_final
    assert initial is not None
    for mode in ("route", "partition"):
        cfg = _pop_cfg(stagnation_limit=6000, max_iterations=40,
                       population_cx_mode=mode, population_cx_rate=1.0)
        archive, engine = _run_engine(tw_instance, cfg, seed=7,
                                      acceptor=_RejectAcceptor())
        assert engine.population_stats["cx_attempts"] >= 1
        assert engine.population_stats["cx_children"] >= 1
        assert engine.population_stats["refresh_events"] == 0
        assert engine.population_stats["member_advances"] == {0: 0, 1: 0, 2: 0, 3: 0}
        final = engine.population_final
        assert len(final) == len(initial)
        for (o0, s0), (o1, s1) in zip(initial, final):
            assert o0 == o1
            assert _routes(s0) == _routes(s1)
        assert archive.size >= 1  # 初始播种在


# ══════════════════════════════════════════════════════════════
# M3 池内重组改进门+教育 (2026-09-07): population_cx_gate improve +
# population_cx_educate (预算竞争修复)。规格: docs/ma-alns-m3-spec-20260907.md。
# gate="improve": cx 子代纯成本严格改进 m* 才替换 (跳过本迭代 destroy/repair);
#   否则 cx_gate_fail 累计并回退执行正常 destroy/repair 迭代 (selector 更新照常)。
# educate=True (仅 gate=improve+partition): 子代先进池内破坏性教育再进门。
# 默认 none/False → M2 行为逐位不变 (键集断言 + 全量 pytest 回归)。
# ══════════════════════════════════════════════════════════════

_CX_GATE_KEYS = {"cx_gate_success", "cx_gate_fail"}
_EDUCATE_KEYS = {"educate_attempts", "educate_success", "educate_delta_sum"}
_M3_DEFAULT_KEYS = _CX_M1_KEYS | _CX_KEYS


def _m3_cfg(**overrides):
    """M3 臂基线: partition + gate=improve + rate=1.0 (强制每迭代 cx 尝试) +
    高停滞上限 → gate/回退分支覆盖率最大化。educate 由用例显式开启。"""
    cfg = _cx_cfg(population_cx_mode="partition", population_cx_gate="improve",
                  population_cx_rate=1.0, stagnation_limit=100000)
    cfg.update(overrides)
    return cfg


# ── M3 纯函数: pool_gate_improve ─────────────────────────────

def test_pool_gate_improve_pure_function_locks_gate():
    """改进门 = 可行 AND 纯成本严格改进 m* (差 > 1e-9)。三用例锁定:
    (改进, 可行)→True; (等成本, 可行)→False; (改进, 不可行)→False。"""
    assert pool_gate_improve(100.0, 101.0, True) is True
    assert pool_gate_improve(100.0, 100.0, True) is False
    assert pool_gate_improve(100.0, 101.0, False) is False
    # 差 ≤ 1e-9 不算严格改进; 劣化更不进
    assert pool_gate_improve(101.0 - 1e-10, 101.0, True) is False
    assert pool_gate_improve(102.0, 101.0, True) is False


# ── M3 config 键默认值 ───────────────────────────────────────

def test_population_cx_m3_default_attrs(tw_instance):
    """config 无 M3 键 → gate none / educate False / burst 0.20 / ls_budget 50。"""
    engine = _make_engine(tw_instance, dict(LOCK_CONFIG), seed=42)
    assert engine.population_cx_gate == "none"
    assert engine.population_cx_educate is False
    assert engine.population_educate_burst_ratio == pytest.approx(0.20)
    assert engine.population_educate_ls_budget == 50


# ── M3 守卫 ─────────────────────────────────────────────────

def test_population_cx_m3_gate_improve_guard(tw_instance):
    """gate=improve 只与 partition 配对测过: mode≠partition → ValueError;
    gate≠none 且 population_mode=False → ValueError (并入既有 cx 守卫)。"""
    for mode in ("none", "route"):
        cfg = dict(LOCK_CONFIG, population_mode=True, population_cx_mode=mode,
                   population_cx_gate="improve")
        with pytest.raises(ValueError, match="population_cx_gate"):
            _make_engine(tw_instance, cfg, 42)
    # gate=improve + mode=partition 但 population_mode=False → 既有 cx 守卫
    cfg = dict(LOCK_CONFIG, population_cx_mode="partition",
               population_cx_gate="improve")
    with pytest.raises(ValueError):
        _make_engine(tw_instance, cfg, 42)


def test_population_cx_m3_educate_guard(tw_instance):
    """educate=True 只与 gate=improve+partition 配对: gate=none 或 mode≠partition
    → ValueError。"""
    for mode in ("none", "route", "partition"):
        cfg = dict(LOCK_CONFIG, population_mode=True, population_cx_mode=mode,
                   population_cx_educate=True)
        with pytest.raises(ValueError, match="population_cx_educate"):
            _make_engine(tw_instance, cfg, 42)
    # educate=True + gate=improve + mode=route → gate 守卫先抛
    cfg = dict(LOCK_CONFIG, population_mode=True, population_cx_mode="route",
               population_cx_gate="improve", population_cx_educate=True)
    with pytest.raises(ValueError):
        _make_engine(tw_instance, cfg, 42)


def test_population_cx_m3_valid_combo_and_default_do_not_raise(tw_instance):
    """合法组合 (gate=improve+educate) 与默认 none/False 全不抛。"""
    _make_engine(tw_instance, _m3_cfg(max_iterations=0,
                                      population_cx_educate=True), 42)  # 合法
    _make_engine(tw_instance, _m3_cfg(max_iterations=0), 42)  # gate=improve 无教育
    _make_engine(tw_instance, _cx_cfg(population_cx_mode="partition",
                                      max_iterations=0), 42)  # M2 默认 gate=none


# ── 键集断言: 默认 gate=none 无新键 (M2 逐位) ────────────────

def test_population_cx_m3_default_no_gate_keys(tw_instance):
    """gate 缺省 none + educate 缺省 False → population_stats 键集 = M2 键集
    (M1 键 + cx_* 键), 无 cx_gate_* / educate_* 键 — M2 行为逐位不变的键集证据。"""
    cfg = _cx_cfg(population_cx_mode="partition", population_cx_rate=1.0,
                  stagnation_limit=100000, max_iterations=0)
    _, engine = _run_engine(tw_instance, cfg, seed=42)
    assert set(engine.population_stats) == _M3_DEFAULT_KEYS
    assert not (_CX_GATE_KEYS & set(engine.population_stats))
    assert not (_EDUCATE_KEYS & set(engine.population_stats))


# ── gate=improve 集成 (rate=1.0, μ=4, 40 iter) ───────────────

def test_population_cx_gate_improve_integration_runs_deterministic(tw_instance):
    """gate=improve: 门过替换 / 门不过回退路径都执行 (cx_gate_success+fail ≥ 1,
    且各 ≥ 1 在本 fixture 上实测成立)。门不过 → 该迭代确有 destroy/repair 痕迹:
    selector.usage 恰 = 2 × (总迭代 − cx_gate_success) (gate-success 不更新
    selector, 其余每迭代 update 一次)。档案纯可行; last_solution 可行;
    同 seed 重跑逐位同 (含全部统计)。"""
    cfg = _m3_cfg(max_iterations=40)
    results = []
    for _ in range(2):
        archive, engine = _run_engine(tw_instance, dict(cfg), seed=42)
        s = engine.population_stats
        assert _CX_M1_KEYS <= set(s)
        assert _CX_KEYS <= set(s)
        assert _CX_GATE_KEYS <= set(s)
        assert s["cx_gate_success"] >= 1, "门过替换路径必须执行过"
        assert s["cx_gate_fail"] >= 1, "门不过回退路径必须执行过"
        assert isinstance(s["cx_gate_success"], int)
        assert isinstance(s["cx_gate_fail"], int)
        assert s["cx_attempts"] >= 1
        assert s["cx_children"] >= 1
        # 门过 → 替换 m*: 每次 gate-success 都贡献一次 member_advances
        assert sum(s["member_advances"].values()) >= s["cx_gate_success"]
        # 门不过 → 回退 destroy/repair + selector.update (cx 惯例 gate-success 不更新)
        n_update = cfg["max_iterations"] - s["cx_gate_success"]
        assert int(engine.selector.usage_count.sum()) == 2 * n_update
        assert engine.last_solution is not None
        assert engine.last_solution.is_feasible()
        assert len(engine.population_final) == 4
        for e in archive.entries:
            sol = e[1]["solution"]
            assert sol.is_feasible()
            assert len(sol.tw_violations()) == 0
        results.append((archive.best_cost(),
                        [r["outcome"]["cost"] for r in engine.trajectory],
                        dict(s)))
    assert results[0][0] == pytest.approx(results[1][0], abs=1e-9)
    assert results[0][1] == results[1][1]
    assert results[0][2] == results[1][2]


# ── education 集成 (gate=improve + educate, rate=1.0, 40 iter) ─

def test_population_cx_educate_integration_runs_deterministic(tw_instance):
    """educate=True: 每个生成的 cx 子代都先教育再进门 (educate_attempts ==
    cx_children); 教育后解可行; educate_success ≤ attempts; delta_sum ≥ 0
    (累计浮点); 档案纯可行; 同 seed 重跑逐位同 (含全部统计)。"""
    cfg = _m3_cfg(max_iterations=40, population_cx_educate=True)
    results = []
    for _ in range(2):
        archive, engine = _run_engine(tw_instance, dict(cfg), seed=42)
        s = engine.population_stats
        assert _CX_M1_KEYS <= set(s)
        assert _CX_KEYS <= set(s)
        assert _CX_GATE_KEYS <= set(s)
        assert _EDUCATE_KEYS <= set(s)
        assert s["cx_gate_success"] >= 1
        assert s["cx_gate_fail"] >= 1
        assert isinstance(s["educate_attempts"], int)
        assert isinstance(s["educate_success"], int)
        assert isinstance(s["educate_delta_sum"], float)
        assert s["educate_attempts"] == s["cx_children"]
        assert s["educate_attempts"] >= 1
        assert 0 <= s["educate_success"] <= s["educate_attempts"]
        assert s["educate_delta_sum"] >= 0.0
        # gate-success 教育产物可行 (公共链兜底)
        assert engine.last_solution.is_feasible()
        for e in archive.entries:
            sol = e[1]["solution"]
            assert sol.is_feasible()
            assert len(sol.tw_violations()) == 0
        results.append((archive.best_cost(),
                        [r["outcome"]["cost"] for r in engine.trajectory],
                        dict(s)))
    assert results[0][0] == pytest.approx(results[1][0], abs=1e-9)
    assert results[0][1] == results[1][1]
    assert results[0][2] == results[1][2]


# ══════════════════════════════════════════════════════════════
# M4 双空间罚域种群 (2026-09-07): population_penalty_mode tw
# 规格: docs/ma-alns-m4-spec-20260907.md。tw = 在 popcxge 形态 (gate=improve +
# partition) 上开 TW 罚域成员通道: λ0 静态校准 + 罚后接受门 (pool_penalty_gate)
# + 锦标赛/驱逐统一罚后成本; 档案仍只收可行; 容量/车辆仍硬。默认 none →
# popcxge (M3) 行为逐位不变 (键集断言 + 全套 pytest 回归)。
# ══════════════════════════════════════════════════════════════

_PENALTY_KEYS = {"penalty_member_iters", "penalty_infeasible_final", "lambda0"}


def _tw_tight_inst():
    """紧 TW 样本: 2 车场各 2 车 (容量 100), 两簇各 3 客户沿线排列。

    每簇唯一 TW 可行序 = 升序 (距离近→远): 任意换序/逆序即产生 TW 违规 →
    可构造 excess 大小可控的不可行解 (白盒罚域注入测试 / λ0 校准用)。
    """
    depots = [Depot(0, 0, 0.0, 0.0, 2, 100.0), Depot(1, 1, 50.0, 0.0, 2, 100.0)]
    custs = []
    specs = [
        (2, 8.0, 11.0), (3, 16.0, 20.0), (4, 24.0, 29.0),   # 簇 A (近 depot0)
        (5, 46.0, 6.0), (6, 40.0, 14.0), (7, 34.0, 21.0),   # 簇 B (近 depot1)
    ]
    for idx, x, due in specs:
        custs.append(Customer(idx, idx, x, 0.0, 5.0, 0,
                              ready_time=0.0, due_time=due, service_time=1.0))
    nodes = [(0.0, 0.0), (50.0, 0.0)] + [(c.x, c.y) for c in custs]
    n = len(nodes)
    dist = [[__import__("math").dist(nodes[i], nodes[j]) for j in range(n)]
            for i in range(n)]
    return Instance("pop_tw_tight", "test", 2, 6, 100.0, depots, custs, dist,
                    EmissionModel(base_emission_rate=0.0,
                                  load_correction_alpha=0.0))


def _m4_tw_cfg(**overrides):
    """M4 tw 臂基线: popcxge 底形 (partition + gate=improve) + 罚域 tw; κ=0。
    在 _m3_cfg 之上加 penalty_mode=tw (rate=1.0 强制每迭代 cx 尝试)。
    educate 由用例显式开启 (popcxgep 臂含 educate)。
    """
    cfg = _m3_cfg(population_penalty_mode="tw", population_penalty_mult=1.0)
    cfg.update(overrides)
    return cfg


def _tw_excess_total(sol):
    return float(sum(v["excess"] for v in sol.tw_violations()))


def _tight_pool_routes():
    """白盒注入的 4 成员路由 (紧 TW 样本): m0/m1 不可行 (excess 14/77),
    m2/m3 可行 (成本 120/132)。λ0 = best_feas / median(14,77) →
    m0 罚后成本 < best_feas → 锦标赛必选 m0 (罚门路径被驱动)。"""
    return [
        {0: [[2, 4, 3]], 1: [[5, 6, 7]]},        # m0: 不可行, low excess
        {0: [[4, 3, 2]], 1: [[7, 6, 5]]},        # m1: 不可行, high excess
        {0: [[2, 3], [4]], 1: [[5], [6, 7]]},    # m2: 可行, cost 120
        {0: [[2, 3], [4]], 1: [[5, 6], [7]]},    # m3: 可行, cost 132
    ]


def _inject_build_pool(engine, members):
    """白盒: 前 μ 次 _build_pool_member 调用返回注入成员 (解/obj 对队列);
    之后回退原方法 (防御: 刷新兜底分支不应被触发)。"""
    orig = engine._build_pool_member
    it = iter(members)

    def _fake():
        try:
            return next(it)
        except StopIteration:
            return orig()

    engine._build_pool_member = _fake


def _craft_members(inst):
    """在引擎实例 inst 上构造注入成员列表 [(sol, obj)] (与 engine 共享 instance
    引用域; obj = calculate_objectives)。"""
    out = []
    for routes in _tight_pool_routes():
        sol = Solution(inst, routes)
        out.append((sol, calculate_objectives(sol)))
    return out


# ── M4 config 键默认值 ────────────────────────────────────────

def test_population_penalty_default_attrs(tw_instance):
    """config 无 penalty 键 → mode none / mult 1.0 / _penalty_lambda0 0.0。"""
    engine = _make_engine(tw_instance, dict(LOCK_CONFIG), seed=42)
    assert engine.population_penalty_mode == "none"
    assert engine.population_penalty_mult == pytest.approx(1.0)
    assert engine._penalty_lambda0 == 0.0


# ── M4 纯函数: pool_penalty_gate / pool_penalized_cost ────────

def test_pool_penalty_gate_pure_function_locks_gate():
    """罚后接受门 = 罚后成本严格改进 m* (差 > 1e-9)。三用例锁定:
    (child 更优)→True; (child 劣)→False; (等值)→False。可行性不进门的语义
    由调用点测试覆盖 (纯函数本身只比罚后成本标量)。"""
    assert pool_penalty_gate(100.0, 101.0) is True
    assert pool_penalty_gate(101.0, 100.0) is False
    assert pool_penalty_gate(100.0, 100.0) is False
    # 差 ≤ 1e-9 不算严格改进
    assert pool_penalty_gate(101.0 - 1e-10, 101.0) is False


def test_pool_penalized_cost_pure_function():
    """罚后成本纯函数: excess ≤ 1e-9 → 真实成本; 否则 cost + λ0×excess。"""
    assert pool_penalized_cost(100.0, 0.0, 5.0) == pytest.approx(100.0)
    assert pool_penalized_cost(100.0, 1e-10, 5.0) == pytest.approx(100.0)
    assert pool_penalized_cost(100.0, 10.0, 5.0) == pytest.approx(150.0)


# ── M4 守卫 ─────────────────────────────────────────────────

def test_population_penalty_tw_guards(tw_instance):
    """tw 非法组合 → ValueError:
    (a) tw + population_mode=False; (b) tw + gate 非 improve;
    (c) tw + cx_mode 非 partition; (d) tw + legacy soft (capacity/tw/educate);
    (e) 非法 mode 值。"""
    # (a) population_mode=False
    cfg = dict(LOCK_CONFIG, population_penalty_mode="tw")
    with pytest.raises(ValueError):
        _make_engine(tw_instance, cfg, 42)
    # (b) gate=none (pop on + partition)
    cfg = dict(LOCK_CONFIG, population_mode=True,
               population_cx_mode="partition", population_penalty_mode="tw")
    with pytest.raises(ValueError):
        _make_engine(tw_instance, cfg, 42)
    # (c) gate=improve + mode=route → 既有 M3 守卫先抛 (tw 同禁)
    cfg = dict(LOCK_CONFIG, population_mode=True, population_cx_mode="route",
               population_cx_gate="improve", population_penalty_mode="tw")
    with pytest.raises(ValueError):
        _make_engine(tw_instance, cfg, 42)
    # (d) legacy soft 组合 (M1/M2 守卫或 M4 守卫, ValueError 即可)
    for ov in ({"soft_capacity": True}, {"soft_tw": True},
               {"educate_mode": "deep"}):
        cfg = dict(LOCK_CONFIG, population_mode=True,
                   population_cx_mode="partition",
                   population_cx_gate="improve",
                   population_penalty_mode="tw", **ov)
        with pytest.raises(ValueError):
            _make_engine(tw_instance, cfg, 42)
    # (e) 非法 mode 值
    cfg = dict(LOCK_CONFIG, population_penalty_mode="bogus")
    with pytest.raises(ValueError, match="population_penalty_mode"):
        _make_engine(tw_instance, cfg, 42)


def test_population_penalty_valid_and_default_no_raise(tw_instance):
    """合法组合 (tw + gate=improve + partition, 有/无 educate) 与默认 none 全不抛。"""
    _make_engine(tw_instance, _m4_tw_cfg(max_iterations=0), 42)  # 无 educate
    _make_engine(tw_instance, _m4_tw_cfg(max_iterations=0,
                                         population_cx_educate=True), 42)
    _make_engine(tw_instance, dict(LOCK_CONFIG), 42)  # 默认 none


# ── 键集断言: 默认 none 无 penalty 键 (popcxge 逐位) ─────────

def test_population_penalty_default_no_penalty_keys(tw_instance):
    """penalty_mode 缺省 none → population_stats 键集 = M3 (gate/educate) 键集,
    无 penalty_* / lambda0 键 — popcxge 逐位不变的键集证据。"""
    cfg = _m3_cfg(max_iterations=0, population_cx_educate=True)
    _, engine = _run_engine(tw_instance, cfg, seed=42)
    assert not (_PENALTY_KEYS & set(engine.population_stats))
    assert "lambda0" not in engine.population_stats


# ── λ0 校准单测 (白盒注入紧 TW 池, max_iterations=0) ─────────

def test_population_penalty_lambda0_calibration(tw_instance):
    """λ0 = mult × best_feas / max(1.0, excess_med): 注入池含 2 不可行成员
    (excess 14/77) + 2 可行 (best_feas=120) → λ0 = 1.0×120/45.5。
    penalty_infeasible_final = 2 (池保持注入态, 0 迭代); 成员罚后成本扩展正确。"""
    cfg = _m4_tw_cfg(max_iterations=0)
    # 紧窗实例 (可构造不可行解, λ0 校准需 excess 中位数 > 0)
    tight = _tw_tight_inst()
    engine2 = _make_engine(tight, cfg, seed=42)
    members = _craft_members(engine2.instance)
    feas_costs = [o[0] for s, o in members if s.is_feasible()]
    excesses = [_tw_excess_total(s) for s, o in members if not s.is_feasible()]
    best_feas = min(feas_costs)
    med = float(np.median(excesses))
    expected = 1.0 * best_feas / max(1.0, med)
    assert expected > 0
    _inject_build_pool(engine2, members)
    engine2.run()
    s = engine2.population_stats
    assert set(s) >= _PENALTY_KEYS
    assert s["lambda0"] == pytest.approx(expected)
    assert s["penalty_member_iters"] == 0      # 0 迭代未选
    assert s["penalty_infeasible_final"] == 2  # m0/m1 不可行留在池
    assert engine2._penalty_lambda0 == pytest.approx(expected)


# ── tw 集成 (宽窗 fixture, μ=4, 40 iter) ──────────────────────

def _assert_m4_tw_integration(tw_instance, cfg, educate):
    results = []
    for _ in range(2):
        archive, engine = _run_engine(tw_instance, dict(cfg), seed=42)
        s = engine.population_stats
        assert _CX_M1_KEYS <= set(s)
        assert _CX_KEYS <= set(s)
        assert _CX_GATE_KEYS <= set(s)
        if educate:
            assert _EDUCATE_KEYS <= set(s)
        assert _PENALTY_KEYS <= set(s)
        assert s["lambda0"] > 0.0
        assert isinstance(s["penalty_member_iters"], int)
        assert isinstance(s["penalty_infeasible_final"], int)
        assert s["penalty_member_iters"] >= 0
        # 宽窗 fixture 全可行 → 罚域通道未用 (退化 all-feasible), 纯净性仍锁
        assert s["penalty_infeasible_final"] == 0
        # 轨迹 outcome 记真实成本 + 罚后成本 + tw_excess (诊断键存在)
        for tr in engine.trajectory:
            out = tr["outcome"]
            assert "penalized_cost" in out
            assert "tw_excess" in out
            assert out["penalized_cost"] == pytest.approx(out["cost"])
            assert out["tw_excess"] == 0.0
        assert engine.last_solution is not None
        assert engine.last_solution.is_feasible()
        for e in archive.entries:
            sol = e[1]["solution"]
            assert sol.is_feasible()
            assert len(sol.tw_violations()) == 0
        results.append((archive.best_cost(),
                        [r["outcome"]["cost"] for r in engine.trajectory],
                        dict(s)))
    assert results[0][0] == pytest.approx(results[1][0], abs=1e-9)
    assert results[0][1] == results[1][1]
    assert results[0][2] == results[1][2]


def test_population_penalty_tw_integration_runs_deterministic(tw_instance):
    """tw 模式 (gate=improve + partition, 无 educate) 40 iter: 完成 / 档案纯可行 /
    penalty 键齐 / 同 seed 重跑逐位同 (含全部统计与轨迹罚后字段)。"""
    _assert_m4_tw_integration(tw_instance, _m4_tw_cfg(max_iterations=40),
                              educate=False)


def test_population_penalty_tw_educate_integration_runs_deterministic(tw_instance):
    """tw 模式 + educate (popcxgep 臂形): 同上, 教育键齐。"""
    _assert_m4_tw_integration(tw_instance, _m4_tw_cfg(max_iterations=40,
                                                       population_cx_educate=True),
                              educate=True)


# ── 白盒注入: 罚门路径真实生效 (紧 TW fixture) ────────────────

def test_population_penalty_tw_whitebox_penalty_path(tw_instance):
    """紧 TW 池注入 (μ=4, tournament=4): m0 不可行 low-excess 罚后成本最低 →
    锦标赛必选 → penalty_member_iters ≥ 1; m1 高罚不可行尸体不被驱逐 →
    penalty_infeasible_final ≥ 1; λ0 与手工公式一致; 档案全可行; 同 seed 逐位同。

    用白盒注入 (spec 测试 4 逃逸条款): 宽窗 fixture 无法自然出现不可行子代。
    """
    cfg = _m4_tw_cfg(max_iterations=20, population_tournament=4)
    outs = []
    for _ in range(2):
        tight = _tw_tight_inst()
        engine = _make_engine(tight, cfg, seed=42)
        members = _craft_members(engine.instance)
        # 前提锁存: m0 罚后成本为池内唯一最低 (驱动罚门路径)
        feas_costs = [o[0] for s, o in members if s.is_feasible()]
        excesses = [_tw_excess_total(s) for s, o in members
                    if not s.is_feasible()]
        best_feas = min(feas_costs)
        med = float(np.median(excesses))
        lam = best_feas / max(1.0, med)
        pens = [o[0] if s.is_feasible()
                else o[0] + lam * _tw_excess_total(s)
                for s, o in members]
        assert min(pens) == pytest.approx(pens[0]), "注入池 m0 须为罚后成本最低"
        _inject_build_pool(engine, members)
        archive = engine.run()
        s = engine.population_stats
        assert set(s) >= _PENALTY_KEYS
        assert s["lambda0"] == pytest.approx(lam)
        assert s["penalty_member_iters"] >= 1, "罚门路径必须真实被驱动"
        assert s["penalty_infeasible_final"] >= 1, "不可行尸体留在池终态"
        for e in archive.entries:
            assert e[1]["solution"].is_feasible(), "档案混入不可行解"
        outs.append((archive.best_cost() if archive.size else None,
                     [r["outcome"]["cost"] for r in engine.trajectory],
                     dict(s)))
    # 同 seed 重跑逐位同 (确定性)
    assert outs[0][0] == outs[1][0]
    assert outs[0][1] == outs[1][1]
    assert outs[0][2] == outs[1][2]
