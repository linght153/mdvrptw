"""软 TW 搜索 (探针, 2026-09-06) 回归测试。

背景: 全部 09-04/05/06 机制保持 TW 硬约束; 引擎无 TW 软罚 (仅容量软罚
soft_capacity, 镜像对象)。机制 = config 键 soft_tw (默认 False): repair 后 TW
强制门放宽 (违规解流到接受准则) + 接受目标 = 真实目标 + λ×Σexcess + 档案仍只收
可行解 + LS 只精化可行解 (违规解跳过) + 每 soft_tw_repair_interval 迭代
_enforce_tw 周期修复播种 (镜像 soft_capacity 的采纳规则)。
红线: 默认配置行为必须逐位不变 (test_default_soft_tw_false_noop)。
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace as dc_replace

from src.core.instance import Customer, Depot, EmissionModel, Instance
from src.core.instance import instance_from_parsed
from src.core.solution import Solution
from src.alns.engine import ALNSEngine
from src.alns.selection import SegmentRewardSelection
from src.operators.local_search import rvnd_improve
from src.operators.destroy import DESTROY_OPERATORS
from src.operators.repair import greedy_cost_tw_insertion
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

DATA = Path(__file__).resolve().parent.parent / "data" / "mdvrptw_raw"

TW_BASE = {
    "max_iterations": 60, "max_time_seconds": 1e9, "stagnation_limit": 100000,
    "segment_size": 100, "reaction_factor": 0.1, "decay_factor": 1.0,
    "min_selection_prob": 0.005, "initial_temperature": None, "cooling_rate": None,
    "archive_capacity": 20, "parent_selection": "crowding",
    "initial_solution": "nearest", "k_min_ratio": 0.10, "k_max_ratio": 0.40,
    "sigma": 3.0, "local_search_max_iter": 10, "local_search_freq": 5,
    "local_search_on_accept": True, "local_search_final": False,
}

STATS_KEYS = {"violation_iters", "repair_events", "repair_feasible",
              "seed_adopted", "lam"}


def _load(name: str):
    text = open(DATA / f"{name}.txt", encoding="utf-8").read()
    return instance_from_parsed(parse_mdvrptw(text), name=name)


@pytest.fixture(scope="module")
def pr01_instance():
    return _load("pr01")


@pytest.fixture(scope="module")
def pr03_instance():
    return _load("pr03")


def _fresh(inst):
    """深拷贝客户列表 — 引擎会改 assigned_depot_index (共享可变状态)。"""
    return dc_replace(inst, customers=[dc_replace(c) for c in inst.customers])


def _run_engine(inst, config, seed):
    """装配 TW 实例引擎 (与 experiments/benchmark_softtw_ab.py 同构)。"""
    inst = _fresh(inst)
    destroy_names = list(DESTROY_OPERATORS.keys())
    selector = SegmentRewardSelection(
        destroy_names, ["greedy_cost_tw"],
        segment_size=config.get("segment_size", 100),
        reaction_factor=config.get("reaction_factor", 0.1),
        decay_factor=config.get("decay_factor", 1.0),
        min_selection_prob=config.get("min_selection_prob", 0.005),
    )
    engine = ALNSEngine(
        instance=inst, config=config, rng=np.random.default_rng(seed),
        selector=selector, repair_ops={"greedy_cost_tw": greedy_cost_tw_insertion},
        local_search=rvnd_improve,
    )
    archive = engine.run()
    return archive, engine


def _best(archive):
    """档案最小成本可行解。"""
    feas = [(e[0][0], e[1]["solution"]) for e in archive.entries
            if e[1]["solution"].is_feasible()]
    if not feas:
        return None, None
    c = min(c for c, _ in feas)
    sol = next(s for c2, s in feas if abs(c2 - c) < 1e-9)
    return c, sol


def _assert_pure_archive(archive):
    """档案纯净性: 全条目可行 + TW 零违规 (违规解不入档是全局规则)。"""
    assert len(archive.entries) > 0
    for e in archive.entries:
        sol = e[1]["solution"]
        assert sol.is_feasible(), "档案混入不可行解"
        assert len(sol.tw_violations()) == 0, "档案混入 TW 违规解"


# ── ⑤ config 缺省 + ① 默认 none: soft_tw=False 显式 == 无键 == 既有行为 ──

def test_soft_tw_default_false_noop(pr01_instance):
    """缺省 (无 soft_tw 键) == 显式 soft_tw=False, 引擎属性默认锁存。"""
    inst = pr01_instance
    a1 = _run_engine(inst, dict(TW_BASE), 42)
    a2 = _run_engine(inst, dict(TW_BASE, soft_tw=False), 42)
    c1, _ = _best(a1[0])
    c2, _ = _best(a2[0])
    assert c1 == pytest.approx(c2, abs=1e-9)
    # 默认属性 (⑤): 无键 → soft_tw False / mult 1.0 / interval 100
    _, eng_default = _run_engine(inst, dict(TW_BASE), 42)
    assert eng_default.soft_tw is False
    assert eng_default.soft_tw_lambda_mult == 1.0
    assert eng_default.soft_tw_repair_interval == 100
    assert set(eng_default.soft_tw_stats) == STATS_KEYS
    assert eng_default.soft_tw_stats["lam"] == 0.0
    # 默认路径 new 分支不可达: 档案与无键逐位一致
    assert len(a1[0].entries) == len(a2[0].entries)


# ── ② _soft_tw_penalize 罚计算单测 ─────────────────────────────

def _build_tight_line_inst():
    """1 车场 2 车; 两客户沿 +x 线, 紧 TW: 单条升序路由可行, 降序违规。

    升序 [c1,c2]: c1 到达 10 ≤ due 10, c2 到达 21 ≤ due 21 → 可行。
    降序 [c2,c1]: c2 到达 20 ≤ 21, 回程 c1 到达 31 > due 10 → excess 21。
    """
    depots = [Depot(index=0, id=0, x=0.0, y=0.0, vehicles_available=2,
                    capacity=100.0)]
    custs = [
        Customer(index=1, id=1, x=10.0, y=0.0, demand=1.0, assigned_depot_index=0,
                 ready_time=0.0, due_time=10.0, service_time=1.0),
        Customer(index=2, id=2, x=20.0, y=0.0, demand=1.0, assigned_depot_index=0,
                 ready_time=0.0, due_time=21.0, service_time=1.0),
    ]
    nodes = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    dist = [[math.dist(nodes[i], nodes[j]) for j in range(3)] for i in range(3)]
    return Instance("tight_line", "test", 1, 2, 100.0, depots, custs, dist,
                    EmissionModel())


def test_soft_tw_penalize_calculation():
    """罚 = λ × Σ(v['excess'] for v in tw_violations()); 违规迭代计数 +1;
    原目标不被修改 (深拷贝语义)。"""
    inst = _build_tight_line_inst()
    engine = ALNSEngine(inst, dict(TW_BASE, soft_tw=True),
                        np.random.default_rng(0))
    engine._set_soft_tw_lam(3.0)
    assert engine.soft_tw is True
    assert engine._soft_tw_lam == 3.0

    feas_sol = Solution(inst, {0: [[1, 2]]})
    viol_sol = Solution(inst, {0: [[2, 1]]})
    assert feas_sol.is_feasible()
    assert not viol_sol.is_feasible()
    viol = viol_sol.tw_violations()
    assert len(viol) == 1
    total_excess = sum(v["excess"] for v in viol)
    assert total_excess == pytest.approx(21.0)

    obj = (100.0, 50.0)
    # 可行解不罚: 原样返回, 不计数
    assert engine._soft_tw_penalize(feas_sol, obj) == obj
    assert engine.soft_tw_stats["violation_iters"] == 0
    # 违规解: 罚 = λ×Σexcess, 返回新 tuple, 原目标不动
    pen = engine._soft_tw_penalize(viol_sol, obj)
    assert pen[0] == pytest.approx(100.0 + 3.0 * total_excess)
    assert pen[1] == 50.0
    assert obj == (100.0, 50.0)
    assert engine.soft_tw_stats["violation_iters"] == 1


def test_soft_tw_penalize_disabled_returns_obj():
    """soft_tw=False (默认): 罚函数原样返回, 不计数 (逐位不变)。"""
    inst = _build_tight_line_inst()
    engine = ALNSEngine(inst, dict(TW_BASE), np.random.default_rng(0))
    viol_sol = Solution(inst, {0: [[2, 1]]})
    obj = (100.0, 50.0)
    assert engine._soft_tw_penalize(viol_sol, obj) == obj
    assert engine.soft_tw_stats["violation_iters"] == 0


# ── ③ soft_tw=True 60 iter 冒烟: 不崩 + stats 键 + 档案纯可行 + 确定性 ──

def test_soft_tw_run_smoke_pure_deterministic(pr01_instance):
    """soft_tw 开启: 运行完整 + 档案纯净 + 同 seed 可复现 + λ 已初始化。"""
    cfg = dict(TW_BASE, soft_tw=True)
    results = []
    for _ in range(2):
        archive, engine = _run_engine(pr01_instance, dict(cfg), 42)
        assert engine.soft_tw is True
        assert set(engine.soft_tw_stats) == STATS_KEYS
        assert engine._soft_tw_lam0 > 0.0
        assert engine.soft_tw_stats["lam"] == pytest.approx(engine._soft_tw_lam)
        for v in engine.soft_tw_stats.values():
            assert v >= 0
        _assert_pure_archive(archive)
        best_cost, sol = _best(archive)
        assert sol is not None
        results.append(best_cost)
    assert results[0] == pytest.approx(results[1], abs=1e-9)  # 确定性


# ── ④ 周期播种: 注入违规轨迹 → interval 触发后计数语义正确 ──────────

def test_soft_tw_periodic_seed_counters(pr03_instance):
    """soft_tw 周期播种: 低 λ 让违规轨迹真实穿越 → 每个 interval 触发修复;
    repair_events = 触发器数; 修复成功 ≥1; 轨迹违规时采纳 → seed_adopted ≥1;
    repair_events/repair_feasible/seed_adopted 三计数语义一致; 档案纯可行。

    用 pr03 (D=460, n=144) 而非 pr16: 2026-09-11 duration 硬约束进入播种
    通道后 (spec §4.1 罚域只松弛 TW 不松弛 duration), 288 客户级紧实例
    (pr16/15/20, D 400-425) 的违规轨迹态在种子修复预算内无法收敛到完全
    可行 (实测 rf=0/6 → 无采纳, 计数链测不到); pr03 可稳定复现
    rf=6/6、adopted≥1 的完整计数链 (2026-09-11 实测)。
    """
    cfg = dict(TW_BASE, soft_tw=True, soft_tw_lambda_mult=0.01,
               soft_tw_repair_interval=10)
    archive, engine = _run_engine(pr03_instance, cfg, seed=42)
    s = engine.soft_tw_stats
    assert s["repair_events"] == 6          # 60 iter / interval 10
    assert s["violation_iters"] >= 1        # 违规轨迹真实发生并被罚评价
    assert 1 <= s["repair_feasible"] <= s["repair_events"]
    assert s["seed_adopted"] >= 1           # 至少一次违规被周期修复并锚回
    assert s["seed_adopted"] <= s["repair_feasible"]
    _assert_pure_archive(archive)


# ── _tw_excess 辅助 (直接语义锚定) ───────────────────────────────

def test_tw_excess_sum():
    """_tw_excess = Σexcess; 可行解 = 0。"""
    inst = _build_tight_line_inst()
    engine = ALNSEngine(inst, dict(TW_BASE, soft_tw=True),
                        np.random.default_rng(0))
    assert engine._tw_excess(Solution(inst, {0: [[1, 2]]})) == 0.0
    assert engine._tw_excess(Solution(inst, {0: [[2, 1]]})) == pytest.approx(21.0)


def test_softtw_runner_names_do_not_collide():
    """命名纪律回归 (2026-09-10): softtw runner 的迭代数与 λ 档必须落文件名。

    背景: benchmark_softtw_ab.py 曾以固定文件名写盘 — pr20-s42 的 6000iter 主记录
    被 700iter 温度曲线诊断运行覆写, λ 两档 (mult 1.0/0.003) 互相覆盖致论文表 A2
    高 λ 臂整列无盘上证据。
    """
    import importlib.util
    runner = (Path(__file__).resolve().parent.parent / "experiments"
              / "benchmark_softtw_ab.py")
    spec = importlib.util.spec_from_file_location("_bench_softtw_ab", runner)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.out_tag(6000, 1.0) == "_6000iter_m1"
    assert mod.out_tag(6000, 0.003) == "_6000iter_m0.003"
    assert mod.out_tag(700, 0.003) == "_700iter_m0.003"
    assert mod.out_tag(6000, 1.0) != mod.out_tag(6000, 0.003), "λ 两档不得同名"
    assert mod.out_tag(6000, 0.003) != mod.out_tag(700, 0.003), "迭代数不得同名"
