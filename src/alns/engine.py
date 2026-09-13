"""ALNS 主引擎 — 可配置组件的统一循环。"""

import time
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict
from numbers import Integral, Real

import numpy as np

from src.core.instance import Instance
from src.core.solution import Solution
from src.core.objectives import calculate_objectives
from src.core.pareto import Objective
from src.core.initial_solution import build_initial_solution

from src.alns.acceptance import ParetoAcceptance
from src.alns.parent_selection import PARENT_SELECTION_STRATEGIES
from src.alns.selection import SegmentRewardSelection
from src.alns.selection_context import OperatorOutcome, SearchContext
from src.alns.archive import ParetoArchive

from src.operators.destroy import DESTROY_OPERATORS
from src.operators.repair import REPAIR_OPERATORS


class ALNSEngine:
    """
    可配置的 ALNS 引擎，用于多目标 Green MDVRP。

    组件可通过构造函数参数替换：
    - selector: 算子选择策略
    - acceptor: 接受准则
    - destroy_ops: 破坏算子名称 → 函数的字典
    - repair_ops: 修复算子名称 → 函数的字典
    - parent_selector: 父代选择函数
    - local_search: 可选局部搜索函数, 按 local_search_freq 调度执行
    """

    def __init__(
        self,
        instance: Instance,
        config: dict,
        rng: np.random.Generator,
        selector=None,
        acceptor=None,
        destroy_ops: dict | None = None,
        repair_ops: dict | None = None,
        parent_selector=None,
        local_search=None,       # (sol) → sol, 可选
        initial_solution: Solution | None = None,
    ):
        self.instance = instance
        self.config = config
        self.rng = rng
        self.local_search_fn = local_search  # 存储 LS 函数
        self._initial_solution = initial_solution

        # ── 局部搜索调度参数 ──────────────────────────────────
        self.ls_freq = config.get("local_search_freq", 10)
        self.ls_on_accept = config.get("local_search_on_accept", True)
        self.ls_final = config.get("local_search_final", True)
        self.ls_max_iter = config.get("local_search_max_iter", 30)

        # 组件
        self.destroy_ops = destroy_ops or dict(DESTROY_OPERATORS)
        self.repair_ops = repair_ops or dict(REPAIR_OPERATORS)
        self.destroy_names = list(self.destroy_ops.keys())
        self.repair_names = list(self.repair_ops.keys())

        # 算子选择器
        if selector is not None:
            self.selector = selector
        else:
            self.selector = SegmentRewardSelection(
                self.destroy_names,
                self.repair_names,
                segment_size=config.get("segment_size", 100),
                reaction_factor=config.get("reaction_factor", 0.1),
                decay_factor=config.get("decay_factor", 1.0),
                min_selection_prob=config.get("min_selection_prob", 0.005),
            )

        # 接受准则 (SA 参数支持 null=自动校准)
        if acceptor is not None:
            self.acceptor = acceptor
        else:
            self.acceptor = ParetoAcceptance(
                initial_temperature=config.get("initial_temperature"),
                cooling_rate=config.get("cooling_rate"),
                max_iterations=config.get("max_iterations", 5000),
            )

        # 父代选择
        ps_name = config.get("parent_selection", "crowding")
        self.parent_selector = parent_selector or PARENT_SELECTION_STRATEGIES[ps_name]

        # 档案
        self.archive_diversity = bool(config.get("archive_diversity", False))
        self.archive_diversity_threshold = float(
            config.get("archive_diversity_threshold", 0.2)
        )
        # A2 (2026-09-04): 多样性贡献进 fitness 裁剪 (默认关)
        self.archive_fitness_dc = bool(config.get("archive_fitness_dc", False))
        self.archive_fitness_kappa = float(
            config.get("archive_fitness_kappa", 0.05)
        )
        # admit_only 判别臂 (2026-09-04): 只放宽准入 (默认关)
        self.archive_admit_relaxed = bool(config.get("archive_admit_relaxed", False))
        # score 语义 (探针 2026-09-05): "admission" = 现默认 (score 随入档成败
        # 2/1/0); "activity" = 可行且未崩溃即反馈 2.0 — 与准入解耦, 复刻 admit
        # 的 score 分布但不放宽准入 (档案/stagnation 语义 = base); "quality" =
        # P&R 改进阶梯 (new<best 2.0 / new<parent 1.5 / accepted 1.0 / 拒 0)。
        # 默认 admission → 行为逐位不变 (126 PASS 锁存)。
        scheme = config.get("score_scheme", "admission")
        if scheme not in ("admission", "activity", "quality"):
            raise ValueError(
                f"score_scheme must be admission|activity|quality, got {scheme!r}"
            )
        self.score_scheme = scheme
        if int(config.get("archive_capacity", 80)) < 1:
            raise ValueError("archive_capacity must be >= 1")
        self.archive = ParetoArchive(
            capacity=config.get("archive_capacity", 80),
            single_objective=(self.instance.emission_model.base_emission_rate == 0.0),
            diversity=self.archive_diversity,
            diversity_threshold=self.archive_diversity_threshold,
            fitness_dc=self.archive_fitness_dc,
            fitness_kappa=self.archive_fitness_kappa,
            admit_relaxed=self.archive_admit_relaxed,
        )

        # 参数
        self.max_iterations = config.get("max_iterations", 5000)
        self.max_time = config.get("max_time_seconds", 3600)
        self.stagnation_limit = config.get("stagnation_limit", 500)
        self.init_method = config.get("initial_solution", "nearest")
        self.k_min_ratio = config.get("k_min_ratio", 0.10)
        self.k_max_ratio = config.get("k_max_ratio", 0.40)
        self.deep_destroy_max_ratio = config.get("deep_destroy_max_ratio", 0.6)
        self.stagnation_deep_ratio = config.get("stagnation_deep_ratio", 0.5)
        self.restart_clear_archive = config.get("restart_clear_archive", False)
        # stagnation 计数源 (2026-09-04 修复候选 v1): "archive" = 档案 add 成败
        # (旧); "current" = current 质量停滞。实测两源逐位等价 (无温度接受下
        # accepted∧改进 ⇔ added) → v1 设计无效, 默认 archive 保留。
        self.stagnation_source = config.get("stagnation_source", "archive")
        # 恢复阈值除数 v2 (2026-09-04): recovery_threshold = max(30,
        # stagnation_limit // divisor)。旧固定 //3; 预算-重启权衡: 短迭代预算
        # (6000 iter) 重启爬坡成本高 → 小除数少重启; 长时间预算 (600s)
        # 重启=多样性 → 大除数多重启。默认 3 = 旧行为逐位不变。
        self.stagnation_recovery_divisor = int(
            config.get("stagnation_recovery_divisor", 3))
        # ── 软容量搜索 (探针 2, 2026-09-04) ────────────────────
        # 默认 False → 行为逐位不变 (111 PASS 回归锁存)。开启时容量约束转为
        # 罚函数 (TW/车辆数仍硬): repair 容量门放宽 + 接受目标 = 距离 +
        # λ×超载量, λ = mult × 初始可行成本 / 总需求; 档案仍只收可行解;
        # LS 只精化可行解; 每 soft_repair_interval 迭代周期修复播种档案。
        self.soft_capacity = bool(config.get("soft_capacity", False))
        self.soft_capacity_penalty_mult = float(
            config.get("soft_capacity_penalty_mult", 1.0)
        )
        self.soft_repair_interval = int(config.get("soft_repair_interval", 100))
        self._soft_lam = 0.0
        self._soft_lam0 = 0.0
        self._soft_lam_max = 0.0
        self._soft_seed_candidate: "Solution | None" = None
        self._soft_seed_obj: "Objective | None" = None
        self.soft_stats = {
            "seed_attempts": 0,
            "seed_adds": 0,
            "max_excess": 0.0,
            "overloaded_iters": 0,
        }
        # 修复/插入算子的容量门读取 (repair.py getattr 检查; False 无行为变化)
        setattr(self.instance, "soft_capacity", self.soft_capacity)
        # ── 软 TW 搜索 (探针, 2026-09-06) ─────────────────────
        # 默认 False → 行为逐位不变 (154 PASS 回归锁存)。开启时时间窗约束转为
        # 罚函数 (容量/车辆数仍硬): repair 后的 TW 强制门放宽 (违规解可流到
        # 接受准则) + 接受目标 = 距离 + λ×Σexcess, λ = mult × 初始可行成本 /
        # 客户数 (每客户平均成本 ≈ 每单位时间违规的罚金量级); 档案仍只收可行
        # 解; LS 只精化可行解 (违规解上 RVND 空转 → 跳过); 每
        # soft_tw_repair_interval 迭代周期 _enforce_tw 修复播种 (镜像 soft_
        # capacity 的 _soft_seed 采纳规则)。种子失败 λ ×1.5 上调 / 入档成功
        # ÷1.5 回落, 上限 λ0×100 — 完全镜像 soft_capacity 的 λ 管理。
        self.soft_tw = bool(config.get("soft_tw", False))
        self.soft_tw_lambda_mult = float(config.get("soft_tw_lambda_mult", 1.0))
        self.soft_tw_repair_interval = int(
            config.get("soft_tw_repair_interval", 100))
        self._soft_tw_lam = 0.0
        self._soft_tw_lam0 = 0.0
        self._soft_tw_lam_max = 0.0
        self._soft_tw_seed_candidate: "Solution | None" = None
        self._soft_tw_seed_obj: "Objective | None" = None
        self.soft_tw_stats = {
            "violation_iters": 0,
            "repair_events": 0,
            "repair_feasible": 0,
            "seed_adopted": 0,
            "lam": 0.0,
        }
        # ── 种群交叉 (探针, 2026-09-04): 路线复制交叉迭代 ────
        # 默认 False → 主循环逐位不变。rate=每迭代触发概率; inherit_prob=
        # 双亲单路由继承率 (见 src/operators/crossover.py)。
        self.population_crossover = bool(config.get("population_crossover", False))
        self.crossover_rate = float(config.get("crossover_rate", 0.2))
        self.crossover_inherit_prob = float(config.get("crossover_inherit_prob", 0.6))
        # 交叉第二父代选择 (探针 2026-09-05): "random" = 现默认 (随机异质路由
        # 集); "diverse" = 与 A 结构距离 (路由 Jaccard) 最大且成本 ≤ A×(1+band)
        # 的档案成员 — 离线教育列实验的互补双亲选择在线化。
        pair = config.get("crossover_pair", "random")
        if pair not in ("random", "diverse"):
            raise ValueError(f"crossover_pair must be random|diverse, got {pair!r}")
        self.crossover_pair = pair
        self.crossover_cost_band = float(config.get("crossover_cost_band", 0.10))
        self.cx_stats = {"iterations": 0, "children": 0, "novel_route_keys": 0}
        # ── 车场分配重平衡 (方向①, 2026-09-06) ─────────────────
        # 默认 none → 主循环逐位不变。removal = 周期错配优先移除 (mismatched
        # 替代当迭代 destroy); redispatch = 周期错配修复 pass (独立后处理,
        # 不进 selector 统计); both = 同开 (共享 rebalance_interval)。
        mode = config.get("rebalance_mode", "none")
        if mode not in ("none", "removal", "redispatch", "both"):
            raise ValueError(
                f"rebalance_mode must be none|removal|redispatch|both, got {mode!r}")
        self.rebalance_mode = mode
        self.rebalance_interval = int(config.get("rebalance_interval", 0))
        self.rebalance_ratio = float(config.get("rebalance_ratio", 0.20))
        self.rebalance_stats = {
            "removal_passes": 0,
            "redispatch_passes": 0,
            "redispatch_moves": 0,
        }
        # ── 破坏性教育 (方向②, 2026-09-06) ───────────────────
        # 默认 none → 主循环逐位不变。deep = cx 后代小规模破坏-重建-教育
        # (随机移除 educate_burst_ratio×n → repair 重插 → 短 LS), 教育后
        # 可行且成本 < 原后代才替换 new_sol (不劣化当前迭代轨迹)。教育不
        # 更新 selector 权重 (同 cx 分支惯例)。
        educate = config.get("educate_mode", "none")
        if educate not in ("none", "deep"):
            raise ValueError(
                f"educate_mode must be none|deep, got {educate!r}")
        self.educate_mode = educate
        self.educate_burst_ratio = float(config.get("educate_burst_ratio", 0.20))
        self.educate_burst_max_iter = int(config.get("educate_burst_max_iter", 1))
        self.educate_ls_budget = int(config.get("educate_ls_budget", 300))
        if self.educate_mode == "deep":
            # cx_stats 扩展 (仅 deep 记录教育结果; 默认 none 键集不变)
            self.cx_stats["educate_improved"] = False
            self.cx_stats["educate_delta"] = 0.0
            # 累计计数 (过程指标: 教育触发/成功比例)
            self.cx_stats["educate_attempts"] = 0
            self.cx_stats["educate_success"] = 0
        # σ 键 (L250 下) 从未接线: shaw_removal 以三参签名 (sol,k,rng) 注册,
        # σ 恒用函数默认 10.0 (2026-09-08 验收确认; 全 runner 显式传 3.0 亦被忽略)。
        # 注意: 不可"修复"接线 — 让 config 生效会使全部历史实验口径从 σ=10 迁移
        # 到 3。σ 键保留为文档占位, 论文口径如实写 σ=10.0。
        self.sigma = config.get("sigma", 3.0)  # 文档占位 (未接线, 恒 10.0)
        # ── 种群壳 (M1, 2026-09-07): population_mode ─────────────
        # 默认 False → 主循环逐位不变。μ 成员池 + 锦标赛 (fitness = cost −
        # κ×池内最优成本×dc) + 池级停滞刷新最劣成员 (M1 无重组/教育/罚域)。
        # population_stats / population_final 默认 None → base 路径键集不变。
        self.population_mode = bool(config.get("population_mode", False))
        self.population_size = int(config.get("population_size", 8))
        self.population_tournament = int(config.get("population_tournament", 2))
        self.population_dc_kappa = float(config.get("population_dc_kappa", 0.0))
        # 成员级驱逐开关 (C4 对照, 2026-09-09): False = 关闭成员级停滞驱逐
        # (停滞成员不再被档案精英替换, 池只由可行子代替换驱动) — 默认 True =
        # M1 裁定 2 行为逐位不变。
        self.population_member_refresh = bool(
            config.get("population_member_refresh", True))
        self.population_stats = None
        self.population_final = None
        # 成员级停滞计数状态 (裁定 2, 2026-09-07): run 内从全 0 起步; 白盒测试
        # 可经 _member_stagnation_override 在 run 前预置 ≥ 阈值计数。
        self._member_stagnation: list[int] | None = None
        self._member_stagnation_override: list[int] | None = None
        # timefix (2026-09-12): 时间预算防饥饿引导状态 (run 内重置; 判定见
        # _timefix_should_fire — 仅时间预算模式, 迭代模式恒不触发)。
        self._timefix_attempts = 0
        self._timefix_last_iter = -(10 ** 9)
        if self.population_mode:
            if self.population_size < 1:
                raise ValueError(
                    "population_size must be >= 1 when population_mode is on")
            if self.population_tournament < 1:
                raise ValueError(
                    "population_tournament must be >= 1 when population_mode is on")
            if self.population_crossover:
                raise ValueError("population_mode M1 forbids population_crossover")
            if self.educate_mode != "none":
                raise ValueError("population_mode M1 forbids educate_mode")
            if self.rebalance_mode != "none":
                raise ValueError("population_mode M1 forbids rebalance_mode")
            if self.soft_capacity:
                raise ValueError("population_mode M1 forbids soft_capacity")
            if self.soft_tw:
                raise ValueError("population_mode M1 forbids soft_tw")
            # 裁定 3 (2026-09-07): 池级停滞只实现 archive 语义; 镜像 base 的
            # current 语义 M1 未测禁开。
            if self.stagnation_source == "current":
                raise ValueError(
                    "population_mode M1 forbids stagnation_source=current")
        # ── 池内重组 (M2, 2026-09-07): population_cx_mode ────────
        # 默认 "none" → M1 行为逐位不变。route = 路由级继承 (route_copy_
        # crossover, A 臂); partition = 车场簇继承 (partition_crossover,
        # B 臂)。rate = 每迭代触发概率 (mode ≠ none 才生效); inherit_prob =
        # 仅 route 臂传给 route_copy_crossover 的继承率 (partition 不用)。
        # cx 统计键仅 mode ≠ none 时存在 (population_stats 键集断言)。
        cx_mode = config.get("population_cx_mode", "none")
        if cx_mode not in ("none", "route", "partition"):
            raise ValueError(
                f"population_cx_mode must be none|route|partition, got {cx_mode!r}")
        self.population_cx_mode = cx_mode
        self.population_cx_rate = float(config.get("population_cx_rate", 0.2))
        self.population_cx_inherit_prob = float(
            config.get("population_cx_inherit_prob", 0.6))
        if cx_mode != "none":
            if not self.population_mode:
                raise ValueError(
                    "population_cx_mode requires population_mode=True")
            # M2 未测禁开 (与 M1 壳守卫同集 — M1 守卫先抛, 此处双保险)
            if self.population_crossover:
                raise ValueError(
                    "population_cx_mode M2 forbids population_crossover")
            if self.educate_mode != "none":
                raise ValueError("population_cx_mode M2 forbids educate_mode")
            if self.rebalance_mode != "none":
                raise ValueError("population_cx_mode M2 forbids rebalance_mode")
            if self.soft_capacity:
                raise ValueError("population_cx_mode M2 forbids soft_capacity")
            if self.soft_tw:
                raise ValueError("population_cx_mode M2 forbids soft_tw")
            if self.stagnation_source == "current":
                raise ValueError(
                    "population_cx_mode M2 forbids stagnation_source=current")
        # ── 池内重组改进门+教育 (M3, 2026-09-07): population_cx_gate ──
        # gate: "none" = M2 的 SA+可行两级门 (pool_accept_replace) 逐位不变;
        # "improve" = 纯成本严格改进门 (pool_gate_improve) — 门过替换 m* 并跳过
        # 本迭代 destroy/repair, 门不过回退执行正常 destroy/repair (预算竞争修复)。
        # educate (仅 gate=improve+partition): cx 子代先进池内破坏性教育 (镜像
        # legacy _educate_child 结构, 预算封顶 population_educate_ls_budget) 再
        # 进门。默认 none/False → M2 行为逐位不变 (键集断言 + 全量回归)。
        cx_gate = config.get("population_cx_gate", "none")
        if cx_gate not in ("none", "improve"):
            raise ValueError(
                f"population_cx_gate must be none|improve, got {cx_gate!r}")
        self.population_cx_gate = cx_gate
        self.population_cx_educate = bool(
            config.get("population_cx_educate", False))
        self.population_educate_burst_ratio = float(
            config.get("population_educate_burst_ratio", 0.20))
        self.population_educate_ls_budget = int(
            config.get("population_educate_ls_budget", 50))
        if cx_gate != "none":
            # gate 只与车场簇重组配对测过; mode≠partition → ValueError
            if cx_mode != "partition":
                raise ValueError(
                    "population_cx_gate=improve requires "
                    "population_cx_mode=partition (only paired with it)")
            if not self.population_mode:
                raise ValueError(
                    "population_cx_gate requires population_mode=True")
        if self.population_cx_educate:
            if cx_gate != "improve":
                raise ValueError(
                    "population_cx_educate=True requires "
                    "population_cx_gate=improve")
            if cx_mode != "partition":
                raise ValueError(
                    "population_cx_educate=True requires "
                    "population_cx_mode=partition")
        # ── 双空间罚域 (M4, 2026-09-07): population_penalty_mode ─────
        # 默认 "none" → popcxge (M3) 行为逐位不变。tw = 在 popcxge 形态上开
        # TW 罚域成员通道 (容量/车辆数/覆盖仍硬, 镜像 legacy soft_tw 的 "TW 转
        # 罚" 边界): λ0 静态校准 (mult × best_feas / max(1, excess 中位)) + 罚后
        # 接受门 (pool_penalty_gate) + 锦标赛/驱逐统一罚后成本。档案仍只收可行。
        # 只测 gate=improve + partition 形态上的罚域 (其它组合禁开)。
        self.population_penalty_mode = config.get(
            "population_penalty_mode", "none")
        if self.population_penalty_mode not in ("none", "tw"):
            raise ValueError(
                "population_penalty_mode must be none|tw, got "
                f"{self.population_penalty_mode!r}")
        self.population_penalty_mult = float(
            config.get("population_penalty_mult", 1.0))
        if self.population_penalty_mode == "tw":
            if not self.population_mode:
                raise ValueError(
                    "population_penalty_mode=tw requires population_mode=True")
            if self.soft_capacity:
                raise ValueError(
                    "population_penalty_mode=tw forbids soft_capacity")
            if self.soft_tw:
                raise ValueError(
                    "population_penalty_mode=tw forbids soft_tw")
            if self.educate_mode != "none":
                raise ValueError(
                    "population_penalty_mode=tw forbids educate_mode")
            if self.population_cx_gate != "improve":
                raise ValueError(
                    "population_penalty_mode=tw requires "
                    "population_cx_gate=improve")
            if self.population_cx_mode != "partition":
                raise ValueError(
                    "population_penalty_mode=tw requires "
                    "population_cx_mode=partition")
        # λ0 状态 (tw 模式 run 内校准后有效; none 恒 0)
        self._penalty_lambda0 = 0.0
        checkpoint_interval = config.get("archive_checkpoint_interval", 0)
        if (
            isinstance(checkpoint_interval, (bool, np.bool_))
            or not isinstance(checkpoint_interval, (Integral, np.integer))
            or checkpoint_interval < 0
        ):
            raise ValueError("archive_checkpoint_interval must be a nonnegative integer")
        self.archive_checkpoint_interval = int(checkpoint_interval)

        # 回调函数
        self.callbacks = {}

        # 回退追踪
        self._fallback_count: dict[str, int] = {}
        # 崩溃追踪
        self._op_crash_count: dict[str, int] = {}
        self._op_call_count: dict[str, int] = {}
        self._op_removed: set[str] = set()

        # 轨迹日志
        self.trajectory = []
        self.archive_checkpoints = []
        self.initial_archive_checkpoint: dict | None = None
        self.pre_final_local_search_checkpoint: dict | None = None
        self.iteration = 0
        self._initial_objective: Objective | None = None
        context_window = config.get("context_window", 50)
        self._recent_accepts = deque(maxlen=context_window)
        self._recent_archive_adds = deque(maxlen=context_window)
        # 探针观察计数 (2026-09-05 score 解耦判定诊断, 纯记录不改行为)
        self.probe_stats = {"adds": 0, "accepts": 0, "equal_accepts": 0,
                            "restarts": 0, "stagnation_peak": 0}
        self._recent_improvements = deque(maxlen=context_window)

        # 最优解追踪
        self._best_sol: Solution | None = None
        self._best_obj: Objective | None = None

    def register_callback(self, name: str, func):
        """注册回调函数：on_iteration、on_new_archive 等。"""
        self.callbacks[name] = func

    def run(self) -> ParetoArchive:
        """
        运行 ALNS 搜索并返回 Pareto 档案。
        """
        run_started = time.perf_counter()
        if self.population_mode:
            return self._run_population_mode(run_started)
        self.archive_checkpoints = []
        self.initial_archive_checkpoint = None
        self.pre_final_local_search_checkpoint = None

        # 使用继承的解或构建新的初始解
        if self._initial_solution is not None:
            current_sol = self._initial_solution.copy()
        else:
            current_sol = build_initial_solution(self.instance, self.rng, self.init_method)
        # 初始解统一可行性兜底: 容量 + TW + 车辆数 (warm-start 解可能违规;
        # 曾只 _enforce_tw — 车辆数上限软约束缺陷 2026-09-01 修复)
        current_sol = self._enforce_feasibility(current_sol)
        current_obj = calculate_objectives(current_sol)
        self._initial_objective = current_obj

        # 软容量: 罚系数 λ0 = mult × 初始成本 / 总需求 (单位超载 ≈ 平均单位
        # 需求成本); repair 插入成本与接受目标共用同一 λ (实例属性透传);
        # 播种失败 λ ×1.5 上调 (拉回可行域), 成功 ÷1.5 回落 (λ0 下限)。
        if self.soft_capacity:
            td = float(self.instance.total_demand)
            self._soft_lam0 = (
                self.soft_capacity_penalty_mult * current_obj[0] / td if td > 0 else 1.0
            )
            self._soft_lam_max = self._soft_lam0 * 100.0
            self._set_soft_lam(self._soft_lam0)

        # 软 TW: 罚系数 λ0 = mult × 初始可行成本 / 客户数 (每客户平均成本 ≈
        # 每单位时间违规的罚金量级); 播种失败 λ ×1.5 上调 (拉回可行域), 入档
        # 成功 ÷1.5 回落 (λ0 下限)。完全镜像 soft_capacity 的 λ 管理。
        if self.soft_tw:
            n = float(self.instance.num_customers)
            self._soft_tw_lam0 = (
                self.soft_tw_lambda_mult * current_obj[0] / n if n > 0 else 1.0
            )
            self._soft_tw_lam_max = self._soft_tw_lam0 * 100.0
            self._set_soft_tw_lam(self._soft_tw_lam0)

        # ── SA 自动校准 ──────────────────────────────────────
        # 未显式提供初始温度时，用初始解目标值校准 T₀ 与冷却率
        if getattr(self.acceptor, "_auto_calibrate", False):
            self.acceptor.calibrate(current_obj, self.max_iterations)

        # 最优解追踪
        self._best_sol = current_sol.copy()
        self._best_obj = current_obj

        # 加入档案 (纯净性门禁: 初始解经兜底链仍可能带 TW 违规 — pr11 紧车
        # 宽窗实测 tw_exc 317; Pareto 模式靠支配掩盖, 单目标精英池暴露)
        if current_sol.is_feasible():
            self._sync_depot_assignments(current_sol)
            self.archive.add(current_obj, {
                "iteration": 0,
                "type": "initial",
                "solution": current_sol.copy(),
            })
        self.initial_archive_checkpoint = {
            "iteration": 0,
            "elapsed_seconds": 0.0,
            "objectives": [[float(current_obj[0]), float(current_obj[1])]],
        }

        stagnation_count = 0
        # timefix (2026-09-12): 预算锚 = run 入口 (run_started), 覆盖构造与
        # 可行化初始化耗时 — 旧锚在循环开始, durfix 后昂贵的初始化 (种群
        # 20~103s) 全部逃逸预算 (时间预算运行实测超时 +22~+97s, 对照失真)。
        # 迭代模式 max_time=1e9 → 检查不可达, 轨迹逐位不变。
        start_time = run_started
        last_completed_iteration = 0

        for iteration in range(1, self.max_iterations + 1):
            self.iteration = iteration
            # 时间检查
            if time.perf_counter() - start_time > self.max_time:
                break

            # 停滞检查
            if stagnation_count >= self.stagnation_limit:
                break

            # 停滞恢复：卡住时从新初始解重启搜索 (V4: 可清空档案,
            # 检验"档案记忆拖累重启"假设)
            recovery_threshold = max(30, self.stagnation_limit
                                      // self.stagnation_recovery_divisor)
            if stagnation_count == recovery_threshold:
                self.probe_stats["restarts"] += 1
                current_sol = build_initial_solution(self.instance, self.rng, self.init_method)
                current_obj = calculate_objectives(current_sol)
                if self.restart_clear_archive:
                    self.archive.clear()
                self.acceptor.reset()
                stagnation_count = 0

            # 更新最优解
            if self._best_obj is None or (current_obj[0] + current_obj[1]) < (self._best_obj[0] + self._best_obj[1]):
                self._best_sol = current_sol.copy()
                self._best_obj = current_obj

            # 父代选择
            parent_entry = self._select_parent()
            if parent_entry is None:
                parent_sol = current_sol
            else:
                parent_sol = self._reconstruct_solution(parent_entry)

            context = self._build_search_context(
                iteration, stagnation_count, current_sol, current_obj,
            )
            selector_started = time.perf_counter()
            action = self.selector.select_pair(context, self.rng)
            selector_elapsed = time.perf_counter() - selector_started
            if not np.isfinite(selector_elapsed) or selector_elapsed < 0.0:
                raise ValueError("selector elapsed time must be finite and nonnegative")
            decision_keys = (
                "predicted_mean",
                "uncertainty",
                "selection_score",
                "action_count_before",
            )
            decision_source = None
            if hasattr(self.selector, "get_last_decision"):
                decision_source = self.selector.get_last_decision()
            decision = {
                key: None if decision_source is None else self._json_snapshot(
                    decision_source.get(key)
                )
                for key in decision_keys
            }
            decision["selector_elapsed_seconds"] = float(selector_elapsed)
            decision["factorized"] = (
                None
                if decision_source is None
                else self._json_snapshot(decision_source.get("factorized"))
            )
            if decision_source is not None and "hierarchical" in decision_source:
                decision["hierarchical"] = self._json_snapshot(
                    decision_source["hierarchical"]
                )
            d_idx, r_idx = action
            if not 0 <= d_idx < len(self.destroy_names):
                raise IndexError(
                    f"{type(self.selector).__name__} returned invalid destroy index {d_idx}"
                )
            if not 0 <= r_idx < len(self.repair_names):
                raise IndexError(
                    f"{type(self.selector).__name__} returned invalid repair index {r_idx}"
                )
            d_idx, r_idx = int(d_idx), int(r_idx)
            destroy_name = self.destroy_names[d_idx]
            repair_name = self.repair_names[r_idx]

            destroy_fn = self.destroy_ops[destroy_name]
            repair_fn = self.repair_ops[repair_name]

            # 追踪调用
            self._track_operator_call(destroy_name)
            self._track_operator_call(repair_name)

            # 计算破坏规模 k
            k = self._compute_k(iteration, stagnation_count)

            # 标记本次迭代是否发生崩溃
            destroy_crashed = False
            repair_crashed = False
            operator_started = time.perf_counter()

            # ── 种群交叉迭代 (探针 2026-09-04) ─────────────────
            # 双亲路线混合生成后代, 替代本迭代的 destroy/repair; 后代走公共
            # LS/accept/档案链 (被接受时 ls_on_accept 即教育)。交叉迭代不更新
            # selector 权重 (学习状态少一次更新)。默认关 → 行为逐位不变。
            cx_used = False
            child_cx = None
            rebalance_removal = False
            if (self.population_crossover and self.archive.size >= 2
                    and self.rng.random() < self.crossover_rate):
                child_cx = self._crossover_child()
            if child_cx is not None:
                cx_used = True
                if self.educate_mode == "deep":
                    # 破坏性教育 (方向②, gated): cx 后代先破坏-重建再准入;
                    # 教育不劣化则用 child_educ, 否则退回 child_cx。
                    new_sol = self._educate_child(child_cx)
                else:
                    new_sol = child_cx
            else:
                # ── 车场分配重平衡 (方向①, gated) ─────────────
                # redispatch: 周期错配修复 pass 作用于 parent (独立后处理,
                # 不更新 selector); removal: 周期用 mismatched_removal 替代
                # 正常 destroy 选择 (同 cx 惯例不更新 selector 权重)。
                if self._rebalance_redispatch_on(iteration):
                    parent_sol = self._redispatch_pass(parent_sol)
                if self._rebalance_removal_on(iteration):
                    rebalance_removal = True
                    from src.operators.destroy import mismatched_removal
                    k_rb = max(1, int(round(self.rebalance_ratio
                                            * self.instance.num_customers)))
                    partial_sol, removed = parent_sol.apply_destroy(
                        mismatched_removal, k_rb, self.rng)
                    self.rebalance_stats["removal_passes"] += 1
                else:
                    # 应用破坏（算子崩溃直接抛出，不做 LLM 特判回退）
                    partial_sol, removed = parent_sol.apply_destroy(
                        destroy_fn, k, self.rng)
                if not isinstance(removed, list):
                    raise TypeError(f"Destroy returned {type(removed).__name__}, expected list")

                # 应用修复（算子崩溃直接抛出，不做 LLM 特判回退）
                new_sol = partial_sol.apply_repair(repair_fn, removed, self.rng)
                if new_sol is None or not hasattr(new_sol, "routes"):
                    raise TypeError(f"Repair returned {type(new_sol).__name__}, expected Solution")

            # 可行化公共链 (repair 与 cx 后代共用): 未分配补插 → 容量/车辆
            # 硬约束 (软容量模式只保车辆数) → TW 硬约束
            new_sol = self._postprocess_after_repair(new_sol)

            # ── 局部搜索调度（频率控制）───────────────────────
            do_ls = False
            if self.local_search_fn is not None and self.ls_freq > 0:
                if iteration % self.ls_freq == 0:
                    do_ls = True
            if do_ls:
                if (self.soft_capacity and not new_sol.is_feasible()) or (
                        self.soft_tw and new_sol.tw_violations()):
                    # 软容量: RVND 只接受完全可行改进 — 超载解上 LS 会空转,
                    # 跳过 (超载重组由 destroy/repair 承担, 周期播种负责修复)
                    # 软 TW: 同 — TW 违规解上 RVND 空转, 跳过 (违规重组由
                    # destroy/repair 承担, 周期播种负责修复)
                    pass
                else:
                    new_sol = self.local_search_fn(new_sol, max_iterations=self.ls_max_iter,
                                                   rng=self.rng)
                    if self.soft_capacity:
                        if new_sol.instance.has_time_windows and new_sol.tw_violations():
                            new_sol = self._enforce_tw(new_sol)
                    else:
                        # LS 后可行性兜底 (容量 + TW; 跨路由邻域可能违反容量)
                        new_sol = self._enforce_feasibility(new_sol)

            new_obj = calculate_objectives(new_sol)
            if self.soft_capacity:
                new_obj = self._soft_penalize(new_sol, new_obj)
            if self.soft_tw:
                new_obj = self._soft_tw_penalize(new_sol, new_obj)

            parent_obj = current_obj

            # 接受准则 (软容量: 比较罚后目标 — 超载穿越以罚金代价被接受)
            accepted = self.acceptor.accept(current_obj, new_obj, self.rng)
            if accepted:
                self.probe_stats["accepts"] += 1
                if abs(new_obj[0] - parent_obj[0]) < 1e-9:
                    self.probe_stats["equal_accepts"] += 1
                current_sol = new_sol
                current_obj = new_obj
                # ── 接受时强制 LS ─────────────────────────────
                if (self.local_search_fn is not None and self.ls_on_accept
                        and not do_ls):  # 避免重复
                    if (self.soft_capacity and not current_sol.is_feasible()) or (
                            self.soft_tw and current_sol.tw_violations()):
                        pass  # 同上: 超载/TW 违规解跳过 LS 精化
                    else:
                        current_sol = self.local_search_fn(current_sol, max_iterations=self.ls_max_iter,
                                                           rng=self.rng)
                        if self.soft_capacity:
                            if current_sol.instance.has_time_windows and current_sol.tw_violations():
                                current_sol = self._enforce_tw(current_sol)
                        else:
                            # LS 后可行性兜底 (容量 + TW; 跨路由邻域可能违反容量)
                            current_sol = self._enforce_feasibility(current_sol)
                        current_obj = calculate_objectives(current_sol)
                        if self.soft_capacity:
                            current_obj = self._soft_penalize(current_sol, current_obj)
                        elif self.soft_tw:
                            current_obj = self._soft_tw_penalize(current_sol, current_obj)
                        new_sol = current_sol
                        new_obj = current_obj

            # 档案更新 (纯净性: 只收可行解 — 兜底链可能残留容量/TW 违规,
            # 违规解曾混入档案并支配可行解, 2026-09-02 pr11 实测)
            self._sync_depot_assignments(new_sol)
            if new_sol.is_feasible():
                added = self.archive.add(new_obj, {
                    "iteration": iteration,
                    "destroy": destroy_name,
                    "repair": repair_name,
                    "solution": new_sol.copy(),
                })
            else:
                added = False
            last_completed_iteration = iteration
            if added:
                self.probe_stats["adds"] += 1
                cb = self.callbacks.get("on_archive_add")
                if cb is not None:
                    cb(iteration, destroy_name, repair_name, new_obj[0])
            if (
                self.archive_checkpoint_interval > 0
                and iteration % self.archive_checkpoint_interval == 0
            ):
                self._record_archive_checkpoint(iteration, run_started)

            if self.stagnation_source == "current":
                # 修复候选: 停滞 = current 质量停滞 — accepted 且 obj 严格改进
                # (vs 接受前 current) 才清零; 拒收/平台/等值接受均累计。
                if accepted and new_obj[0] < parent_obj[0] - 1e-9:
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                if added and "on_new_archive" in self.callbacks:
                    self.callbacks["on_new_archive"](iteration, self.archive)
            elif added:
                stagnation_count = 0
                if "on_new_archive" in self.callbacks:
                    self.callbacks["on_new_archive"](iteration, self.archive)
            else:
                stagnation_count += 1

            self.probe_stats["stagnation_peak"] = max(
                self.probe_stats["stagnation_peak"], stagnation_count)

            # ── 软容量: 周期修复播种 ──────────────────────────
            # 超载轨迹中可行解稀少 → 定期把当前解强制修复回可行并尝试入档
            # (档案保持可行精英作父代 + 给搜索可行性压力反馈)
            if (self.soft_capacity and self.soft_repair_interval > 0
                    and iteration % self.soft_repair_interval == 0):
                self._soft_seed(current_sol, iteration)
                cand = self._soft_seed_candidate
                if cand is not None and (
                        self._cap_excess(current_sol) > 1e-9
                        or self._soft_seed_obj[0] < current_obj[0]):
                    # 采纳可行锚点 (轨迹超载 → 锚回可行域; 或修复解真实成本更优)
                    current_sol = cand
                    current_obj = self._soft_seed_obj  # 可行解: 罚后目标 = 真实目标
                    stagnation_count = 0

            # ── 软 TW: 周期修复播种 ───────────────────────────
            # TW 违规轨迹中可行解稀少 → 定期把当前解 _enforce_tw 强制修复回
            # 可行并尝试入档 (档案保持可行精英作父代 + 给搜索可行性压力反馈)
            if (self.soft_tw and self.soft_tw_repair_interval > 0
                    and iteration % self.soft_tw_repair_interval == 0):
                self._soft_tw_seed(current_sol, iteration)
                cand = self._soft_tw_seed_candidate
                if cand is not None and (
                        len(current_sol.tw_violations()) > 0
                        or self._soft_tw_seed_obj[0] < current_obj[0]):
                    # 采纳可行锚点 (轨迹违规 → 锚回可行域; 或修复解真实成本
                    # 更优 — current_obj 可能为罚后目标)
                    current_sol = cand
                    current_obj = self._soft_tw_seed_obj  # 可行解: 罚后目标 = 真实目标
                    self.soft_tw_stats["seed_adopted"] += 1
                    stagnation_count = 0

            # ── score 语义 (探针 2026-09-05) ────────────────────
            # admission (默认): 入档成功 2 / 接受未入档 1 / 拒 0 — 平台期
            # add 常假 → score 0 → 常用算子权重被逐段压向地板 (倒置劣化)。
            # activity: 可行且未崩溃即 2.0 (探索活动反馈, 与准入/接受解耦)。
            # quality: 只按质量阶梯 — new<轨迹 best 2.0 / new<parent 1.5 /
            # accepted (等值或 SA 更差) 1.0 / 拒 0。
            scheme = self.score_scheme
            if scheme == "activity":
                score = (
                    0.0
                    if (destroy_crashed or repair_crashed)
                    else (2.0 if new_sol.is_feasible() else 0.0)
                )
            elif scheme == "quality":
                if destroy_crashed or repair_crashed:
                    score = 0.0
                elif new_obj[0] < self._best_obj[0] - 1e-9:
                    score = 2.0
                elif new_obj[0] < parent_obj[0] - 1e-9:
                    score = 1.5
                elif accepted:
                    score = 1.0
                else:
                    score = 0.0
            else:  # admission — 默认, 表达式与旧实现逐位一致
                score = 2.0 if added else (1.0 if accepted else 0.0)
            reward = float(score)  # 连续奖励已随学习型选择器一并移除，保留 score 语义
            elapsed = time.perf_counter() - operator_started
            dominates_parent = (
                new_obj[0] <= parent_obj[0]
                and new_obj[1] <= parent_obj[1]
                and new_obj != parent_obj
            )
            outcome = OperatorOutcome(
                accepted=accepted,
                archive_added=added,
                dominates_parent=dominates_parent,
                feasible=new_sol.is_feasible(),
                crashed=destroy_crashed or repair_crashed,
                score=score,
                reward=reward,
                elapsed_seconds=elapsed,
                destroy_crashed=destroy_crashed,
                repair_crashed=repair_crashed,
            )
            self._recent_accepts.append(accepted)
            self._recent_archive_adds.append(added)
            self._recent_improvements.append(max(0.0, reward))
            next_context = self._build_search_context(
                iteration, stagnation_count, current_sol, current_obj,
            )
            normalized_reward = (
                0.0
                if outcome.crashed
                else float(np.clip(outcome.score / 2.0, 0.0, 1.0))
            )
            # 交叉迭代未执行所选算子 → 不更新 selector 权重 (学习状态保留);
            # rebalance removal 迭代同 cx 惯例: destroy 被 mismatched 替代,
            # 不更新所选算子的权重。
            if not cx_used and not rebalance_removal:
                self.selector.update(action, outcome, next_context)

            # 降温
            self.acceptor.cool()

            # 轨迹日志
            outcome_payload = {
                "accepted": outcome.accepted,
                "archive_added": outcome.archive_added,
                "feasible": outcome.feasible,
                "crashed": outcome.crashed,
                "discrete_score": outcome.score,
                "normalized_reward": normalized_reward,
                "elapsed_seconds": outcome.elapsed_seconds,
                "cost": current_obj[0],
                "emission": current_obj[1],
                "archive_size": self.archive.size,
            }
            if self.soft_capacity:
                cur_exc = self._cap_excess(current_sol)
                outcome_payload["current_cap_excess"] = float(cur_exc)
                self.soft_stats["max_excess"] = max(
                    self.soft_stats["max_excess"], float(cur_exc)
                )
                if cur_exc > 1e-9:
                    self.soft_stats["overloaded_iters"] += 1
                    # HGS 式快反馈: 轨迹超载 → λ ×1.05/迭代 (压回可行域),
                    # 恢复可行后 ÷1.05 回落 — 防止超载不受控漂移 (v1 实测
                    # 300/300 迭代超载; 步长 1.02 太慢, 参数扫描定 1.05)
                    if self._soft_lam < self._soft_lam_max:
                        self._set_soft_lam(
                            min(self._soft_lam_max, self._soft_lam * 1.05)
                        )
                elif self._soft_lam > self._soft_lam0:
                    self._set_soft_lam(
                        max(self._soft_lam0, self._soft_lam / 1.05)
                    )
            if "calibration_update" in decision:
                outcome_payload["dominates_parent"] = bool(
                    dominates_parent
                )
            self.trajectory.append({
                "iteration": iteration,
                "context": asdict(context),
                "decision": decision,
                "action": {
                    "destroy_index": d_idx,
                    "repair_index": r_idx,
                    "destroy_name": destroy_name,
                    "repair_name": repair_name,
                    "flat_action_index": d_idx * len(self.repair_names) + r_idx,
                    "cold_start": (
                        None
                        if decision["action_count_before"] is None
                        else decision["action_count_before"] == 0
                    ),
                },
                "outcome": outcome_payload,
            })

            # 迭代回调
            if "on_iteration" in self.callbacks:
                self.callbacks["on_iteration"](iteration, current_obj, self.archive)

        # ── 软容量: 终局修复 — 轨迹可能停在超载解, 强制修复回可行并入档
        #    (TW 实例 final LS 跳过, 修复后可行解即最终输出候选) ─────────
        if self.soft_capacity:
            candidate = current_sol.copy()
            self._sync_depot_assignments(candidate)
            candidate = self._soft_repair_hard(candidate)
            self._sync_depot_assignments(current_sol)  # 共享 instance 状态还原
            if candidate.is_feasible():
                self.archive.add(calculate_objectives(candidate), {
                    "iteration": last_completed_iteration,
                    "type": "final_soft_repair",
                    "solution": candidate.copy(),
                })

        # ── 最终 LS 精化档案 ──────────────────────────────────
        if self.local_search_fn is not None and self.ls_final and not self.archive.is_empty():
            if last_completed_iteration > 0:
                checkpoint = next(
                    (
                        item
                        for item in reversed(self.archive_checkpoints)
                        if item["iteration"] == last_completed_iteration
                    ),
                    None,
                )
                if checkpoint is None:
                    checkpoint = self._archive_checkpoint_snapshot(
                        last_completed_iteration, run_started
                    )
                self.pre_final_local_search_checkpoint = self._json_snapshot(
                    checkpoint
                )
            self._apply_final_local_search()

        if self.archive_checkpoint_interval > 0 and last_completed_iteration > 0:
            self._record_archive_checkpoint(
                last_completed_iteration,
                run_started,
                replace_same_iteration=True,
            )

        # 探针暴露: 轨迹末端解 (2026-09-05 F 探针 — 罚域末端硬修复口径; 行为中性)
        self.last_solution = current_sol.copy()
        self.last_obj = tuple(current_obj)

        return self.archive

    def _run_population_mode(self, run_started: float) -> ParetoArchive:
        """种群壳主循环 (M1, 2026-09-07): μ 成员池 + 锦标赛 + 池级刷新。

        逐项镜像 run() 的 base 语义, 差异只来自"多成员":
        - 池成员 = 完整可行解 (裁定 1: 两级接受 pool_accept_replace — SA 接受
          AND 子代可行才替换成员; SA 接受但不可行 → 按拒绝记账); 每迭代锦标赛
          选 1 成员 (engine.rng) 作演化对象 (父代源 = 池, 不调用 _select_parent);
        - 裁定 2: init/刷新重建走 _build_pool_member 重试兜底 (至多 10 次,
          全败取成本最低候选入池 — 残余违规实例不阻塞, 成员可被锦标赛演化自愈);
        - 接受且可行 → 该成员解/obj 整体替换 (新解副本); 否则 → 池不变;
        - 档案 = 精英池 (共享, 语义与 base 一致); 池级停滞 → 刷新最劣成员
          (只换 1 个, 不重置其余成员与轨迹);
        - M1 禁开 cx/educate/rebalance/soft_capacity/soft_tw → 对应分支不执行。
        确定性: 全部随机取自 engine.rng; 路由键集只用于 dc/统计。
        """
        mu = int(self.population_size)
        self.archive_checkpoints = []
        self.initial_archive_checkpoint = None
        self.pre_final_local_search_checkpoint = None

        # 1. 初始化池: engine.rng → build_initial_solution → 可行化 → obj
        #    (裁定 2: 建成员走 _build_pool_member 重试兜底, 至多 10 次)
        pool_sol: list[Solution] = []
        pool_obj: list[Objective] = []
        pool_keys: list[frozenset] = []
        for _ in range(mu):
            sol, obj = self._build_pool_member()
            pool_sol.append(sol.copy())
            pool_obj.append(obj)
            pool_keys.append(ParetoArchive._route_keys({"solution": sol}))

        # ── 罚域 (M4, 2026-09-07): λ0 静态校准 + 成员罚后成本 ─────
        # λ0 = mult × best_feas / max(1.0, excess_med); best_feas = 池内最优
        # 可行成员真实成本 (全池不可行的兜底取池内最小成本); excess_med = 初始
        # 不可行成员 (10 次重试全败的兜底候选) 的 _tw_excess 中位数, 无则 1.0。
        # pool_pen[i] = 成员罚后成本: 可行 = 真实成本, 不可行 = cost + λ0×excess
        # (λ0 静态贯穿全程)。penalty_mode=none → pool_pen None / 下方逐位走 M3。
        pen_tw = self.population_penalty_mode == "tw"
        pool_pen: list[float] | None = None
        pen_lambda0 = 0.0
        if pen_tw:
            _feas_cost = [
                float(pool_obj[i][0])
                for i in range(mu) if pool_sol[i].is_feasible()
            ]
            if _feas_cost:
                _best_feas = min(_feas_cost)
            else:
                _best_feas = min(float(pool_obj[i][0]) for i in range(mu))
            _excesses = [
                self._tw_excess(pool_sol[i])
                for i in range(mu) if not pool_sol[i].is_feasible()
            ]
            _excess_med = float(np.median(_excesses)) if _excesses else 1.0
            pen_lambda0 = (
                self.population_penalty_mult * _best_feas
                / max(1.0, _excess_med))
            self._penalty_lambda0 = pen_lambda0
            pool_pen = [
                pool_penalized_cost(
                    float(pool_obj[i][0]),
                    self._tw_excess(pool_sol[i]), pen_lambda0)
                for i in range(mu)
            ]

        def _pen_cost(sol: "Solution", obj: Objective) -> float:
            """成员罚后成本 (tw 模式): 经 pool_penalized_cost 纯函数计算。"""
            return pool_penalized_cost(
                float(obj[0]), self._tw_excess(sol), pen_lambda0)

        def _postprocess_tw(sol: "Solution") -> "Solution":
            """可行化公共链 tw 变体 (M4 规格 §2.4): 未分配补插 → 容量/车辆数
            硬兜底, 跳过 TW 强制 (违规解流到罚后评价)。镜像
            _postprocess_after_repair 链结构但去掉 _enforce_tw — 本体零改动,
            base/legacy/M3 路径仍走原方法。"""
            unassigned = sol.unassigned_customers()
            if unassigned:
                sol = self._repair_unassigned(sol, unassigned)
            if not sol.is_feasible():
                sol = self._enforce_capacity(sol)
                if not sol.is_feasible():
                    sol = self._enforce_vehicle_limit(sol)
            # 罚域只松弛 TW, 不松弛 duration (规格 §4.1); 候选路径小预算
            sol = self._enforce_duration(sol, work_budget=15000,
                                         step_cap=1500)
            return sol

        _postprocess = _postprocess_tw if pen_tw else self._postprocess_after_repair

        # 2. SA 校准: 用池内最优成员 obj (base 用初始解 obj — 一致化到池最优)
        best_i = min(range(mu),
                     key=lambda i: (pool_obj[i][0] + pool_obj[i][1], i))
        self._initial_objective = pool_obj[best_i]
        if getattr(self.acceptor, "_auto_calibrate", False):
            self.acceptor.calibrate(pool_obj[best_i], self.max_iterations)
        self._best_sol = pool_sol[best_i].copy()
        self._best_obj = pool_obj[best_i]

        # 3. 档案初始播种: 只加池内最优可行成员 (iteration 0, population_init)
        if pool_sol[best_i].is_feasible():
            self._sync_depot_assignments(pool_sol[best_i])
            self.archive.add(pool_obj[best_i], {
                "iteration": 0,
                "type": "population_init",
                "solution": pool_sol[best_i].copy(),
            })
        self.initial_archive_checkpoint = {
            "iteration": 0,
            "elapsed_seconds": 0.0,
            "objectives": [[float(pool_obj[best_i][0]),
                            float(pool_obj[best_i][1])]],
        }

        self.population_stats = {
            "init_members": mu,
            "refresh_events": 0,
            "member_refresh_events": 0,
            "member_advances": {i: 0 for i in range(mu)},
            "dc_min_mean": 0.0,
            # timefix (2026-09-12): 时间预算防饥饿引导统计 — 仅时间预算模式
            # 可非零; 迭代模式恒 0 (M1 键集断言测试已同步更新)。
            "timefix_bootstrap_attempts": 0,
            "timefix_bootstrap_success": 0,
        }
        # M2 (2026-09-07): cx 统计键仅 population_cx_mode ≠ none 时存在 —
        # 全部累计整数 (教训: 机制过程统计必须累计, 禁只留最近值)。M1 既有
        # 键与语义零改动 (mode none → 键集不变)。
        if self.population_cx_mode != "none":
            self.population_stats["cx_attempts"] = 0
            self.population_stats["cx_children"] = 0
            self.population_stats["cx_duplicates"] = 0
            self.population_stats["cx_novel_route_keys"] = 0
            # M3 (2026-09-07): 改进门/教育统计键 — 仅相应 mode 开启时存在,
            # 全部累计 (教训: 机制过程统计必须累计, 禁只留最近值)。
            if self.population_cx_gate == "improve":
                self.population_stats["cx_gate_success"] = 0
                self.population_stats["cx_gate_fail"] = 0
            if self.population_cx_educate:
                self.population_stats["educate_attempts"] = 0
                self.population_stats["educate_success"] = 0
                self.population_stats["educate_delta_sum"] = 0.0
        # M4 (2026-09-07): 罚域统计键仅 population_penalty_mode=tw 时存在 —
        # penalty_member_iters / penalty_infeasible_final 累计整数,
        # lambda0 为静态校准值 (float)。mode none → 键集不变。
        if pen_tw:
            self.population_stats["penalty_member_iters"] = 0
            self.population_stats["penalty_infeasible_final"] = 0
            self.population_stats["lambda0"] = float(pen_lambda0)
        self.population_final = None

        pool_stagnation = 0
        # timefix (2026-09-12): 预算锚 = run 入口 (同 run() 普通路径); 种群
        # 初始化 20~103s 计入预算。迭代模式 max_time=1e9 → 不可达, 轨迹不变。
        start_time = run_started
        last_completed_iteration = 0
        last_selected = best_i
        dc_sum = 0.0
        n_iters = 0

        # 成员级停滞计数 (裁定 2): 每成员连续"未成功替换"迭代数。默认全 0
        # (每次 run 从头建池, 计数清零); 白盒测试经 _member_stagnation_override
        # 预置 ≥ 阈值计数以锁定驱逐路径。
        if self._member_stagnation_override is not None:
            member_stagnation = list(self._member_stagnation_override)
            self._member_stagnation_override = None
        else:
            member_stagnation = [0] * mu
        self._member_stagnation = member_stagnation
        # timefix: 引导状态 run 内重置 (同 _member_stagnation 语义)
        self._timefix_attempts = 0
        self._timefix_last_iter = -(10 ** 9)

        for iteration in range(1, self.max_iterations + 1):
            self.iteration = iteration
            member_reset: set[int] = set()  # 本迭代被替换/刷新的成员 → 计数清零
            # 时间检查
            if time.perf_counter() - start_time > self.max_time:
                break

            # 池级停滞检查 (break 语义 = base)
            if pool_stagnation >= self.stagnation_limit:
                break

            # 池级停滞恢复: 刷新最劣成员为新初始解 (只换 1 个, 不重置其余
            # 成员与轨迹; fitness 并列取序号小者)
            recovery_threshold = max(30, self.stagnation_limit
                                     // self.stagnation_recovery_divisor)
            if pool_stagnation == recovery_threshold:
                self.population_stats["refresh_events"] += 1
                costs_now = list(pool_pen) if pen_tw else [o[0] for o in pool_obj]
                best_cost_now = min(costs_now)
                worst = max(
                    range(mu),
                    key=lambda i: (
                        costs_now[i]
                        - self.population_dc_kappa * best_cost_now
                        * _pool_member_dc(i, pool_keys),
                        -i,
                    ),
                )
                # 裁定 2: 刷新重建走同一重试兜底 (至多 10 次)
                new_sol, new_obj = self._build_pool_member()
                pool_sol[worst] = new_sol.copy()
                pool_obj[worst] = new_obj
                pool_keys[worst] = ParetoArchive._route_keys({"solution": new_sol})
                if pool_pen is not None:
                    pool_pen[worst] = _pen_cost(new_sol, new_obj)
                # 成员级停滞: 被池级刷新替换的成员视为"自身被刷新" → 计数清零
                member_stagnation[worst] = 0
                member_reset.add(worst)
                if self.restart_clear_archive:
                    self.archive.clear()
                self.acceptor.reset()
                pool_stagnation = 0

            # ── 成员级停滞刷新 (裁定 2, 2026-09-07) ────────────
            # 每迭代开头 (时间/池停滞检查之后, 锦标赛之前) 扫描全体成员, 凡
            # member_stagnation ≥ member_refresh_threshold 者全部刷新。刷新动作:
            # 新成员 ← 档案精英副本 (_member_refresh_replacement: _select_parent
            # crowding 选择, 无 rng 消耗; 档案空 → _build_pool_member 兜底)。
            # 不重置 acceptor; 不改 pool_stagnation (全局池停滞仍只由 archive
            # add 驱动); 不动其它成员。驱逐路径: 尸体 (不可行/长期停滞成员) 被
            # 档案可行精英替代, 池保持"活成员为主"。
            # population_member_refresh=False (C4 对照, 2026-09-09): 整段跳过
            # (成员只被可行子代替换驱动), member_refresh_events 恒 0。
            if self.population_member_refresh:
                member_refresh_threshold = _member_refresh_threshold(
                    self.stagnation_limit)
                for _mi in range(mu):
                    if member_stagnation[_mi] < member_refresh_threshold:
                        continue
                    _ns, _no = self._member_refresh_replacement()
                    pool_sol[_mi] = _ns.copy()
                    pool_obj[_mi] = _no
                    pool_keys[_mi] = ParetoArchive._route_keys(
                        {"solution": _ns})
                    if pool_pen is not None:
                        pool_pen[_mi] = _pen_cost(_ns, _no)
                    member_stagnation[_mi] = 0
                    member_reset.add(_mi)
                    self.population_stats["member_refresh_events"] += 1

            # ── timefix 防饥饿引导 (2026-09-12) ──────────────────────
            # 档案空 + 池全不可行时, 刷新风暴 (档案空 → 每次重建 ~13s 且全
            # 败) 会烧光时间预算且零可行解 (pr20 s3 600s 实测确定性复现)。
            # 触发时对最优成员做重预算爬坡: 成功 → 可行替补换最劣成员 +
            # 播种档案 (档案非空后刷新走精英路径, 死锁解除)。
            if self._timefix_should_fire(iteration, start_time):
                self._timefix_attempts += 1
                self._timefix_last_iter = iteration
                self.population_stats["timefix_bootstrap_attempts"] += 1
                _replaced = self._timefix_feasibility_bootstrap(
                    pool_sol, pool_obj, pool_keys)
                if _replaced is not None:
                    self.population_stats["timefix_bootstrap_success"] += 1
                    member_reset.add(_replaced)

            # 更新最优解 (池内最优成员; base 同构标量 = obj[0]+obj[1])
            best_i = min(range(mu),
                         key=lambda i: (pool_obj[i][0] + pool_obj[i][1], i))
            if (self._best_obj is None
                    or (pool_obj[best_i][0] + pool_obj[best_i][1])
                    < (self._best_obj[0] + self._best_obj[1])):
                self._best_sol = pool_sol[best_i].copy()
                self._best_obj = pool_obj[best_i]

            # 锦标赛选成员: 先抽子集再比 fitness (engine.rng, 次序固定)。
            # tw 模式: 决策成本 = 罚后成本 (κ=0 臂即纯罚后), 驱逐/最劣刷新同源。
            costs_now = list(pool_pen) if pen_tw else [o[0] for o in pool_obj]
            best_cost_now = min(costs_now)
            selected = _pool_select_member(
                costs_now, pool_keys, best_cost_now, self.population_dc_kappa,
                self.population_tournament, self.rng)
            last_selected = selected
            dc_sum += _pool_member_dc(selected, pool_keys)
            n_iters += 1
            m_sol = pool_sol[selected]
            m_obj = pool_obj[selected]
            parent_sol = m_sol.copy()
            parent_obj = m_obj
            if pen_tw and not pool_sol[selected].is_feasible():
                self.population_stats["penalty_member_iters"] += 1

            context = self._build_search_context(
                iteration, pool_stagnation, m_sol, m_obj)
            selector_started = time.perf_counter()
            action = self.selector.select_pair(context, self.rng)
            selector_elapsed = time.perf_counter() - selector_started
            if not np.isfinite(selector_elapsed) or selector_elapsed < 0.0:
                raise ValueError(
                    "selector elapsed time must be finite and nonnegative")
            decision_keys = (
                "predicted_mean",
                "uncertainty",
                "selection_score",
                "action_count_before",
            )
            decision_source = None
            if hasattr(self.selector, "get_last_decision"):
                decision_source = self.selector.get_last_decision()
            decision = {
                key: None if decision_source is None else self._json_snapshot(
                    decision_source.get(key)
                )
                for key in decision_keys
            }
            decision["selector_elapsed_seconds"] = float(selector_elapsed)
            decision["factorized"] = (
                None
                if decision_source is None
                else self._json_snapshot(decision_source.get("factorized"))
            )
            if decision_source is not None and "hierarchical" in decision_source:
                decision["hierarchical"] = self._json_snapshot(
                    decision_source["hierarchical"]
                )
            d_idx, r_idx = action
            if not 0 <= d_idx < len(self.destroy_names):
                raise IndexError(
                    f"{type(self.selector).__name__} returned invalid destroy index {d_idx}"
                )
            if not 0 <= r_idx < len(self.repair_names):
                raise IndexError(
                    f"{type(self.selector).__name__} returned invalid repair index {r_idx}"
                )
            d_idx, r_idx = int(d_idx), int(r_idx)
            destroy_name = self.destroy_names[d_idx]
            repair_name = self.repair_names[r_idx]

            destroy_fn = self.destroy_ops[destroy_name]
            repair_fn = self.repair_ops[repair_name]

            # 追踪调用
            self._track_operator_call(destroy_name)
            self._track_operator_call(repair_name)

            # 计算破坏规模 k
            k = self._compute_k(iteration, pool_stagnation)

            # 崩溃追踪 (镜像 base: 无 cx/rebalance/soft 分支)
            destroy_crashed = False
            repair_crashed = False
            operator_started = time.perf_counter()

            # ── 池内重组 (M2, 2026-09-07): population_cx_mode ─────
            # rate 命中且池 ≥2 → 双亲 (m* = 当轮锦标赛选中成员, m' = 池内
            # 随机异键集成员, 12 次尝试) 生成子代替代本迭代 destroy/repair;
            # 子代 → 下方公共可行化链 (与 M1 destroy/repair 后代同链) →
            # SA 接受时 ls_on_accept 精化 → pool_accept_replace 门才替换 m*
            # (与 M1 完全同链)。cx 触发迭代不更新 selector 权重 (legacy
            # cx_used 惯例)。mode none → 逐位走 M1 destroy/repair 路径。
            # ── 改进门+教育 (M3, 2026-09-07): gate="improve" ─────
            # cx 子代先经公共链 + (可选) 池内教育 (破坏-重建+封顶 LS), 再经
            # pool_gate_improve (可行 AND 纯成本严格改进 m*, 不走 SA) 决门:
            #   - 门过: 替换 m*, 本迭代结束 — 不 destroy/repair、不更新 selector
            #     (cx 惯例), 档案/score/停滞/轨迹记账与 M2 cx 路径一致;
            #   - 门不过: cx_gate_fail 累计, 回退执行正常 destroy/repair 迭代
            #     (与 M1 非 cx 迭代一致, selector 更新照常 — cx 尝试变纯增量)。
            # gate="none" → M2 语义逐位不变。
            cx_used = False
            cx_gate_improve_passed = False
            if (self.population_cx_mode != "none"
                    and self.rng.random() < self.population_cx_rate):
                child_cx = self._population_cx_child(pool_sol, pool_keys, selected)
                if child_cx is not None:
                    if self.population_cx_gate == "improve":
                        child_cx = _postprocess(child_cx)
                        if self.population_cx_educate:
                            child_cx = self._educate_pool_child(child_cx)
                        child_obj = calculate_objectives(child_cx)
                        # M4 tw: 罚后接受门替换 M3 的 pool_gate_improve 调用点
                        # (可行性不再进门, 子代可行与否都按罚后成本比)
                        if pen_tw:
                            _gate_pass = pool_penalty_gate(
                                _pen_cost(child_cx, child_obj),
                                pool_pen[selected])
                        else:
                            _gate_pass = pool_gate_improve(
                                child_obj[0], m_obj[0],
                                child_cx.is_feasible())
                        if _gate_pass:
                            self.population_stats["cx_gate_success"] += 1
                            cx_used = True
                            cx_gate_improve_passed = True
                            new_sol = child_cx
                            new_obj = child_obj
                        else:
                            self.population_stats["cx_gate_fail"] += 1
                    else:  # gate = none → M2 语义
                        cx_used = True
                        new_sol = child_cx
            if not cx_used:
                # 应用破坏/修复 (M1 父代源 = 成员 m*; 不调用 _select_parent)
                partial_sol, removed = parent_sol.apply_destroy(
                    destroy_fn, k, self.rng)
                if not isinstance(removed, list):
                    raise TypeError(
                        f"Destroy returned {type(removed).__name__}, expected list")
                new_sol = partial_sol.apply_repair(repair_fn, removed, self.rng)
                if new_sol is None or not hasattr(new_sol, "routes"):
                    raise TypeError(
                        f"Repair returned {type(new_sol).__name__}, expected Solution")

            # 可行化公共链 (M3 逐位 = _postprocess_after_repair; tw 模式 = 同链
            # 跳过 TW 强制的 tw 变体, 见 _postprocess_tw)
            new_sol = _postprocess(new_sol)

            # ── 局部搜索调度 (频率控制, 与 base 完全一致) ────────
            # gate=improve 门过迭代: 子代已 (可选) 教育, 本迭代直接替换 m* —
            # 不再跑周期 LS (do_ls 仍计算供轨迹/决策语义, 不执行)。
            # tw 模式: LS 只在子代可行时执行 (TW 违规子代跳过, 镜像 legacy
            # soft_tw 分支 — 违规解上 RVND 空转)。
            do_ls = False
            if self.local_search_fn is not None and self.ls_freq > 0:
                if iteration % self.ls_freq == 0:
                    do_ls = True
            if do_ls and not cx_gate_improve_passed:
                if (not pen_tw) or new_sol.is_feasible():
                    new_sol = self.local_search_fn(
                        new_sol, max_iterations=self.ls_max_iter, rng=self.rng)
                    new_sol = self._enforce_feasibility(new_sol)

            new_obj = calculate_objectives(new_sol)

            # 接受准则 (裁定 1: 池成员只允许可行解 — 两级接受拆 pool_accept_replace:
            # SA 接受 AND 子代可行同时成立才替换成员。顺序镜像 base: SA 接受后先走
            # ls_on_accept 精化 (base 同构自愈 — 接受子代经 LS+可行化兜底再入池),
            # 精化后候选可行才替换成员; 精化后仍不可行 → 成员不变, 该迭代按"拒绝"
            # 记账 (score=0 / outcome.accepted=False/feasible=False), 停滞照常 +1。
            # 决策点禁内联绕过 — 统一经 pool_accept_replace。软罚分支禁开)
            # gate=improve 门过迭代: 子代已过 pool_gate_improve (纯成本严格改进,
            # 不走 SA), sa_accepted 直接置 True — 不调用 acceptor.accept (无 rng
            # 消耗); 门不过回退迭代与 M1/M2 一致照常走 SA。
            # M4 tw 罚域: destroy/repair 子代也走罚后接受门 (确定性, 不走 SA;
            # 与 M3 改进门同构但比较量 = 罚后成本, 不再要求子代可行) — 门过即
            # 替换 m* (成员可变为不可行), 门不过回退 destroy/repair 语义不变。
            if cx_gate_improve_passed:
                sa_accepted = True
            elif pen_tw:
                sa_accepted = pool_penalty_gate(
                    _pen_cost(new_sol, new_obj), pool_pen[selected])
            else:
                sa_accepted = self.acceptor.accept(m_obj, new_obj, self.rng)
            if sa_accepted:
                self.probe_stats["accepts"] += 1
                if abs(new_obj[0] - parent_obj[0]) < 1e-9:
                    self.probe_stats["equal_accepts"] += 1
            # ── SA 接受时强制 LS 精化 (镜像 base ls_on_accept; soft 分支禁开) ──
            # 只对 SA 接受的子代做; rvnd 内部只收可行改进, 其后 _enforce_feasibility
            # 做容量/车辆/TW 可行化 — base 单解轨迹的自愈即由此路径 (μ=1 退化同构)。
            # tw 模式: LS 只在候选可行时执行 (TW 违规候选跳过, 镜像 legacy soft_tw)。
            candidate_sol = new_sol
            candidate_obj = new_obj
            # gate=improve 门过迭代: 教育已含封顶 LS, 不再走 ls_on_accept 精化
            # (门判据锁定在教育后成本, 二次精化会漂移替换对象)。
            if (sa_accepted and self.local_search_fn is not None
                    and self.ls_on_accept and not do_ls
                    and not cx_gate_improve_passed
                    and ((not pen_tw) or candidate_sol.is_feasible())):
                candidate_sol = self.local_search_fn(
                    candidate_sol, max_iterations=self.ls_max_iter,
                    rng=self.rng)
                candidate_sol = self._enforce_feasibility(candidate_sol)
                candidate_obj = calculate_objectives(candidate_sol)
            if pen_tw:
                # M4: 罚门过即替换 — 成员可变为不可行 (TW 罚域), 不再要求可行
                accepted = bool(sa_accepted)
            else:
                accepted = pool_accept_replace(
                    sa_accepted, candidate_sol.is_feasible())
            if accepted:
                # 子代被接受 → 该成员解/obj 整体替换 (新解副本;
                # member_advances = 子代替换次数, refresh 不计; tw 模式成员
                # 可为 TW 不可行, 罚后成本同步记录)
                pool_sol[selected] = candidate_sol.copy()
                pool_obj[selected] = candidate_obj
                pool_keys[selected] = ParetoArchive._route_keys(
                    {"solution": candidate_sol})
                if pool_pen is not None:
                    pool_pen[selected] = _pen_cost(candidate_sol, candidate_obj)
                self.population_stats["member_advances"][selected] += 1
                member_reset.add(selected)
                new_sol = candidate_sol
                new_obj = candidate_obj

            # 档案更新 (纯净性: 只收可行解; 镜像 base — 拒绝子代也可入档)
            self._sync_depot_assignments(new_sol)
            if new_sol.is_feasible():
                added = self.archive.add(new_obj, {
                    "iteration": iteration,
                    "destroy": destroy_name,
                    "repair": repair_name,
                    "solution": new_sol.copy(),
                    "type": "population",
                })
            else:
                added = False
            last_completed_iteration = iteration
            if added:
                self.probe_stats["adds"] += 1
                cb = self.callbacks.get("on_archive_add")
                if cb is not None:
                    cb(iteration, destroy_name, repair_name, new_obj[0])
            if (self.archive_checkpoint_interval > 0
                    and iteration % self.archive_checkpoint_interval == 0):
                self._record_archive_checkpoint(iteration, run_started)

            # 停滞 (M1: archive 语义 — added → 0, 否则 +1)
            if added:
                pool_stagnation = 0
                if "on_new_archive" in self.callbacks:
                    self.callbacks["on_new_archive"](iteration, self.archive)
            else:
                pool_stagnation += 1
            self.probe_stats["stagnation_peak"] = max(
                self.probe_stats["stagnation_peak"], pool_stagnation)

            # 成员级停滞计数推进 (裁定 2, 2026-09-07): 本迭代被可行子代替换或
            # 被刷新的成员清零, 其余成员 +1 (连续未成功替换迭代数) — 尸体
            # (不可行/冷温饿死成员) 单调累计, 达阈值后被迭代开头扫描驱逐。
            for _mi in range(mu):
                if _mi in member_reset:
                    member_stagnation[_mi] = 0
                else:
                    member_stagnation[_mi] += 1

            # ── score 语义 (与 base 逐位一致; 无 cx → 每迭代都 update) ──
            scheme = self.score_scheme
            if scheme == "activity":
                score = (
                    0.0
                    if (destroy_crashed or repair_crashed)
                    else (2.0 if new_sol.is_feasible() else 0.0)
                )
            elif scheme == "quality":
                if destroy_crashed or repair_crashed:
                    score = 0.0
                elif new_obj[0] < self._best_obj[0] - 1e-9:
                    score = 2.0
                elif new_obj[0] < parent_obj[0] - 1e-9:
                    score = 1.5
                elif accepted:
                    score = 1.0
                else:
                    score = 0.0
            else:  # admission — 默认
                score = 2.0 if added else (1.0 if accepted else 0.0)
            if sa_accepted and not accepted:
                # 裁定 1: SA 接受但子代不可行 → 按拒绝记账 — quality 的改进
                # 阶梯对不可行子代不适用 (池成员只允许可行), 任何 scheme 均记 0。
                score = 0.0
            reward = float(score)
            elapsed = time.perf_counter() - operator_started
            dominates_parent = (
                new_obj[0] <= parent_obj[0]
                and new_obj[1] <= parent_obj[1]
                and new_obj != parent_obj
            )
            outcome = OperatorOutcome(
                accepted=accepted,
                archive_added=added,
                dominates_parent=dominates_parent,
                feasible=new_sol.is_feasible(),
                crashed=destroy_crashed or repair_crashed,
                score=score,
                reward=reward,
                elapsed_seconds=elapsed,
                destroy_crashed=destroy_crashed,
                repair_crashed=repair_crashed,
            )
            self._recent_accepts.append(accepted)
            self._recent_archive_adds.append(added)
            self._recent_improvements.append(max(0.0, reward))
            next_context = self._build_search_context(
                iteration, pool_stagnation, pool_sol[selected],
                pool_obj[selected])
            normalized_reward = (
                0.0
                if outcome.crashed
                else float(np.clip(outcome.score / 2.0, 0.0, 1.0))
            )
            # cx 触发迭代未执行所选算子 → 不更新 selector 权重 (legacy cx_used
            # 惯例); 非 cx 迭代 (含 M1 默认路径) 每迭代都更新。
            if not cx_used:
                self.selector.update(action, outcome, next_context)

            # 降温
            self.acceptor.cool()

            # 轨迹日志 (base 同构字段; decision/action 同 base 结构)
            outcome_payload = {
                "accepted": outcome.accepted,
                "archive_added": outcome.archive_added,
                "feasible": outcome.feasible,
                "crashed": outcome.crashed,
                "discrete_score": outcome.score,
                "normalized_reward": normalized_reward,
                "elapsed_seconds": outcome.elapsed_seconds,
                "cost": pool_obj[selected][0],
                "emission": pool_obj[selected][1],
                "archive_size": self.archive.size,
            }
            # M4 tw 罚域诊断: outcome.cost 记真实成本, 另记罚后成本与 TW excess
            if pen_tw:
                outcome_payload["penalized_cost"] = float(pool_pen[selected])
                outcome_payload["tw_excess"] = float(
                    self._tw_excess(pool_sol[selected]))
            self.trajectory.append({
                "iteration": iteration,
                "context": asdict(context),
                "decision": decision,
                "action": {
                    "destroy_index": d_idx,
                    "repair_index": r_idx,
                    "destroy_name": destroy_name,
                    "repair_name": repair_name,
                    "flat_action_index": d_idx * len(self.repair_names) + r_idx,
                    "cold_start": (
                        None
                        if decision["action_count_before"] is None
                        else decision["action_count_before"] == 0
                    ),
                },
                "outcome": outcome_payload,
            })

            # 迭代回调
            if "on_iteration" in self.callbacks:
                self.callbacks["on_iteration"](
                    iteration, pool_obj[selected], self.archive)

        self.population_stats["dc_min_mean"] = (
            dc_sum / n_iters if n_iters > 0 else 0.0)

        # ── 收尾: 终局 soft 修复块不执行 (soft 禁开); final LS 镜像 base ──
        if (self.local_search_fn is not None and self.ls_final
                and not self.archive.is_empty()):
            if last_completed_iteration > 0:
                checkpoint = next(
                    (
                        item
                        for item in reversed(self.archive_checkpoints)
                        if item["iteration"] == last_completed_iteration
                    ),
                    None,
                )
                if checkpoint is None:
                    checkpoint = self._archive_checkpoint_snapshot(
                        last_completed_iteration, run_started
                    )
                self.pre_final_local_search_checkpoint = self._json_snapshot(
                    checkpoint
                )
            self._apply_final_local_search()

        if (self.archive_checkpoint_interval > 0
                and last_completed_iteration > 0):
            self._record_archive_checkpoint(
                last_completed_iteration,
                run_started,
                replace_same_iteration=True,
            )

        # 探针暴露: 循环结束时最后被选成员的解/obj (base 语义 = 轨迹末端解)
        self.last_solution = pool_sol[last_selected].copy()
        self.last_obj = tuple(pool_obj[last_selected])
        self.population_final = [
            (pool_obj[i], pool_sol[i].copy()) for i in range(mu)
        ]
        if pen_tw:
            self.population_stats["penalty_infeasible_final"] = sum(
                1 for i in range(mu) if not pool_sol[i].is_feasible())
        return self.archive

    def _population_cx_child(self, pool_sol: list, pool_keys: list,
                             selected: int) -> "Solution | None":
        """池内重组生成子代 (M2, 2026-09-07): m* = 锦标赛选中成员 (selected),
        m' = 池内随机另一成员 (路由键集与 m* 不同, 12 次尝试, 镜像 legacy
        _crossover_child 1942-1951 语义; 键集用 ParetoArchive._route_keys 约定)。

        返回 None = 找不到异键集 m' (双亲不齐 → 调用方退回 destroy/repair)。
        route 臂 = route_copy_crossover (A 臂, 路由级继承); partition 臂 =
        partition_crossover (B 臂, 车场簇继承)。触发且双亲齐备 → cx_attempts
        += 1; 子代生成 → cx_children += 1; partition 臂修复池客户 (跨父本
        重复) 累计进 cx_duplicates (route 臂恒 0); 子代路由键 − 池全体成员
        路由键并集 的新键累计进 cx_novel_route_keys (镜像 legacy novel 语义)。
        全部随机走 engine.rng; 不更新 selector 权重 (由调用方 cx_used 决定)。
        """
        mu = len(pool_keys)
        if mu < 2:
            return None
        keys_a = pool_keys[selected]
        second = None
        for _ in range(12):
            j = int(self.rng.integers(0, mu))
            if j == selected:
                continue
            if pool_keys[j] == keys_a:
                continue
            second = j
            break
        if second is None:
            return None
        self.population_stats["cx_attempts"] += 1
        sol_a = pool_sol[selected]
        sol_b = pool_sol[second]
        if self.population_cx_mode == "route":
            from src.operators.crossover import route_copy_crossover
            child = route_copy_crossover(
                sol_a, sol_b, self.rng, self.population_cx_inherit_prob)
        else:  # partition
            from src.operators.crossover import partition_crossover
            child = partition_crossover(sol_a, sol_b, self.rng)
            self.population_stats["cx_duplicates"] += int(
                getattr(child, "_cx_duplicates", 0))
        self.population_stats["cx_children"] += 1
        # 池内新路由键计数 (镜像 legacy 1963-1968 novel 语义; 池成员键集 =
        # pool_keys 快照 — 本迭代尚未因替换更新, 与 selector 快照同域)
        child_keys = ParetoArchive._route_keys({"solution": child})
        if child_keys:
            pool_union: set = set()
            for k in pool_keys:
                pool_union |= k
            self.population_stats["cx_novel_route_keys"] += len(child_keys - pool_union)
        return child

    def _educate_pool_child(self, child: "Solution") -> "Solution":
        """池内破坏性教育 (M3, 2026-09-07): 镜像 legacy _educate_child 结构
        (2026 旧行区) 但用 population_educate_* 键 + 封顶 LS 预算 (默认 50,
        legacy 300 过重 — 池内按次封顶)。

        只作用于 gate=improve 的 cx 子代 (destroy/repair 产物不教育): 随机移除
        population_educate_burst_ratio×n 客户 (random_removal 语义) → 既有 repair
        重插 (TW 实例 greedy_cost_tw_insertion, 否则 greedy_cost_insertion) →
        短 LS (预算 population_educate_ls_budget) → _enforce_feasibility 兜底。
        教育后解可行且成本 < 教育前才返回教育解 (educate_success/delta_sum 累计);
        否则退回原 child — 教育不劣化 cx 子代, 后续改进门仍按退回 child 决门。
        不更新 selector; 全部随机 engine.rng。
        """
        from src.operators.destroy import random_removal

        inst = child.instance
        n = inst.num_customers
        k = max(1, int(round(self.population_educate_burst_ratio * n)))
        child_cost = calculate_objectives(child)[0]

        candidate = child
        partial, removed = candidate.apply_destroy(random_removal, k, self.rng)
        if removed:
            if inst.has_time_windows:
                from src.operators.repair import greedy_cost_tw_insertion as repair_fn
            else:
                from src.operators.repair import greedy_cost_insertion as repair_fn
            candidate = partial.apply_repair(repair_fn, removed, self.rng)
            candidate = self._postprocess_after_repair(candidate)

        # 教育 LS: 破坏-重建后再抛光 (预算封顶, 现 ls_on_accept 短 LS = ls_max_iter)
        if self.local_search_fn is not None and candidate.is_feasible():
            candidate = self.local_search_fn(
                candidate, max_iterations=self.population_educate_ls_budget,
                rng=self.rng)
        candidate = self._enforce_feasibility(candidate)

        educ_cost = calculate_objectives(candidate)[0]
        self.population_stats["educate_attempts"] += 1
        if candidate.is_feasible() and educ_cost < child_cost - 1e-9:
            self.population_stats["educate_success"] += 1
            self.population_stats["educate_delta_sum"] += float(
                child_cost - educ_cost)
            return candidate
        return child

    def _build_pool_member(self) -> tuple:
        """建一个池成员 (裁定 2 重试兜底): build → _enforce_feasibility, 仍不可行
        则重 build (engine.rng 链取源), 至多 10 次; 10 次全败取成本最低候选入池。

        常数 10 写死 (不加 config 键): pr11 类残余 TW 违规实例不得阻塞运行 —
        该成员仍可被锦标赛选中演化, 其可行子代一旦出现即替换它 (与 base 单解
        自愈路径同构)。
        """
        best: tuple | None = None  # (sol, obj) — 10 次全败时的最低成本候选
        for _attempt in range(10):
            sol = build_initial_solution(self.instance, self.rng, self.init_method)
            sol = self._enforce_feasibility(sol)
            obj = calculate_objectives(sol)
            if sol.is_feasible():
                return sol, obj
            if best is None or (obj[0] + obj[1]) < (best[1][0] + best[1][1]):
                best = (sol, obj)
        return best

    def _member_refresh_replacement(self) -> tuple:
        """成员级刷新用新解 (裁定 2, 2026-09-07): 档案精英副本优先, 档案空兜底。

        _select_parent (既有 crowding 选择, 无 rng 消耗) 得档案条目 → 解副本 →
        obj 重算 (calculate_objectives); 档案空 (None) → _build_pool_member
        (裁定 1 的 10 次重试逻辑) 兜底。返回 (sol, obj); 调用方负责替换成员、
        更新路由键缓存与成员级停滞计数。
        """
        entry = self._select_parent()
        if entry is not None:
            sol = self._reconstruct_solution(entry).copy()
            return sol, calculate_objectives(sol)
        return self._build_pool_member()

    # ── timefix (2026-09-12): 时间预算防饥饿引导 ─────────────────────

    def _timefix_should_fire(self, iteration: int, start_time: float) -> bool:
        """timefix 引导触发判定 (仅时间预算模式)。

        - 迭代模式 (max_time ≥ 1e8) 恒 False → 已批 6000iter 结果逐位不变;
        - 罚域 (M4)/软约束模式不介入 (各自流程自有语义);
        - 触发条件: 档案空 + 已耗预算 ≥ 30% + 距上次尝试 ≥ 100 迭代 +
          总尝试 < 5 次 (间隔与上限防连续触发)。
        """
        if self.max_time >= _TIMEFIX_TIME_MODE_CEIL:
            return False
        if self.population_penalty_mode != "none":
            return False
        if self.soft_capacity or self.soft_tw:
            return False
        if not self.archive.is_empty():
            return False
        if self._timefix_attempts >= _TIMEFIX_MAX_ATTEMPTS:
            return False
        if iteration - self._timefix_last_iter < _TIMEFIX_MIN_GAP_ITERS:
            return False
        return ((time.perf_counter() - start_time)
                >= _TIMEFIX_ELAPSED_FRACTION * self.max_time)

    def _timefix_climb(self, sol: "Solution") -> "Solution":
        """引导用重预算联合超额爬坡 (参数依据: pr20 s3 全不可行池实测
        wb=1M/step_cap=20k 在 2.3s 内收敛; 联合口径同 _enforce_duration)。"""
        from src.operators.repair import combined_excess_repair
        return combined_excess_repair(
            sol.copy(), self.rng, max_rounds=100, kick_size=3,
            max_iterations=60, include_tw=True,
            work_budget=1_000_000, step_cap=20_000)

    def _timefix_feasibility_bootstrap(
            self, pool_sol: list, pool_obj: list, pool_keys: list
    ) -> int | None:
        """对池内最优成员做重预算爬坡; 成功 → 可行替补换掉最劣成员并播种
        档案 (镜像 §初始播种), 返回被替换下标; 失败 → None (池/档案不动)。"""
        best_i = min(
            range(len(pool_obj)),
            key=lambda i: (float(pool_obj[i][0]) + float(pool_obj[i][1]), i))
        cand = self._timefix_climb(pool_sol[best_i])
        if cand is None or not cand.is_feasible():
            return None
        worst_i = max(
            range(len(pool_obj)),
            key=lambda i: (float(pool_obj[i][0]) + float(pool_obj[i][1]), -i))
        pool_sol[worst_i] = cand
        pool_obj[worst_i] = calculate_objectives(cand)
        pool_keys[worst_i] = ParetoArchive._route_keys({"solution": cand})
        self._sync_depot_assignments(cand)
        self.archive.add(pool_obj[worst_i], {
            "iteration": self.iteration,
            "type": "timefix_bootstrap",
            "solution": cand.copy(),
        })
        return worst_i

    def _build_search_context(
        self,
        iteration: int,
        stagnation_count: int,
        solution: Solution,
        current_objective: Objective,
    ) -> SearchContext:
        """从当前搜索过程构造选择器可见的归一化上下文。"""
        temperature = float(getattr(self.acceptor, "T", 0.0))
        initial_temperature = max(float(getattr(self.acceptor, "T0", 1.0)), 1e-12)
        if self._initial_objective is None:
            raise RuntimeError("initial objective is unavailable before run initialization")
        # 容量利用率（原 ContextV1Extractor 公式内联）
        capacity = max(float(solution.instance.max_vehicle_capacity), 1e-12)
        loads = [
            float(np.clip(solution.route_load_fast(route) / capacity, 0.0, 1.0))
            for routes in solution.routes.values()
            for route in routes
        ]
        archive_size = int(self.archive.size)
        archive_capacity = int(getattr(self.archive, "capacity", 1)) or 1
        return SearchContext(
            iteration=iteration,
            progress=float(np.clip(iteration / max(self.max_iterations, 1), 0.0, 1.0)),
            stagnation_ratio=float(
                np.clip(stagnation_count / max(self.stagnation_limit, 1), 0.0, 1.0)
            ),
            temperature_ratio=float(
                np.clip(temperature / initial_temperature, 0.0, 1.0)
            ),
            recent_acceptance_rate=self._mean_history(self._recent_accepts),
            recent_archive_add_rate=self._mean_history(self._recent_archive_adds),
            archive_size_ratio=min(archive_size / archive_capacity, 1.0),
            capacity_utilization_mean=float(np.mean(loads)) if loads else 0.0,
            capacity_utilization_std=float(np.std(loads)) if loads else 0.0,
            recent_improvement=self._mean_history(self._recent_improvements),
        )

    @classmethod
    def _json_snapshot(cls, value):
        """Recursively detach selector diagnostics into JSON-safe Python values."""
        if isinstance(value, np.ndarray):
            return cls._json_snapshot(value.tolist())
        if isinstance(value, np.generic):
            return cls._json_snapshot(value.item())
        if isinstance(value, Mapping):
            return {
                str(key): cls._json_snapshot(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._json_snapshot(item) for item in value]
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not np.isfinite(value):
                raise ValueError("selector diagnostics must contain finite values")
            return value
        raise TypeError(
            f"selector diagnostics contain non-JSON value {type(value).__name__}"
        )

    def _record_archive_checkpoint(
        self,
        iteration: int,
        run_started: float,
        *,
        replace_same_iteration: bool = False,
    ) -> None:
        checkpoint = self._archive_checkpoint_snapshot(iteration, run_started)
        if (
            replace_same_iteration
            and self.archive_checkpoints
            and self.archive_checkpoints[-1]["iteration"] == iteration
        ):
            self.archive_checkpoints[-1] = checkpoint
        else:
            self.archive_checkpoints.append(checkpoint)

    def _archive_checkpoint_snapshot(
        self,
        iteration: int,
        run_started: float,
    ) -> dict:
        elapsed = float(time.perf_counter() - run_started)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("checkpoint elapsed time must be finite and nonnegative")
        return {
            "iteration": int(iteration),
            "elapsed_seconds": elapsed,
            "objectives": [
                [float(cost), float(emission)]
                for cost, emission in self.archive.objectives
            ],
        }

    @staticmethod
    def _mean_history(values) -> float:
        if not values:
            return 0.0
        return min(max(float(np.mean(values)), 0.0), 1.0)

    def _apply_final_local_search(self):
        """对档案中所有解运行 2-opt 最终精化。

        TW 实例跳过: two_opt 是纯距离重排, 会破坏 TW 可行解结构,
        _enforce_tw 修复后成本反而恶化 (实测 1189.80 → 1766.67)。
        """
        if self.instance.has_time_windows:
            return
        refined = []
        final_iter = max(self.ls_max_iter, 50)
        for obj, meta in self.archive.entries:
            sol = meta.get("solution")
            if sol is not None:
                refined_sol = self.local_search_fn(sol, max_iterations=final_iter,
                                                   rng=self.rng)
                # final LS 后可行性兜底 (容量 + TW) + 车场分配同步
                refined_sol = self._enforce_feasibility(refined_sol)
                refined_obj = calculate_objectives(refined_sol)
                refined.append((refined_obj, {**meta, "solution": refined_sol.copy()}))
            else:
                refined.append((obj, meta))

        self.archive.clear()
        for obj, meta in refined:
            self.archive.add(obj, meta)

    def _select_parent(self):
        """从档案中选择父代解。"""
        if self.archive.is_empty():
            return None
        return self.parent_selector(self.archive.entries, self.rng)

    def _reconstruct_solution(self, entry: tuple[Objective, dict]) -> Solution:
        """
        从档案条目重建解。
        如果条目不包含完整解，返回最佳近似。
        """
        # 当前如果没有存储解，回退到初始解
        # 实践中求解器应将解存储在 info 中
        info = entry[1]
        if "solution" in info:
            return info["solution"].copy()
        return build_initial_solution(self.instance, self.rng, self.init_method)

    def _compute_k(self, iteration: int, stagnation: int) -> int:
        """计算动态破坏规模。"""
        n = self.instance.num_customers
        k_min = max(4, int(self.k_min_ratio * n))
        k_max = max(8, int(self.k_max_ratio * n))

        # 如果停滞，增大破坏规模 (深破坏: 触发阈值与上限可配置;
        # 默认 0.5×limit=150 — 注意 recovery(limit//3=100)先清零,
        # 默认配置下该分支是死代码, 逃逸实验 V5 提前触发检验)
        if stagnation > self.stagnation_limit * self.stagnation_deep_ratio:
            k_min = max(4, int(0.3 * n))
            k_max = max(8, int(self.deep_destroy_max_ratio * n))

        return int(self.rng.integers(k_min, k_max + 1))

    # ── 软容量辅助 (探针 2, 2026-09-04) ──────────────────────
    def _cap_excess(self, solution: "Solution") -> float:
        """容量超载总量 Σ max(0, load − cap) (需求单位)。"""
        inst = solution.instance
        cap = inst.max_vehicle_capacity
        nd = inst.num_depots
        total = 0.0
        for rl in solution.routes.values():
            for route in rl:
                load = sum(inst.customers[i - nd].demand for i in route)
                if load > cap:
                    total += load - cap
        return float(total)

    def _soft_penalize(self, solution: "Solution", obj: Objective) -> Objective:
        """把真实目标转罚后目标 (软容量模式; 硬模式原样返回)。"""
        if not self.soft_capacity:
            return obj
        exc = self._cap_excess(solution)
        if exc <= 1e-9:
            return obj
        return (float(obj[0] + self._soft_lam * exc), float(obj[1]))

    def _soft_seed(self, sol: "Solution", iteration: int) -> bool:
        """把当前解强制修复回可行并入档 (软容量周期播种)。

        成功入档时经 self._soft_seed_candidate/_soft_seed_obj 暴露修复解
        (调用方决定是否采纳为轨迹锚点); 失败 → λ ×1.5 上调 (拉回可行域),
        成功 → λ ÷1.5 回落 (下限 λ0)。返回是否入档 (精英池满且不更优时为
        False — 此时仍记录修复解供锚定)。
        """
        self.soft_stats["seed_attempts"] += 1
        candidate = sol.copy()
        self._sync_depot_assignments(candidate)
        candidate = self._soft_repair_hard(candidate)  # 硬语义多轮可行修复
        self._sync_depot_assignments(sol)  # 共享 instance 的 assigned 状态还原
        self._soft_seed_candidate = None
        self._soft_seed_obj = None
        if not candidate.is_feasible():
            self._adapt_soft_lam(seeded_ok=False)
            return False
        obj = calculate_objectives(candidate)
        # 种子 LS 精化: 修复解先用 RVND 抛光再入档/锚定 (参数扫描: 种子无 LS
        # 时 soft 臂落后 hard ~9%, 有 LS 时收窄到 ~3%)
        if self.local_search_fn is not None:
            polished = self.local_search_fn(candidate, max_iterations=self.ls_max_iter,
                                            rng=self.rng)
            if polished.is_feasible():
                candidate = polished
                obj = calculate_objectives(candidate)
        self._soft_seed_candidate = candidate
        self._soft_seed_obj = obj
        added = self.archive.add(obj, {
            "iteration": iteration,
            "type": "soft_seed",
            "solution": candidate.copy(),
        })
        if added:
            self.soft_stats["seed_adds"] += 1
            self._adapt_soft_lam(seeded_ok=True)
        return added

    def _soft_repair_hard(self, solution: "Solution", max_rounds: int = 12) -> "Solution":
        """硬语义多轮可行修复 (软容量播种/终局用)。

        深超载 (数百单位) 下单遍贪心强制会顺序依赖失败 (v1 实测种子全败);
        每轮: 抽空所有超载路由 → 乱序 → greedy_cost_tw (硬容量) 重插;
        至多 max_rounds 轮。期间临时关闭实例 soft 标志 (repair 读它),
        finally 恢复 — 调用方解与共享 instance 的其他解不受污染。
        """
        from src.operators.repair import greedy_cost_tw_insertion

        inst = solution.instance
        nd = inst.num_depots
        cap = inst.max_vehicle_capacity
        saved_soft = getattr(inst, "soft_capacity", False)
        setattr(inst, "soft_capacity", False)
        try:
            sol = solution.copy()
            for _ in range(max_rounds):
                self._sync_depot_assignments(sol)
                sol = self._enforce_feasibility(sol)
                if sol.is_feasible():
                    return sol
                pool: list[int] = []
                for di in list(sol.routes.keys()):
                    kept, dropped = [], []
                    for route in sol.routes[di]:
                        load = sum(inst.customers[i - nd].demand for i in route)
                        if load > cap + 1e-9:
                            dropped.append(route)
                        else:
                            kept.append(route)
                    sol.routes[di] = kept
                    for route in dropped:
                        pool.extend(route)
                seen = sorted(set(pool))  # 去重保覆盖 (曾重复抽取)
                self.rng.shuffle(seen)
                sol = greedy_cost_tw_insertion(sol, seen, self.rng)
            return sol
        finally:
            setattr(inst, "soft_capacity", saved_soft)

    def _set_soft_lam(self, value: float):
        """更新 λ 并同步到实例 (repair 插入成本读取)。"""
        self._soft_lam = float(value)
        setattr(self.instance, "soft_capacity_lambda", self._soft_lam)

    def _adapt_soft_lam(self, seeded_ok: bool):
        if not self.soft_capacity:
            return
        if seeded_ok:
            self._set_soft_lam(max(self._soft_lam0, self._soft_lam / 1.5))
        else:
            self._set_soft_lam(min(self._soft_lam_max, self._soft_lam * 1.5))

    # ── 软 TW 辅助 (探针, 2026-09-06) ───────────────────────
    def _tw_excess(self, solution: "Solution") -> float:
        """TW 违规总量 Σ max(0, arrival − due) (时间单位)。"""
        return float(sum(v["excess"] for v in solution.tw_violations()))

    def _soft_tw_penalize(self, solution: "Solution",
                          obj: Objective) -> Objective:
        """把真实目标转罚后目标 (软 TW 模式; 硬模式原样返回)。

        罚 = λ × Σ(v["excess"] for v in tw_violations()); 每罚一次记一个违规
        迭代 (violation_iters) — 机制生效判据的观测计数。
        """
        if not self.soft_tw:
            return obj
        exc = self._tw_excess(solution)
        if exc <= 1e-9:
            return obj
        self.soft_tw_stats["violation_iters"] += 1
        return (float(obj[0] + self._soft_tw_lam * exc), float(obj[1]))

    def _soft_tw_seed(self, sol: "Solution", iteration: int) -> bool:
        """把当前解强制 _enforce_tw 修复回可行并入档 (软 TW 周期播种)。

        修复 = _enforce_tw (含违规客户的路由整条移除 → TW-aware 贪心重插)。
        成功入档时经 self._soft_tw_seed_candidate/_soft_tw_seed_obj 暴露修复解
        (调用方决定是否采纳为轨迹锚点); 修复不可行 → λ ×1.5 上调 (拉回可行
        域), 入档成功 → λ ÷1.5 回落 (下限 λ0)。返回是否入档 (精英池满且不
        更优时为 False — 此时仍记录修复解供锚定)。
        """
        self.soft_tw_stats["repair_events"] += 1
        candidate = sol.copy()
        self._sync_depot_assignments(candidate)
        candidate = self._enforce_tw(candidate)
        # 播种需完全可行: 强制联合口径 (TW+duration, 绕过软 TW 的 dur-only 口径),
        # 低频路径给足预算 (同 _enforce_feasibility 的初始可行化)
        candidate = self._enforce_duration(candidate, full=True, max_rounds=60,
                                           work_budget=200000, step_cap=6000)
        self._sync_depot_assignments(sol)  # 共享 instance 的 assigned 状态还原
        self._soft_tw_seed_candidate = None
        self._soft_tw_seed_obj = None
        if not candidate.is_feasible():
            self._adapt_soft_tw_lam(seeded_ok=False)
            return False
        obj = calculate_objectives(candidate)
        self.soft_tw_stats["repair_feasible"] += 1
        self._soft_tw_seed_candidate = candidate
        self._soft_tw_seed_obj = obj
        added = self.archive.add(obj, {
            "iteration": iteration,
            "type": "soft_tw_seed",
            "solution": candidate.copy(),
        })
        if added:
            self._adapt_soft_tw_lam(seeded_ok=True)
        return added

    def _set_soft_tw_lam(self, value: float):
        """更新软 TW 罚系数 λ 并同步观测计数。"""
        self._soft_tw_lam = float(value)
        self.soft_tw_stats["lam"] = float(value)

    def _adapt_soft_tw_lam(self, seeded_ok: bool):
        if not self.soft_tw:
            return
        if seeded_ok:
            self._set_soft_tw_lam(max(self._soft_tw_lam0, self._soft_tw_lam / 1.5))
        else:
            self._set_soft_tw_lam(min(self._soft_tw_lam_max, self._soft_tw_lam * 1.5))

    def _postprocess_after_repair(self, new_sol: "Solution") -> "Solution":
        """repair/交叉后代之后的可行化公共链 (原主循环内联逻辑, 提取共用)。

        顺序 (逐位等价): 未分配补插 → 容量/车辆硬约束 (软容量模式只保车辆数,
        容量交由罚函数) → TW 硬约束。
        """
        # 确保可行性：重新插入未分配的客户
        unassigned = new_sol.unassigned_customers()
        if unassigned:
            new_sol = self._repair_unassigned(new_sol, unassigned)
        if self.soft_capacity:
            # 软容量模式: 允许超载穿越 — 只保车辆数硬约束 (TW 见下),
            # 容量违规由罚函数 (λ×excess) 经接受准则引导, 不在此强制
            if any(
                len(rl) > new_sol.instance.depots[di].vehicles_available
                for di, rl in new_sol.routes.items()
            ):
                new_sol = self._enforce_vehicle_limit(new_sol)
        else:
            if not new_sol.is_feasible():
                new_sol = self._enforce_capacity(new_sol)
                # 车辆数违规只在 _enforce_feasibility (LS 后) 处理 —
                # 非 LS 迭代 repair 兜底曾超限开新车, 违规解漏进档案
                # (实测 pr11: 每车场 1 辆上限, 最终解 depot3 2 条路由)
                if not new_sol.is_feasible():
                    new_sol = self._enforce_vehicle_limit(new_sol)

        # ── 时间窗硬约束：局部搜索前强制 TW 可行 ──────────
        # (修复算子可能是纯距离的, 或 destroy 破坏了 TW;
        #  违规解直接丢弃, 避免进入搜索 — 软 TW 模式除外: TW 约束转罚函数,
        #  违规解需流到接受准则被 λ×Σexcess 评价, 此处不强制)
        if (new_sol.instance.has_time_windows and new_sol.tw_violations()
                and not self.soft_tw):
            new_sol = self._enforce_tw(new_sol)
        # ── 最大路线时长硬约束 (软 TW 也不松弛 duration) ──────────
        # 候选路径用小预算 (每迭代 1-2 次调用, 大预算拖垮墙钟); 初始/播种
        # 等低频路径用默认全预算 (见 _enforce_feasibility / _soft_tw_seed)
        new_sol = self._enforce_duration(new_sol, work_budget=15000,
                                         step_cap=1500)
        return new_sol

    # ── 车场分配重平衡 (方向①, 2026-09-06) ──────────────────
    def _rebalance_removal_on(self, iteration: int) -> bool:
        """removal/both 模式的重平衡迭代判定 (interval 命中)。"""
        return (self.rebalance_mode in ("removal", "both")
                and self.rebalance_interval > 0
                and iteration % self.rebalance_interval == 0)

    def _rebalance_redispatch_on(self, iteration: int) -> bool:
        """redispatch/both 模式的重平衡迭代判定 (interval 命中)。"""
        return (self.rebalance_mode in ("redispatch", "both")
                and self.rebalance_interval > 0
                and iteration % self.rebalance_interval == 0)

    def _redispatch_pass(self, solution: "Solution") -> "Solution":
        """错配修复 pass: 对错配客户按错配度降序, 逐个尝试移动到最近车场的
        TW/容量可行位置, 仅接受成本严格下降的移动。

        独立后处理 (不进 selector 统计); 保覆盖/容量/车辆数/TW 可行 —
        只做"移除后插入仍可行且总距离下降"的移动。无移动时返回等价副本。
        """
        from src.operators.repair import (
            _insert_incremental_excess,
            _insertion_delta,
            _route_timeline,
        )
        from src.core.objectives import removal_cost_delta

        sol = solution.copy()
        inst = sol.instance
        dm = inst.distance_matrix
        nd = inst.num_depots
        cap = inst.max_vehicle_capacity
        tw = inst.has_time_windows
        R = inst.reach_matrix if tw else None
        unit = inst.emission_model.unit_cost_per_km
        self._sync_depot_assignments(sol)

        scored = []
        for di, routes in sol.routes.items():
            for route in routes:
                for ci in route:
                    nearest = min(range(nd), key=lambda d: dm[d][ci])
                    mismatch = dm[di][ci] - dm[nearest][ci]
                    if mismatch > 1e-9:
                        scored.append((ci, nearest, mismatch))
        scored.sort(key=lambda t: -t[2])
        self.rebalance_stats["redispatch_passes"] += 1
        if not scored:
            return sol

        def _locate(cidx):
            for di, routes in sol.routes.items():
                for ri, route in enumerate(routes):
                    if cidx in route:
                        return di, ri, route.index(cidx)
            return None

        def _remove_customer_from(cidx, di, ri, pos):
            route = sol.routes[di][ri]
            del route[pos]
            if not route:
                del sol.routes[di][ri]

        for ci, nearest, _m in scored:
            cur = _locate(ci)
            if cur is None:
                continue
            di, ri, pos = cur
            if di == nearest:
                continue  # 已回最近车场 (前序移动所致)
            # 候选: 最近车场现有路由 + 新车 (TW/容量可行位)
            best = None  # (add_delta, tri, p)
            for tri, trotte in enumerate(sol.routes.get(nearest, [])):
                load = sum(inst.customers[x - nd].demand for x in trotte)
                if (load + inst.customers[ci - nd].demand > cap
                        and not getattr(inst, "soft_capacity", False)):
                    continue
                if tw:
                    arr, done, cum_exc = _route_timeline(
                        inst, nearest, trotte, tri)
                for p in range(len(trotte) + 1):
                    if R is not None:
                        prev = trotte[p - 1] if p > 0 else None
                        nxt = trotte[p] if p < len(trotte) else None
                        if ((prev is not None and not R[prev - nd][ci - nd])
                                or (nxt is not None and not R[ci - nd][nxt - nd])):
                            continue
                    if tw and _insert_incremental_excess(
                            inst, nearest, trotte, tri, p, ci, arr, done,
                            cum_exc) > 1e-9:
                        continue
                    d = _insertion_delta(inst, trotte, p, ci, nearest)
                    if best is None or d < best[0]:
                        best = (d, tri, p)
            if (len(sol.routes.get(nearest, []))
                    < inst.depots[nearest].vehicles_available):
                nd_add = dm[nearest][ci] + dm[ci][nearest]
                if best is None or nd_add < best[0]:
                    best = (nd_add, -1, None)
            if best is None:
                continue
            add_delta, tri, p = best
            # 仅接受成本严格下降: 插入增量 − 移除节省 < 0
            saving_cost = removal_cost_delta(sol, ci)[0]
            if add_delta * unit - saving_cost >= -1e-9:
                continue
            _remove_customer_from(ci, di, ri, pos)
            if tri < 0:
                sol.routes[nearest].append([ci])
            else:
                sol.routes[nearest][tri].insert(p, ci)
            inst.customers[ci - nd].assigned_depot_index = nearest
            self.rebalance_stats["redispatch_moves"] += 1

        self._sync_depot_assignments(sol)
        return sol

    # ── 破坏性教育 (方向②, 2026-09-06) ──────────────────────
    def _educate_child(self, child: "Solution") -> "Solution":
        """对 cx 后代做破坏性教育: 随机移除 educate_burst_ratio×n 客户 → 现有
        repair 重插 → postprocess + 短 LS (educate_ls_budget)。

        教育后解可行且成本 < 原后代才返回教育解 (cx_stats 记 educate_improved /
        educate_delta); 否则退回 child (教育不劣化当前迭代轨迹)。教育不更新
        selector 权重 (cx 分支惯例)。破坏语义 = random_removal: 改变路由成员
        组合, 而非纯 LS 精化 (纯 LS 已知无法迁移划分盆地)。
        """
        from src.operators.destroy import random_removal

        inst = child.instance
        n = inst.num_customers
        k = max(1, int(round(self.educate_burst_ratio * n)))
        child_cost = calculate_objectives(child)[0]

        candidate = child
        for _ in range(max(1, self.educate_burst_max_iter)):
            partial, removed = candidate.apply_destroy(random_removal, k, self.rng)
            if not removed:
                break
            if inst.has_time_windows:
                from src.operators.repair import greedy_cost_tw_insertion as repair_fn
            else:
                from src.operators.repair import greedy_cost_insertion as repair_fn
            candidate = partial.apply_repair(repair_fn, removed, self.rng)
            candidate = self._postprocess_after_repair(candidate)

        # 教育 LS: 破坏-重建后再抛光 (只接受可行改进; 预算 educate_ls_budget,
        # 现 ls_on_accept 短 LS 预算 = ls_max_iter 10-30)
        if self.local_search_fn is not None and candidate.is_feasible():
            polished = self.local_search_fn(
                candidate, max_iterations=self.educate_ls_budget, rng=self.rng)
            candidate = polished
        candidate = self._enforce_feasibility(candidate)

        educ_cost = calculate_objectives(candidate)[0]
        self.cx_stats["educate_attempts"] += 1
        if candidate.is_feasible() and educ_cost < child_cost - 1e-9:
            self.cx_stats["educate_improved"] = True
            self.cx_stats["educate_success"] += 1
            self.cx_stats["educate_delta"] = float(child_cost - educ_cost)
            # 教育产出的档案池外新路由键补计 (child_cx 键已在 _crossover_child
            # 计过; 教育新增结构在此补计 — 划分变化证据)
            self._count_novel_keys(candidate)
            return candidate
        self.cx_stats["educate_improved"] = False
        self.cx_stats["educate_delta"] = 0.0
        return child

    def _count_novel_keys(self, sol: "Solution") -> None:
        """把 sol 中档案池没有的路由键数补进 cx_stats.novel_route_keys。"""
        keys = ParetoArchive._route_keys({"solution": sol})
        if not keys:
            return
        pool_keys = set()
        for e in self.archive.entries:
            pool_keys |= ParetoArchive._route_keys(e[1])
        self.cx_stats["novel_route_keys"] += len(keys - pool_keys)

    def _crossover_child(self) -> "Solution | None":
        """种群交叉: 选双亲 (A=crowding, B=随机异质路由集) → 路线复制后代。

        返回 None = 无合适双亲 (档案 <2 或找不到路由集不同的 B) — 调用方
        退回常规 destroy/repair 迭代。
        """
        if self.archive.size < 2:
            return None
        entry_a = self._select_parent()
        if entry_a is None:
            return None
        keys_a = ParetoArchive._route_keys(entry_a[1])
        cost_a = entry_a[0][0]
        sol_b = None
        if self.crossover_pair == "diverse":
            # 互补双亲 (探针 P, 2026-09-05): 与 A 路由 Jaccard 距离最大且成本
            # ≤ A×(1+band) 的档案成员 — 09-04 cx 失效归因 = 双亲全客户覆盖
            # 冲突, 互补双亲应产出结构性新后代 (离线教育列实验的选择在线化)。
            best_d, best_entry = -1.0, None
            for entry_b in self.archive.entries:
                if entry_b is entry_a:
                    continue
                if entry_b[0][0] > cost_a * (1.0 + self.crossover_cost_band):
                    continue
                d = ParetoArchive._jaccard_dist(
                    keys_a, ParetoArchive._route_keys(entry_b[1]))
                if d > best_d:
                    best_d, best_entry = d, entry_b
            if best_entry is not None:
                sol_b = self._reconstruct_solution(best_entry)
        else:
            for _ in range(12):
                entry_b = self.archive.entries[
                    int(self.rng.integers(0, self.archive.size))
                ]
                if entry_b is entry_a:
                    continue
                if ParetoArchive._route_keys(entry_b[1]) == keys_a:
                    continue
                sol_b = self._reconstruct_solution(entry_b)
                break
        if sol_b is None:
            return None
        from src.operators.crossover import route_copy_crossover

        sol_a = self._reconstruct_solution(entry_a)
        child = route_copy_crossover(sol_a, sol_b, self.rng,
                                     self.crossover_inherit_prob)
        self.cx_stats["iterations"] += 1
        self.cx_stats["children"] += 1
        # E 诊断 (行为中性计数): 后代路由键中档案全池没有的新键数 —
        # "在线教育能否生成档案外新结构"的产率观测
        child_keys = ParetoArchive._route_keys({"solution": child})
        if child_keys:
            pool_keys = set()
            for e in self.archive.entries:
                pool_keys |= ParetoArchive._route_keys(e[1])
            self.cx_stats["novel_route_keys"] += len(child_keys - pool_keys)
        return child

    def _repair_unassigned(self, solution: Solution, unassigned: list[int]) -> Solution:
        """用贪心插入强制插入未分配的客户 (跨车场遍历所有路由)。

        修复前读 assigned_depot_index 决定车场 (共享可变状态, 可能被上一轮
        _sync_depot_assignments 污染)。现在跨车场遍历所有路由找容量可行位。
        """
        sol = solution.copy()
        inst = sol.instance
        cap = inst.max_vehicle_capacity

        from src.operators.repair import _append_duration_ok
        for ci in unassigned:
            c = inst.customers[ci - inst.num_depots]
            placed = False
            for di, routes in sol.routes.items():
                for ri, route in enumerate(routes):
                    route_demand = sum(
                        inst.customers[x - inst.num_depots].demand for x in route
                    )
                    if (route_demand + c.demand <= cap
                            and _append_duration_ok(inst, di, route, ci, ri)):
                        route.append(ci)
                        c.assigned_depot_index = di
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                # 无容量可行位 (紧绑定): 优先未满车场开新车; 全满才兜底
                # 插负载最轻路由 (不增车) — 旧"就近开新车"无条件增车,
                # 与车辆数约束冲突 → _trim_to_vehicle_limit 死循环
                from src.operators.repair import _lightest_route_insert, \
                    _nearest_depot_with_capacity
                di = _nearest_depot_with_capacity(sol, ci)
                if len(sol.routes[di]) < inst.depots[di].vehicles_available:
                    sol.routes[di].append([ci])
                    c.assigned_depot_index = di
                else:
                    _lightest_route_insert(sol, ci)

        return sol


    def _enforce_capacity(self, solution):
        if solution.is_feasible():
            return solution
        sol = solution.copy()
        inst = sol.instance
        cap = inst.max_vehicle_capacity
        to_reinsert = []
        for di in list(sol.routes.keys()):
            new_routes = []
            for route in sol.routes[di]:
                if not route:
                    continue
                load = sum(inst.customers[i - inst.num_depots].demand for i in route)
                if load <= cap:
                    new_routes.append(route)
                else:
                    to_reinsert.extend(route)
            sol.routes[di] = new_routes
        for ci in to_reinsert:
            c = inst.customers[ci - inst.num_depots]
            placed = False
            for di, routes in sol.routes.items():
                for route in routes:
                    rd = sum(inst.customers[x - inst.num_depots].demand for x in route)
                    if rd + c.demand <= cap:
                        route.append(ci)
                        c.assigned_depot_index = di
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                # 无容量可行位 (紧绑定): 优先未满车场开新车; 全满才兜底
                # 插负载最轻路由 (不增车) — 旧"就近开新车"无条件增车,
                # 与车辆数约束冲突 → _trim_to_vehicle_limit 死循环
                from src.operators.repair import _lightest_route_insert, \
                    _nearest_depot_with_capacity
                di = _nearest_depot_with_capacity(sol, ci)
                if len(sol.routes[di]) < inst.depots[di].vehicles_available:
                    sol.routes[di].append([ci])
                    c.assigned_depot_index = di
                else:
                    _lightest_route_insert(sol, ci)

        return sol

    def _enforce_feasibility(self, solution: "Solution") -> "Solution":
        """LS/修复后的可行性兜底: 容量强制 + TW 强制 (统一入口)。

        跨路由邻域 (relocate/swap/2-opt*) 曾不检查容量, LS 后无容量
        强制 → 档案混入容量违规解 (p01 max_load 278 vs 80)。此方法
        保证任何进入档案的解都通过容量 + TW 双重修复。

        性能: 内联容量/覆盖检查 (不调 is_feasible, 避免其内部重复计算
        TW 调度), TW 检查只算一次。
        """
        self._sync_depot_assignments(solution)
        inst = solution.instance
        cap = inst.max_vehicle_capacity
        expected = set(c.index for c in inst.customers)
        covers = solution.all_served_customers() == expected
        cap_ok = all(
            solution.route_load_fast(r) <= cap
            for rl in solution.routes.values() for r in rl
        )
        veh_ok = all(
            len(rl) <= inst.depots[di].vehicles_available
            for di, rl in solution.routes.items()
        )
        if not (covers and cap_ok and veh_ok):
            solution = self._enforce_capacity(solution)
            solution = self._enforce_vehicle_limit(solution)
        if inst.has_time_windows and solution.tw_violations():
            solution = self._enforce_tw(solution)
        # 初始可行化 = 搜索入场券 (低频高价值): 时长修复给足预算。
        # pr17 (72 客户/6 车/D=500) 实测: 默认 40k 预算修不动 (残 5 条),
        # 200k/60 轮收敛 (1.05s) — 紧车队实例的逃生需要长的扰动序列。
        solution = self._enforce_duration(solution, max_rounds=60,
                                          work_budget=200000, step_cap=6000)
        return solution

    def _enforce_vehicle_limit(self, solution: "Solution") -> "Solution":
        """车辆数硬约束: 超限车场移除最短路由, 客户跨车场贪心重插。

        (曾缺失 — ALNS 产出车辆数违规解: p08 车场 0 用 15 辆 > 上限 14,
        P0 探针 SCP 每车场车辆数约束抓到。2026-09-01 修复。)
        """
        from src.core.initial_solution import _trim_to_vehicle_limit

        return _trim_to_vehicle_limit(solution)

    def _enforce_tw(self, solution: "Solution") -> "Solution":
        """时间窗硬约束：将违规路由的客户全部移出并用 TW-aware 修复重插。

        策略: 找出所有含 TW 违规的路由 → 整条移除 → 用
        greedy_cost_tw_insertion 重新插入 (TW 权重回退 1000)。
        """
        from src.operators.repair import greedy_cost_tw_insertion

        inst = solution.instance
        if not inst.has_time_windows:
            return solution

        # 找出所有违规客户, 定位其所在路由并整条移除
        viol_customers = {v["customer"] for v in solution.tw_violations()}
        if not viol_customers:
            return solution

        removed: set[int] = set()
        routes = solution.routes
        for di, rlist in routes.items():
            for route in rlist:
                if set(route) & viol_customers:
                    removed.update(route)

        if not removed:
            return solution

        # 重建路由: 保留无违规路由
        new_sol = solution.copy()
        for di in list(new_sol.routes.keys()):
            kept = []
            for route in new_sol.routes[di]:
                if set(route) & removed:
                    continue  # 该路由含违规客户, 整条移除
                kept.append(route)
            new_sol.routes[di] = kept

        # 用 TW-aware 修复重插
        new_sol = greedy_cost_tw_insertion(new_sol, sorted(removed), self.rng)
        return new_sol

    def _tw_is_soft(self) -> bool:
        """TW 在当前配置下是否为软约束 (soft_tw 基础模式 / M4 种群罚域)。"""
        return bool(self.soft_tw or self.population_penalty_mode == "tw")

    def _enforce_duration(self, solution: "Solution",
                          full: bool | None = None,
                          work_budget: int = 40000,
                          step_cap: int = 3000,
                          max_rounds: int = 20) -> "Solution":
        """最大路线时长硬约束 (口径见 docs/durfix-spec-20260911.md §1)。

        快速路径: 无 D 或无超长路线 → 原样返回 (无 D 实例逐位不变)。
        兜底通道: "联合超额爬坡修复" (relocate/swap + 扰动重启, 见
        repair.combined_excess_repair)。full=None 时按模式定口径:
        TW 为硬约束 → 联合口径 (TW+duration : 时长修复的兜底放置会带出
        TW 违规, 紧车队下二者互相干扰, 联合求解才收敛 — 实测 pr11 每场
        1 辆); TW 为软约束 (soft_tw / M4 罚域) → 只算 duration (违规
        流量是那些机制的本体, 不得在时长修复中被顺手抹掉)。播种通道
        (_soft_tw_seed) 传 full=True — 播种需的是完全可行解。只搬移
        客户不增车; 仍违规时由上层 (接受/档案) 照常拒绝, 与 _enforce_tw
        同语义。
        """
        inst = solution.instance
        if inst.route_duration_limit is None:
            return solution
        if not solution.duration_violations():
            return solution
        from src.operators.repair import combined_excess_repair

        if full is None:
            full = not self._tw_is_soft()
        return combined_excess_repair(solution.copy(), self.rng,
                                      include_tw=full,
                                      work_budget=work_budget,
                                      step_cap=step_cap,
                                      max_rounds=max_rounds)

    def _sync_depot_assignments(self, solution: "Solution") -> "Solution":
        """同步所有客户的车场分配与当前路由一致 (多车场不变量)。

        所有修改路由的路径 (repair/_enforce_capacity/_enforce_tw/LS) 之后,
        在档案入库前统一对齐, 保证 assigned_depot_index 与路由一致。
        """
        inst = solution.instance
        for c in inst.customers:
            c.assigned_depot_index = -1
        for di, rl in solution.routes.items():
            for route in rl:
                for ci in route:
                    inst.customers[ci - inst.num_depots].assigned_depot_index = di
        return solution

    def _print_fallback(self, op_name: str, op_type: str, error: Exception, fallback: str):
        """打印回退消息，抑制重复出现。"""
        if op_name not in self._fallback_count:
            self._fallback_count[op_name] = 0
        self._fallback_count[op_name] += 1

        # Only print first occurrence and every 10th after that
        count = self._fallback_count[op_name]
        if count == 1 or count % 10 == 0:
            print(f"\n  [ALNS] LLM {op_type} operator '{op_name}' crashed: {error}")
            print(f"  [ALNS] Falling back to {fallback} (crash #{count})")

    def _track_operator_call(self, op_name: str):
        """追踪算子调用次数。"""
        self._op_call_count[op_name] = self._op_call_count.get(op_name, 0) + 1

    def _track_operator_crash(self, op_name: str):
        """追踪算子崩溃。高频崩溃的算子自动从池中移除。"""
        if op_name not in ("random_removal", "greedy_cost", "greedy_cost_insertion"):
            # 只追踪 LLM 算子
            pass
        self._op_crash_count[op_name] = self._op_crash_count.get(op_name, 0) + 1

        # 检查是否需要自动移除
        calls = self._op_call_count.get(op_name, 0)
        crashes = self._op_crash_count[op_name]
        if calls >= 10 and crashes >= calls * 0.5 and op_name not in self._op_removed:
            self._op_removed.add(op_name)
            # 从算子池中移除
            if op_name in self.destroy_ops:
                del self.destroy_ops[op_name]
                self.destroy_names = list(self.destroy_ops.keys())
                # 重新初始化选择器名称列表
                if hasattr(self.selector, 'names'):
                    self.selector.names = self.destroy_names + self.repair_names
            elif op_name in self.repair_ops:
                del self.repair_ops[op_name]
                self.repair_names = list(self.repair_ops.keys())
                if hasattr(self.selector, 'names'):
                    self.selector.names = self.destroy_names + self.repair_names
            print(f"\n  [ALNS] Auto-removed '{op_name}': {crashes}/{calls} crashes ({crashes/calls:.0%})")


# ── 种群壳纯函数 (M1, 2026-09-07) ──────────────────────────────
# 只做确定性计算, 不取随机源 (rng 由调用方从 engine.rng 链传入); 路由键集
# 只用于 dc/fitness/统计, 不决定搜索路径 (禁遍历 set 决定路径)。


# ── timefix (2026-09-12) 常量: 时间预算防饥饿引导 ──────────────
_TIMEFIX_TIME_MODE_CEIL = 1e8       # max_time ≥ 此值 = 迭代模式 (不触发)
_TIMEFIX_MIN_GAP_ITERS = 100        # 两次引导尝试最小间隔 (迭代)
_TIMEFIX_MAX_ATTEMPTS = 5           # 单 run 引导尝试上限
_TIMEFIX_ELAPSED_FRACTION = 0.30    # 触发所需已耗预算比例


def _member_refresh_threshold(stagnation_limit: int) -> int:
    """成员级刷新阈值 (裁定 2, 2026-09-07): max(50, stagnation_limit // 30)。

    公式写死, 不加 config 键 — 6000iter BASE (stagnation_limit=6000) → 200;
    短预算臂 (stagnation_limit=300) → max(50, 10) = 50。
    """
    return max(50, int(stagnation_limit) // 30)


def pool_accept_replace(sa_accepted: bool, child_feasible: bool) -> bool:
    """成员替换判定 (裁定 1, 2026-09-07): SA 接受 AND 子代可行才替换池成员。

    池成员只允许可行解 — SA 接受但子代不可行 → 成员不变, 该迭代按拒绝记账
    (score=0 / outcome.accepted=False)。主循环所有成员替换决策必须经此函数,
    禁内联绕过 (决策点抽成纯函数供测试锁策略)。
    """
    return bool(sa_accepted) and bool(child_feasible)


def pool_gate_improve(child_cost: float, m_cost: float, feasible: bool) -> bool:
    """成员改进门 (M3, 2026-09-07): 可行 AND 纯成本严格改进 m* (差 > 1e-9)。

    gate="improve" 下所有 cx 子代替换决策必须经此函数, 禁内联绕过 (纯成本
    严格改进, 不走 SA)。差 ≤ 1e-9 不算改进 (浮点相等视为未改进)。
    """
    return bool(feasible) and child_cost < m_cost - 1e-9


def pool_penalty_gate(child_pen: float, m_pen: float) -> bool:
    """罚后接受门 (M4, 2026-09-07): 罚后成本严格改进 m* (差 > 1e-9)。

    tw 罚域模式的成员替换判定: 子代可行与否都按罚后成本比 (可行性不再进门);
    差 ≤ 1e-9 不算改进 (浮点相等视为未改进)。确定性, 不走 SA。决策点必须经此
    函数, 禁内联绕过。
    """
    return child_pen < m_pen - 1e-9


def pool_penalized_cost(cost: float, excess: float, lambda0: float) -> float:
    """罚后成本 (M4, 2026-09-07): 可行 (excess ≤ 1e-9) = 真实成本;
    不可行 = cost + λ0 × excess (λ0 静态)。纯计算, 供 λ0 校准与成员扩展共用。
    """
    if excess <= 1e-9:
        return float(cost)
    return float(cost + lambda0 * excess)


def _pool_member_dc(i: int, keys_list: list) -> float:
    """成员 i 的池内多样性贡献 dc = min 对其它成员的路由集 Jaccard 距离。

    池唯一成员 (len(keys_list) <= 1) 时 dc = 1.0 (规格定义)。keys_list[i] =
    成员 i 的路由键集 (frozenset, 键约定 = ParetoArchive._route_keys 同构)。
    确定性: 按下标顺序全列表扫描, 无 set 迭代顺序依赖。
    """
    n = len(keys_list)
    if n <= 1:
        return 1.0
    ki = keys_list[i]
    best = 1.0
    for j, kj in enumerate(keys_list):
        if j == i:
            continue
        d = ParetoArchive._jaccard_dist(ki, kj)
        if d < best:
            best = d
    return best


def _pool_select_member(
    costs: list,
    keys_list: list,
    best_pool_cost: float,
    kappa: float,
    tournament: int,
    rng,
) -> int:
    """锦标赛选成员: 先无放回抽 min(tournament, μ) 个, 再取 fitness 最小者。

    fitness = cost − κ×best_pool_cost×dc (dc = _pool_member_dc; κ=0 → 退化
    纯成本)。并列取序号小者 (确定性)。抽取消耗 rng 的次序固定 (一次 choice
    先抽子集, 再比 fitness) — 同 (seed, config) 确定性。
    """
    n = len(costs)
    if n <= 0:
        raise ValueError("population pool is empty")
    k = min(int(tournament), n)
    candidates = rng.choice(n, size=k, replace=False)

    def _fitness(i: int) -> float:
        return costs[i] - kappa * best_pool_cost * _pool_member_dc(i, keys_list)

    return int(min(candidates, key=lambda i: (_fitness(i), i)))
