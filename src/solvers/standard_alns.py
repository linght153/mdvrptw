"""Standard ALNS solver — the primary baseline (净化版, 迁移自旧项目)。"""

import numpy as np

from src.core.instance import Instance
from src.alns.selection import SegmentRewardSelection
from src.alns.engine import ALNSEngine
from src.alns.archive import ParetoArchive
from src.operators.local_search import rvnd_improve


def solve_standard_alns(
    instance: Instance,
    config: dict,
    rng: np.random.Generator,
    initial_solution: "Solution | None" = None,
) -> ParetoArchive:
    """运行基于段的算子选择标准 ALNS（Pisinger & Ropke 2007）。

    Args:
        instance: 实例
        config: ALNS 配置
        rng: 随机数生成器
        initial_solution: 热启动解 (warm-start, 滚动时域重优化用)

    防御性拷贝: 求解过程会写 customers[].assigned_depot_index
    (repair 算子), 深拷贝隔离, 避免污染调用方共享的实例对象。
    """
    from dataclasses import replace as dc_replace
    from src.core.instance import Customer

    # 拷贝 customers (每客户独立对象), 求解过程修改不泄漏到调用方
    customers_copy = [
        dc_replace(c) for c in instance.customers
    ]
    instance = dc_replace(instance, customers=customers_copy)

    # 若传入热启动解, 其 instance 引用必须重定向到拷贝实例,
    # 否则 repair 算子写 initial_solution.instance.customers 仍污染原始对象
    if initial_solution is not None:
        from src.core.solution import Solution
        new_init = Solution(instance)
        new_init.routes = {
            d: [list(route) for route in routes]
            for d, routes in initial_solution.routes.items()
        }
        initial_solution = new_init
    destroy_names = list(__import__("src.operators.destroy", fromlist=["DESTROY_OPERATORS"]).DESTROY_OPERATORS.keys())
    if instance.has_time_windows:
        # TW 实例: 只用 TW-aware 修复 (纯距离插入几乎必然 TW 违规)
        repair_names = ["greedy_cost_tw"]
        from src.operators.repair import greedy_cost_tw_insertion

        repair_ops = {"greedy_cost_tw": greedy_cost_tw_insertion}
    else:
        repair_names = list(__import__("src.operators.repair", fromlist=["REPAIR_OPERATORS"]).REPAIR_OPERATORS.keys())
        repair_ops = None  # engine 默认全算子
    selector = SegmentRewardSelection(
        destroy_names,
        repair_names,
        segment_size=config.get("segment_size", 100),
        reaction_factor=config.get("reaction_factor", 0.1),
        decay_factor=config.get("decay_factor", 1.0),
        min_selection_prob=config.get("min_selection_prob", 0.005),
    )

    engine = ALNSEngine(
        instance=instance,
        config=config,
        rng=rng,
        selector=selector,
        repair_ops=repair_ops,
        local_search=rvnd_improve,
        initial_solution=initial_solution,
    )

    return engine.run()
