"""Cordeau MDVRPTW 复现表: pyvrp SOTA 基线 vs 公开 BKS。

BKS 来源: Cordeau et al. (2001) 报告的最优/已知最优 (VRP-REP)。
跑 pr01-pr05 (小规模), 每实例 30 秒, 报告 gap。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from src.solvers.pyvrp_solver import solve_with_pyvrp
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

# Cordeau 2001 MDVRPTW BKS (总距离, 来自文献引用; 待登录 VRP-REP 核对)
# pr01 值 (1217.55) 在 Cordeau 2001 JORS 论文结果表中; pr02-pr05 待核对。
# 内部对比以"pyvrp 固定时间预算"为 SOTA 基线, 外部 BKS 仅作参考。
BKS = {
    "pr01": 1217.55,
    "pr02": None,
    "pr03": None,
    "pr04": None,
    "pr05": None,
}

INSTANCES = ["pr01", "pr02", "pr03", "pr04", "pr05"]
RUNTIME = 30


def main():
    print(f"{'实例':<8}{'客户':<6}{'车场':<6}{'pyvrp成本':<12}{'时间':<8}{'BKS':<12}{'gap':<10}")
    print("-" * 62)
    for name in INSTANCES:
        text = open(f"data/mdvrptw_raw/{name}.txt", encoding="utf-8").read()
        parsed = parse_mdvrptw(text)
        inst = instance_from_parsed(parsed, name=name)

        started = time.perf_counter()
        result = solve_with_pyvrp(inst, max_runtime_seconds=RUNTIME, seed=42)
        elapsed = time.perf_counter() - started
        cost = result.cost()
        feasible = result.is_feasible()

        bks = BKS.get(name)
        gap = f"{100 * (cost - bks) / bks:+.2f}%" if bks else "N/A"
        print(f"{name:<8}{inst.num_customers:<6}{inst.num_depots:<6}"
              f"{cost:<12.2f}{elapsed:<8.1f}{str(bks):<12}{gap:<10}"
              f"{'✓' if feasible else '✗'}")


if __name__ == "__main__":
    main()
