"""生成各算例静态解的路线可视化点线图。

对 pr01-pr05 各运行一次静态双目标 ALNS (120s 预算, 与论文一致),
取 Pareto 最优成本解, 绘制:
- 车场: 彩色方框 (含编号)
- 客户: 灰色圆点 (含编号, 字号小)
- 路线: 按所属车场着色连线, 起点为车场
- 标题: 实例名 + 成本 + gap vs 权威 .res + 车辆数

输出: docs/paper/figs/routes_pr01.png ... routes_pr05.png

用法: python experiments/plot_solution_routes.py [--budget 120]
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")

from scripts.parse_cordeau_mdvrptw import parse_mdvrptw
from src.core.instance import instance_from_parsed
from src.solvers.standard_alns import solve_standard_alns

plt.rcParams.update({"font.size": 8, "axes.titlesize": 11, "figure.dpi": 150})
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "KaiTi", "DejaVu Sans"]
# 坐标轴负号: 禁用 Unicode 减号 (U+2212), 否则 JhengHei 缺字形渲染成 ✔
plt.rcParams["axes.unicode_minus"] = False

# 车场配色 (最多 4 车场, 可扩展)
DEPOT_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

# 权威 .res 解 (成本单目标参考)
RES_BKS = {
    "pr01": 1083.98, "pr02": 1763.07, "pr03": 2408.42,
    "pr04": 2958.23, "pr05": 3134.04,
}


def load_static(name: str):
    text = open(f"data/mdvrptw_raw/{name}.txt", encoding="utf-8").read()
    return instance_from_parsed(parse_mdvrptw(text), name=name)


def make_config(n: int) -> dict:
    """与论文一致的 ALNS 配置 (120s 预算)。"""
    return {
        "max_iterations": 2000,
        "max_time_seconds": 120.0,
        "segment_length": 25,
        "reaction_factor": 0.1,
        "noise_stddev": 0.1,
        "min_improvement": 1e-6,
        "initial_temperature": 0.9,
        "cooling_factor": 0.995,
        "reheat_period": 200,
        "rng_seed": 1,
    }


def best_cost(archive) -> float:
    return min(e[0][0] for e in archive.entries)


def best_solution(archive, instance):
    """返回 Pareto 最优成本解 (最小成本)。"""
    from src.core.solution import Solution

    best_e = min(archive.entries, key=lambda e: e[0][0])
    if len(best_e) > 1 and isinstance(best_e[1], dict) and "solution" in best_e[1]:
        return best_e[1]["solution"]
    # 从路由重建
    return Solution(instance)


def plot_instance(name: str, budget: float = 120.0):
    static = load_static(name)
    cfg = make_config(static.num_customers)
    archive = solve_standard_alns(static, cfg, np.random.default_rng(1))
    sol = best_solution(archive, static)
    cost = best_cost(archive)
    gap = (cost - RES_BKS[name]) / RES_BKS[name] * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    depots = static.depots
    customers = static.customers

    # 客户点
    cx = [c.x for c in customers]
    cy = [c.y for c in customers]
    ax.scatter(cx, cy, s=18, c="#666666", zorder=3)
    for c in customers:
        ax.annotate(str(c.index), (c.x, c.y), fontsize=4.5,
                    textcoords="offset points", xytext=(2, 2), color="#555555")

    # 车场
    for i, d in enumerate(depots):
        ax.scatter([d.x], [d.y], marker="s", s=90, c=DEPOT_COLORS[i],
                   edgecolors="black", zorder=5)
        ax.annotate(f"D{i + 1}", (d.x, d.y), fontsize=9, fontweight="bold",
                    textcoords="offset points", xytext=(0, -16),
                    ha="center", color=DEPOT_COLORS[i])

    # 路线
    n_vehicles = 0
    for d_idx, routes in sol.routes.items():
        color = DEPOT_COLORS[d_idx % len(DEPOT_COLORS)]
        depot = depots[d_idx]
        for route in routes:
            n_vehicles += 1
            pts = [(depot.x, depot.y)] + \
                  [(customers[ci - static.num_depots].x,
                    customers[ci - static.num_depots].y) for ci in route] + \
                  [(depot.x, depot.y)]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-", color=color, lw=1.0, alpha=0.75, zorder=2)

    ax.set_title(f"{name}: 成本 {cost:.2f} (gap {gap:+.1f}%), "
                 f"{n_vehicles} 车, {static.num_depots} 车场, "
                 f"{static.num_customers} 客户".replace("\u2212", "-"), fontsize=11)
    ax.set_xlabel("x 坐标")
    ax.set_ylabel("y 坐标")
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out = f"docs/paper/figs/routes_{name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[图] {out}  (成本 {cost:.2f}, gap {gap:+.1f}%, {n_vehicles} 车)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=120.0)
    args = ap.parse_args()
    for name in ["pr01", "pr02", "pr03", "pr04", "pr05"]:
        plot_instance(name, args.budget)


if __name__ == "__main__":
    main()
