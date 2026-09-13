"""验证: 整数距离矩阵 vs 浮点距离, 哪个复现权威解 1083.98。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

from src.core.instance import instance_from_parsed
from src.core.objectives import calculate_cost
from src.core.solution import Solution
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw


def main():
    text = open("data/mdvrptw_raw/pr01.txt", encoding="utf-8").read()
    inst = instance_from_parsed(parse_mdvrptw(text), name="pr01")

    # 节点坐标
    nodes = [(d.x, d.y) for d in inst.depots] + [(c.x, c.y) for c in inst.customers]
    n = len(nodes)

    # 两种距离矩阵
    d_float = [[math.dist(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]
    d_round = [[round(math.dist(nodes[i], nodes[j])) for j in range(n)] for i in range(n)]
    d_floor = [[math.floor(math.dist(nodes[i], nodes[j])) for j in range(n)] for i in range(n)]
    d_ceil = [[math.ceil(math.dist(nodes[i], nodes[j])) for j in range(n)] for i in range(n)]

    # 解析权威解路由
    lines = open(str(Path(__file__).resolve().parent.parent / "data/mdvrptw_sol/pr01.res"),
                 encoding="utf-8").read().strip().splitlines()
    routes = {i: [] for i in range(inst.num_depots)}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        depot = int(parts[0]) - 1
        seq = [int(parts[i]) + inst.num_depots - 1 for i in range(4, len(parts), 2)
               if int(parts[i]) <= inst.num_customers]
        routes[depot].append(seq)

    # 用各距离矩阵计算成本
    for name, mat in [("浮点", d_float), ("四舍五入", d_round),
                      ("floor", d_floor), ("ceil", d_ceil)]:
        inst2 = inst._replace_distance_matrix(mat) if hasattr(inst, "_replace_distance_matrix") else None
        if inst2 is None:
            # 手动构造
            import copy

            inst2 = copy.copy(inst)
            inst2.distance_matrix = mat
        sol = Solution(inst2, routes)
        cost = calculate_cost(sol)
        tw = len(sol.tw_violations())
        print(f"{name:<10} 成本={cost:8.2f}  TW违规={tw}  {'✅ 匹配 .res' if abs(cost-1083.98)<0.01 else ''}")


if __name__ == "__main__":
    main()
