"""验证 C-mdvrptw-sol 的 pr01.res 解: 口径/可行性/成本。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.core.instance import instance_from_parsed
from src.core.objectives import calculate_cost
from src.core.solution import Solution
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw


def parse_res(path):
    """解析 .res: 第一行目标值, 后续行 车场 车辆 距离 负载 路径(客户(时间))。"""
    lines = open(path, encoding="utf-8").read().strip().splitlines()
    objective = float(lines[0].strip())
    routes = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        depot = int(parts[0])      # 车场编号 (49-52)
        vehicle = int(parts[1])
        dist = float(parts[2])
        load = float(parts[3])
        # 路径: 交替 客户号 (到达时间); 车场原始节点号 >= 49 (n_customers+1)
        seq = []
        i = 4
        while i < len(parts):
            cust = int(parts[i])
            if cust <= 48:  # 客户 (车场 49-52 跳过)
                seq.append(cust)
            i += 2  # 跳过 (时间)
        routes.append((depot, vehicle, dist, load, seq))
    return objective, routes


def main():
    text = open(str(Path(__file__).resolve().parent.parent) + r"/data/mdvrptw_raw/pr01.txt", encoding="utf-8").read()
    inst = instance_from_parsed(parse_mdvrptw(text), name="pr01")
    print(f"实例: {inst.name} 客户={inst.num_customers} 车场={inst.num_depots} 容量={inst.vehicle_capacity}")
    print(f"车场索引: {[d.index for d in inst.depots]} (客户索引 {inst.num_depots}-{inst.total_nodes-1})")

    obj, routes = parse_res(str(Path(__file__).resolve().parent.parent / "data/mdvrptw_sol/pr01.res"))
    print(f"\n.res 报告目标值: {obj}")

    # 转项目 Solution: .res 车场编号 1-4 (相对) → 项目索引 0-3
    # 客户编号 1-48 → 项目索引 4-51 (客户索引 = 原始号 + num_depots - 1)
    project_routes = {i: [] for i in range(inst.num_depots)}
    for depot, vehicle, dist, load, seq in routes:
        depot_idx = depot - 1  # .res 车场编号 1-4 → 0-3
        custs = [c + inst.num_depots - 1 for c in seq]  # 1-48 → 4-51
        project_routes[depot_idx].append(custs)

    sol = Solution(inst, project_routes)
    print(f"\n项目口径计算: 成本={calculate_cost(sol):.2f}")

    # 可行性
    print(f"容量可行: {sol.is_feasible()}")
    print(f"TW 违规: {len(sol.tw_violations())}")
    served = sol.all_served_customers()
    print(f"客户覆盖: {len(served)}/{inst.num_customers}")

    # 与 pyvrp / 公开 BKS 对比
    print(f"\n对比: .res={obj} vs pyvrp(30s)=1277.00 vs 引用BKS=1217.55")
    print(f"差距: .res 比 pyvrp 好 {(1277.00-obj)/1277.00*100:.1f}%")


if __name__ == "__main__":
    main()
