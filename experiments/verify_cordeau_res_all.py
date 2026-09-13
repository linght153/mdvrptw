"""批量验证 Cordeau .res 权威解: 可行性 + 成本复现 + 对比。

对 pr01-pr05: 解析 .res → 项目 Solution → 验证容量/TW/覆盖 → 成本复现。
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from src.core.objectives import calculate_cost
from src.core.solution import Solution
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

RES_DIR = str(Path(__file__).resolve().parent.parent / "data/mdvrptw_sol")
DATA_DIR = str(Path(__file__).resolve().parent.parent) + "/data/mdvrptw_raw"


def parse_res(path, n_customers):
    """解析 .res: 第一行目标值, 后续 车场 车辆 距离 负载 路径(客户(时间))。"""
    lines = open(path, encoding="utf-8").read().strip().splitlines()
    objective = float(lines[0].strip())
    routes = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        depot = int(parts[0])      # 车场编号 1-4
        seq = []
        i = 4
        while i < len(parts):
            cust = int(parts[i])
            if cust <= n_customers:  # 客户 (车场原始号 n+1.. 跳过)
                seq.append(cust)
            i += 2
        routes.append((depot, seq))
    return objective, routes


def main():
    print(f"{'实例':<8}{'.res目标':<12}{'项目成本':<12}{'容量':<6}{'TW违规':<8}{'覆盖':<8}"
          f"{'pyvrp(30s)':<12}{'.res优势'}")
    print("-" * 76)
    for name in ["pr01", "pr02", "pr03", "pr04", "pr05"]:
        text = open(f"{DATA_DIR}/{name}.txt", encoding="utf-8").read()
        inst = instance_from_parsed(parse_mdvrptw(text), name=name)
        n_cust = inst.num_customers
        obj, routes = parse_res(f"{RES_DIR}/{name}.res", n_cust)

        project_routes = {i: [] for i in range(inst.num_depots)}
        for depot, seq in routes:
            depot_idx = depot - 1
            custs = [c + inst.num_depots - 1 for c in seq]
            project_routes[depot_idx].append(custs)

        sol = Solution(inst, project_routes)
        cost = calculate_cost(sol)
        tw = len(sol.tw_violations())
        served = len(sol.all_served_customers())
        feasible = sol.is_feasible()

        # pyvrp 参照 (已有基准数据)
        pyvrp_ref = {"pr01": 1277.00, "pr02": 2016.00, "pr03": 3071.00,
                     "pr04": 3255.00, "pr05": 3652.00}[name]
        advantage = (pyvrp_ref - obj) / pyvrp_ref * 100

        print(f"{name:<8}{obj:<12.2f}{cost:<12.2f}{str(feasible):<6}{tw:<8}"
              f"{served}/{n_cust:<4}{pyvrp_ref:<12.2f}{advantage:+.1f}%")


if __name__ == "__main__":
    main()
