"""分析权威解 pr01: 每条路由的到达时间 vs TW, 理解 TW 可行结构。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from src.core.solution import Solution
from src.core.objectives import calculate_cost
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw


def main():
    text = open("data/mdvrptw_raw/pr01.txt", encoding="utf-8").read()
    inst = instance_from_parsed(parse_mdvrptw(text), name="pr01")
    lines = open(str(Path(__file__).resolve().parent.parent / "data/mdvrptw_sol/pr01.res"),
                 encoding="utf-8").read().strip().splitlines()

    print(f"实例: {inst.num_customers} 客户, 容量 {inst.max_vehicle_capacity}")
    print(f"服务时间: 客户0 s={inst.customers[0].service_time}")
    print(f"速度: {getattr(inst, 'speed', 'N/A')}")

    routes = {i: [] for i in range(inst.num_depots)}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        depot = int(parts[0]) - 1
        seq = [int(parts[i]) + inst.num_depots - 1 for i in range(4, len(parts), 2)
               if int(parts[i]) <= inst.num_customers]
        routes[depot].append(seq)

    sol = Solution(inst, routes)
    print(f"\n权威解: 成本={calculate_cost(sol):.2f}, TW违规={len(sol.tw_violations())}")
    print(f"路由数: {sum(len(v) for v in routes.values())}")

    # 逐路由 schedule
    for di in range(inst.num_depots):
        for ri, route in enumerate(routes[di]):
            schedule = sol.route_schedule(di, route)
            if schedule:
                first = schedule[0]
                last = schedule[-1]
                print(f" 车场{di} 路由{ri}: {len(route)}客户 "
                      f"首到达={first['arrival']:.0f}(窗[{inst.customers[route[0]-inst.num_depots].ready_time},"
                      f"{inst.customers[route[0]-inst.num_depots].due_time}]) "
                      f"末到达={last['arrival']:.0f} 末窗["
                      f"{inst.customers[route[-1]-inst.num_depots].ready_time},"
                      f"{inst.customers[route[-1]-inst.num_depots].due_time}]")


if __name__ == "__main__":
    main()
