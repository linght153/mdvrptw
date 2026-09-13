"""关键验证: ALNS 静态解的 TW 违规情况。

疑点: ALNS 收敛 861.32 (MDVRP 参考值) < 权威 MDVRPTW 解 1083.98 (-20%)。
若 ALNS 解 TW 违规 > 0, 则 861.32 是"忽略 TW 的距离最优", 不能与权威解比较。
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.core.instance import instance_from_parsed
from src.solvers.standard_alns import solve_standard_alns
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

CONFIG = {
    "max_iterations": 1500, "max_time_seconds": 20.0, "stagnation_limit": 300,
    "segment_size": 100, "reaction_factor": 0.1, "decay_factor": 1.0,
    "min_selection_prob": 0.005, "initial_temperature": None, "cooling_rate": None,
    "archive_capacity": 20, "parent_selection": "crowding",
    "initial_solution": "nearest", "k_min_ratio": 0.10, "k_max_ratio": 0.40,
    "sigma": 3.0, "local_search_max_iter": 10, "local_search_freq": 5,
    "local_search_on_accept": True, "local_search_final": False,
}


def main():
    text = open("data/mdvrptw_raw/pr01.txt", encoding="utf-8").read()
    inst = instance_from_parsed(parse_mdvrptw(text), name="pr01")
    print(f"实例 pr01: {inst.num_customers} 客户, has_time_windows={inst.has_time_windows}")
    print(f"客户 TW 示例: c0 [{inst.customers[0].ready_time}, {inst.customers[0].due_time}]")
    print(f"tw_penalty_weight: {inst.tw_penalty_weight}")

    archive = solve_standard_alns(inst, CONFIG, np.random.default_rng(42))
    # 检查档案每个解的 TW 违规 (entry = (obj_tuple, meta_dict), 解在 meta['solution'])
    print(f"\n档案点: {len(archive.entries)}")
    for i, entry in enumerate(archive.entries[:5]):
        obj = entry[0]
        sol = entry[1]["solution"]
        tw = len(sol.tw_violations())
        feas = sol.is_feasible()
        print(f"  #{i}: cost={obj[0]:.2f} emission={obj[1]:.2f} "
              f"容量可行={feas} TW违规={tw}")

    # 最优解详细
    best = min(archive.entries, key=lambda e: e[0][0])
    best_sol = best[1]["solution"]
    print(f"\n最优解 cost={best[0][0]:.2f}: TW 违规 {len(best_sol.tw_violations())}")
    if best_sol.tw_violations():
        print(f"  违规详情 (前3): {best_sol.tw_violations()[:3]}")


if __name__ == "__main__":
    main()
