"""LS 收敛对照 (2026-09-06): 两侧 dump 解在额外深度 RVND 下的降幅。

判别: 若自产解在额外 LS 下不降 (已是 LS 局部最优) 而 pyvrp 解也不降 →
两方法都是各自 LS 的局部最优, 结构性成本差 = 路由划分/成员组合差异
(LS 邻域无法修复), 不是 LS 深度差异。若自产解还能大幅下降 → 自产 LS
未跑透 (链组织技术差), 机制归因不同。只读 dump, rng 固定。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import instance_from_parsed
from src.core.objectives import calculate_objectives
from src.core.solution import Solution
from src.operators.local_search import rvnd_improve
from scripts.parse_cordeau_mdvrptw import parse_mdvrptw

ROOT = Path(__file__).resolve().parent.parent


def main():
    for name in ["pr15", "pr20"]:
        inst = instance_from_parsed(
            parse_mdvrptw(open(ROOT / f"data/mdvrptw_raw/{name}.txt",
                               encoding="utf-8").read()), name=name)
        print(f"== {name} ==")
        for seed in (42, 43, 44):
            sides = {}
            po = ROOT / "results" / "base600s_routes" / f"{name}_s{seed}_routes.json"
            if po.exists():
                sides["ours"] = json.loads(po.read_text(encoding="utf-8"))["routes"]
            else:
                po = ROOT / "results" / f"connectivity_{name}_s42_routes.json"
                if seed == 42 and po.exists():
                    sides["ours"] = json.loads(po.read_text(encoding="utf-8"))["routes"]
            pp = ROOT / "results" / "pyvrp_wide" / f"{name}_s{seed}_600s.json"
            if pp.exists():
                d = json.loads(pp.read_text(encoding="utf-8"))
                if d.get("feasible"):
                    sides["pyvrp"] = {}
                    for r in d["routes"]:
                        sides["pyvrp"].setdefault(int(r["depot"]), []).append(
                            list(r["seq"]))
            for side, routes in sides.items():
                sol = Solution(inst, {int(k): v for k, v in routes.items()})
                assert sol.is_feasible(), f"{name} s{seed} {side} 起点不可行"
                before = calculate_objectives(sol)[0]
                rng = np.random.default_rng(seed)
                t0 = time.time()
                refined = sol
                # 多轮深 LS (每轮 rvnd 到无改进自停), 直到连续两轮零改进
                for _ in range(5):
                    cand = rvnd_improve(refined, max_iterations=5000, rng=rng)
                    c = calculate_objectives(cand)[0]
                    if c >= calculate_objectives(refined)[0] - 1e-9:
                        break
                    refined = cand
                after = calculate_objectives(refined)[0]
                print(f"  {side:6s} s{seed}: closed {before:.2f} → {after:.2f} "
                      f"(Δ {after-before:+.2f}, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
