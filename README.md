# MDVRPTW 优化算法研究项目（重启基线）

省级大创《多车场车辆路径问题优化算法研究》研究代码库。
**2026-09-02 精简式整理后，本目录定为 MDVRPTW（带时间窗）研究重启基地
（同日由 D:\mdvrp_backup_20260901 重命名为 D:\mdvrptw；算例主线 = pr 系列，
Cordeau 2001 MDVRPTW，VRP-REP 2017-0012）。**

## 仓库沿革（重要，先读）

- 2026-08 全史：动态需求预测 → 场景化 ALNS → 绿色双目标（D4 路线）多轮实验与论文，
  经用户拍板判定动态/绿色主线为"前人结论复现"，无独立研究价值（详见
  `docs/future-directions-alns-mdvrp.md` 与 D:\mdvrp 仓库的 09-02 方向调研文档）。
- 2026-09-01：D:\mdvrp 执行"回归纯 MDVRP（无 TW）"重构（commit 563e7a4），
  删除动态/场景化/绿色资产与 MDVRPTW 数据；本目录为其回归前完整快照。
- 2026-09-02：本目录精简式整理 → 成为 MDVRPTW 重启基地（本次整理）。
  - **保留**：带 TW 的完整 ALNS 引擎、18 个引擎/TW 回归测试、pr01-20 实例 +
    pr01-05 官方权威解、无 TW p 系列（green_mdvrp）、引擎验证实验与关键方法论文档。
  - **移除（git rm，完整可恢复）**：src/forecast、src/green、src/scenario（动态/绿色线）、
    全部旧论文（docs/paper）、动态验证文档与实验、results/ 旧数据、tmp 渲染与临时文件。
  - **恢复手段**：`git tag pre-reorg-20260902`（整理前 HEAD）＋
    `D:\mdvrp_backup_20260901_pre-reorg.bundle`（全历史 bundle，35MB）＋
    `D:\mdvrp_backup_20260901_results_archive_20260902.zip`（results/，gitignored 不入库）。
    例：`git checkout pre-reorg-20260902 -- src/scenario` 即可取回任意文件。
- **姊妹仓库**：D:\mdvrp = 纯 MDVRP（无 TW）演进线。引擎含本目录没有的更新：
  granular 邻域（0ca54f8）、repair 无 TW 快速路径、车辆数上限硬约束（304153b）、
  紧绑定死循环修复（2c94aa5），另有 P0-P6 探针（方向探索，已止损）与
  p01-p23 无 TW 实例（mdvrp_raw .txt 33 个 + green_mdvrp .json 21 个）。
  **合流决策（何时把演进引擎移植回 TW 线、如何回归 TW 行为）留待方向确定后执行。**

## 目录结构

```
├── src/
│   ├── core/          # Instance / Solution / objectives（成本+TW+排放惰性引用）/
│   │                  #   initial_solution / pareto
│   ├── alns/          # ALNS 引擎（acceptance/archive/parent_selection/selection…）
│   ├── operators/     # destroy / repair（TW-aware greedy 等）/ local_search（RVND）
│   ├── solvers/       # standard_alns 入口 + pyvrp_solver（SOTA 对照基线）
│   ├── evaluation/    # metrics（ScPO 式归一化 / HV / IGD）
│   └── utils/
├── data/
│   ├── mdvrptw_raw/   # ★ pr01-pr20 MDVRPTW 实例（Cordeau 2001 格式，含 TW）
│   ├── mdvrptw_sol/   # ★ pr01-pr05 官方权威解 .res（BKS 口径）
│   └── green_mdvrp/   # p01-p23 子集等无 TW MDVRP 实例（跨口径对比用）
├── tests/             # 18 个引擎 + TW 硬约束回归测试
├── experiments/       # 15 个：TW 权威验证/静态基准/收敛/路线可视化
├── scripts/           # parse_cordeau_mdvrptw.py + strict_verification*.py
├── docs/              # BKS 权威/口径红线/正确性修复史/静态方向调研
└── 调研成果/          # ALNS/DL/ML-MDVRP 算法综述 + 方向凝练
```

## 权威 BKS 与口径红线（写论文/跑实验必守）

- **pr01-05 MDVRPTW 权威解 = 1083.98 / 1763.07 / 2408.42 / 2958.23 / 3134.04**
  （官方 .res，浮点欧氏距离口径，math.dist 逐位复现；1217.55 等文献旧值已被改进替代，
  详见 `docs/bks-verification.md`）。
- 求解器输出必须用权威解 / TW 违规计数交叉核对再写论文；TW 违规解历史教训
  见 `docs/tw-hard-constraint-fix.md`。
- 大实例（≥144 客户）时间预算实验不可逐位复现（系统负载 → 迭代数波动），
  必须串行 + 独立进程跑，可复现性验证用固定迭代数（详见
  `docs/rerun-after-tw-fix.md` 与技能 mdvrp-research）。
- pr06-20 无官方 .res（本目录仅有 pr01-05），做大规模对比需文献值或自行运行 SOTA。

## 快速开始

```bash
# 依赖（无 .venv；可临时复用 D:\mdvrp\.venv 或自建）
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt

# 测试
python -m pytest tests/ -q

# 冒烟：pr01 短预算 ALNS（TW 可行 + 零违规 + 全客户服务）
python experiments/verify_cordeau_res.py   # 或 scripts/strict_verification.py（多维度验收）
```

## 重启待办（方向确定后）

1. 方向决策（用户思考中；素材：本仓库 `docs/future-directions-alns-mdvrp.md` +
   D:\mdvrp\docs/three-directions-deep-survey-20260902.md）。
2. 引擎合流评估：移植 D:\mdvrp 的 granular/车辆数硬约束/死循环修复等演进，
   在 TW 实例上全量回归（TW 行为不得退化）。
3. 预注册实验设计（时间预算公平对比、seeds 统计口径，沿用历史协议）。
