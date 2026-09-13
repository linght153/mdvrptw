# MDVRPTW · 种群化大邻域搜索（MA-ALNS）

多车场带时间窗车辆路径问题（Multi-Depot Vehicle Routing Problem with Time Windows，
MDVRPTW）的算法研究代码库。核心问题：把 ALNS 内核装进**种群外壳**后，让重组真正
起作用的最小继承单位是什么 —— 路由，还是车场分配？本仓库给出该判别实验的完整实现、
组件叠加链、以及与国际主流求解器（pyvrp / HGS）的双口径对照。

## 研究概要

| 项 | 说明 |
|---|---|
| 问题 | MDVRPTW（Cordeau et al. 2001；VRP-REP 数据集 2017-0012）。目标 = 总行驶距离最小；时间窗为硬约束；每车场车辆数有上限；路线最大时长有约束 |
| 引擎 | 单轨迹 ALNS：destroy / repair 算子族 + RVND 多邻域局部搜索 + 分数 / 接受 / 降温控制。所有机制 config gated、默认关，默认路径行为逐位不变 |
| 方法 | MA-ALNS：成员种群池 + 锦标赛选择（fitness = cost − κ·best·dc，dc 为池内路由集 Jaccard 距离）+ **车场分配簇级重组** + 改进门隔离预算 + 池内强化 + 预算感知教育 |
| 对照 | pyvrp 0.13.4（HGS 家族，C++ 内核）：固定迭代与固定时间预算两个口径 |

## 目录结构

```text
src/               引擎与解法
  core/            实例 / 解 / 目标函数 / 初始解 / Pareto 档案
  alns/            ALNS 引擎（接受准则 / 精英档案 / 父代选择 / 算子选择）
  operators/       destroy / repair / RVND 局部搜索 / 交叉算子
  solvers/         standard_alns 入口、pyvrp 对照适配
  evaluation/      评价指标（HV / IGD 等）
tests/             回归测试（230 项，含时间窗硬约束与增量一致性）
experiments/       实验脚本：基准 runner / 消融 / 收敛轨迹 / 对照分析
scripts/           Cordeau 算例解析器（parse_cordeau_mdvrptw.py，tests 与 experiments 共用）
data/              算例与参照值
results/           实验输出（逐次运行的 JSON）
docs/              实验预注册（prereg）与判定（verdict）文档链
```

## 环境与安装

Python 3.12。

```bash
pip install numpy scipy pandas pyyaml pyvrp==0.13.4 pytest
# 路线图与论文图脚本另需：pip install matplotlib
```

## 快速开始

```bash
# 1) 全量回归测试
python -m pytest tests/ -q

# 2) 官方解口径核验（pr01 官方 .res：成本 / 容量 / 时间窗 / 客户覆盖）
python experiments/verify_cordeau_res.py

# 3) 基准 runner（单实例 × 单 seed × 单臂，独立进程）
python experiments/benchmark_population_ab.py  pr15 42 base 6000    # 探针臂：base|div|cx
python experiments/benchmark_population_m2.py  pr15 42 popcx_part   # 重组单位判别
python experiments/benchmark_population_m5.py  pr15 42 popcxge      # 6000 次迭代
python experiments/benchmark_population_m5.py  pr15 42 popcxge 600  # 600s 时间预算

# 4) 汇总判定（读 results/ 出表）
python experiments/analyze_population_ab.py

# 5) pyvrp 对照基线（pr01–pr05，每实例 30 s）
python experiments/benchmark_pyvrp_cordeau.py
```

> 注意：请在仓库根目录以 `python -m pytest`（而非裸 `pytest`）运行，使 `src/`、
> `scripts/` 可被解析。

## 算例数据（data/）

| 路径 | 内容 |
|---|---|
| `mdvrptw_raw/` | pr01–pr20 原始算例（Cordeau 2001 格式，含时间窗），`manifest.json` 为清单 |
| `mdvrptw_sol/` | pr01–pr05 官方 `.res` 解（口径核验基准） |
| `green_mdvrp/` | p 系列无时间窗子集（跨口径对照用） |
| `mdvrptw_bks_2026.json` | pr01–pr20 参照值（1997 官方 .res / 文献 BKS / MDFIHA 2026 新界），整理与口径说明见 `docs/mdvrptw-bks-2026.md` |

参照值口径：pr01–pr05 有官方 `.res`；pr06–pr20 无官方解，参照值取文献 BKS 与
MDFIHA (2026) fBest 中的更优者。

## 实验结果

以下为 6000 次迭代（s42–44 均值，Δ = 臂 − 单轨迹 base）的组件叠加链；
原始数据在 `results/`，判定依据见对应 `docs/*-verdict-*.md`。

| 臂 | pr15 | pr16 | pr20 | Δ15 | Δ16 | Δ20 |
|---|---:|---:|---:|---:|---:|---:|
| base（单轨迹） | 2664.4 | 3128.0 | 3375.4 | — | — | — |
| pop（种群池） | 2677.1 | 3029.9 | 3498.6 | +12.7 | −98.1 | +123.2 |
| popcx_route（路由重组） | 2599.1 | 3078.4 | 3403.4 | −65.3 | −49.6 | +28.1 |
| popcx_part（车场分配簇重组） | 2594.7 | 3000.0 | 3261.5 | −69.7 | −128.0 | −113.9 |
| popcxg（+ 改进门） | 2641.8 | 3027.8 | 3357.6 | −22.6 | −100.2 | −17.8 |
| popcxge（+ 预算感知教育） | 2607.8 | 2926.9 | 3274.9 | −56.6 | −201.1 | −100.5 |

**主判别（10 seeds 配对，方法 vs 单轨迹 base）**

| 实例 | base | MA-ALNS | Δ | 负种子 |
|---|---:|---:|---:|---:|
| pr15 | 2640.8 | 2543.7 | −97.1 | 10/10 |
| pr16 | 3144.5 | 2941.0 | −203.5 | 10/10 |
| pr20 | 3516.2 | 3300.7 | −215.5 | 10/10 |

**与参照值对照（扩展算例，s42–44 均值）**

| 实例 | 参照值 | MA-ALNS | gap |
|---|---:|---:|---:|
| pr01 | 1074.12 | 1104.1 | +2.79% |
| pr04 | 2814.34 | 2909.0 | +3.36% |
| pr06 | 3588.78 | 3846.4 | +7.18% |
| pr07 | 1418.22 | 1435.9 | +1.24% |
| pr10 | 3465.54 | 3644.5 | +5.16% |
| pr13 | 2001.81 | 2025.0 | +1.16% |

8 例（pr15、pr16 + 上表 6 例）平均 gap：MA-ALNS **+3.91%**，单轨迹 base +7.35%。

**时间预算口径（600 s，pyvrp 走标准转换）**

| 实例 | 参照 BKS | MA-ALNS 600s | pyvrp 600s |
|---|---:|---:|---:|
| pr15 | 2433.15 | 2589.9（+6.44%） | 2577.2（+5.92%） |
| pr16 | 2836.67 | 3033.4（+6.93%） | 2961.3（+4.40%） |
| pr20 | 2983.78 | 3681.9（+23.40%） | 3175.7（+6.43%） |

600 s 口径下 pyvrp ≥ MA-ALNS（三实例）；固定迭代主口径下的方向性结论见上表。

> 综合账与数字底稿：`docs/durfix-summary-20260912.md`；等墙钟对照（pr16 −102.7、
> pr20 −138.4）与参数敏感性见同文档 §5 / §4。

## 复现口径（重要）

- **判定主口径 = 固定迭代（6000 次）**；时间预算（600 s）为辅口径 —— 后者受系统
  负载影响迭代数，必须**串行 + 独立进程**运行。
- **判定门槛**：臂间配对差须大于同臂跨 seed 散布；关键判定 ≥ 3 seeds（主判别为
  10 seeds 配对）。
- **预注册 → 跑批 → 判定**：每个实验先写判据死值的预注册文档，跑完出判定文档，
  两者都在 `docs/`（`*-prereg-*.md` / `*-verdict-*.md`）。未通过判定的机制探针
  （负结论）同样保留在仓库与文档链中。
- 结果 JSON 字段：`instance / seed / arm / iterations / wall_s / best_feasible_cost /
  vehicles / tw_violations / served / archive_* / probe_stats / population_stats`。

## 引用

算例来自 Cordeau, Laporte & Mercier (2001), *A unified tabu search heuristic for
vehicle routing problems with time windows*, JORS 52(8):928–936（VRP-REP 数据集
2017-0012）。现代参照值来源与核对记录见 `docs/mdvrptw-bks-2026.md` 与
`docs/bks-verification.md`。
