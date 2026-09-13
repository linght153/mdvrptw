# pyvrp 官方 10-seed 对照预注册 (2026-09-08) — 论文平台表升级

背景: M5 平台表为 3-seed(42-44); pyvrp 官方基准协议 = 10 seeds 均值 gap
(pyvrp.org/setup/benchmarks.html; v0.13.0 起 MDVRPTW 官方口径 = 1h 固定终止,
0.69% avg gap; 且 v0.13 版 MDVRPTW 求解器已从 HGS 换 ILS — 版本注记入论文)。
复现官方 1h×10seeds = 30h 串行不可行 → **对照口径 = 同预算 600s × 10 seeds ×
双引擎**(预算公平), 官方 1h 数字仅作文本上下文(引用 pyvrp 官方表, 不重跑)。

## 1. 协议(冻结)

- 实例: pr15 / pr16 / pr20。种子: 1-10(两引擎同种子集; 自产惯例 42-44 为既有
  3-seed 表, 不混入本表)。
- 预算: 600s 固定(双方同预算; 时间预算红线: 串行独立进程)。
- 引擎:
  - pyvrp: scripts/probe_pyvrp_wide.py {inst} 600 {seed}(0.13.4; int 求解 +
    float 重算口径; pr16 float 不可行 → int + repair 上界注记, 与既有
    pyvrp_wide 3-seed 同款处理; 环境漂移 retry 逻辑沿用) →
    results/pyvrp_wide/{inst}_s{seed}_600s.json
  - 自产: experiments/benchmark_population_m5.py {inst} {seed} popcxge 600
    (popcxge 终形态原参) → results/population_m5/600s/{inst}_s{seed}_popcxge.json
  - base600 保持 M5 3-seed 数据(不扩 10-seed; 其角色 = 内部形态对照, 3-seed
    已判)。
- 编排: scripts/rerun_official10_600s.sh — 60 组串行(10 seeds × 3 实例 ×
  2 引擎), 逐组增量日志, **断点续跑**(输出已存在则跳过), 预计 ~10h 过夜。
- 环境: D:/mdvrp/.venv; OMP/OPENBLAS=1; 每进程独立。

## 2. 产物与判定(死值)

- 主表: 每实例每引擎 mean ± std over 10 seeds(best_feasible_cost;
  pr16 注记口径)。gap vs pyvrp 均值(自产 − pyvrp)/pyvrp。
- 判定: 报告性对照(论文平台表), 无机制开关 — 不设"过/不过"门槛; 但须如实
  报告: ①配对差方向与散布(10-seed 散布应远小于 600s 3-seed 的经验, 给出
  稳定的均值差); ②pyvrp 官方 1h avg gap(0.69% @v0.13.0)上下文; ③引擎语言/
  实现效率注记(pyvrp C++ vs 自产 Python — 同预算下迭代数差距为效率度量,
  与 6000iter 口径结果并列呈现: 论文"双口径"叙事)。
- 表注约定(用户偏好): 上标 ᵃ/ᵇ 标注配对显著性(paried t, n=10), 表内不列 p 值。
- 判定文档: docs/ma-alns-pyvrp-official10-verdict-20260908.md(数据表 + 论文
  表格 ready-to-use 版)。

## 3. 边界

不跑 6000iter 批(已有); 不并行(时间预算红线); 不断点重跑已存在组(续跑跳过);
结果 gitignore(results/), 分析脚本读 JSON 直接产出论文表。
