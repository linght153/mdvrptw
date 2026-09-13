# M1 种群壳 A/B 预注册 (2026-09-07) — 轴2 Memetic-ALNS 里程碑 1

设计背景: docs/ma-alns-design-20260907.md; 实现规格(CC 执行): docs/ma-alns-m1-spec-20260907.md。
判定协议归 Hermes; 本文件在实验跑批前冻结, 事后判定写 docs/ma-alns-m1-verdict-*.md。
增补 1 (2026-09-07, 裁定 2 后): 见文末"预注册增补 1"。判定规则(第 4 节)不受增补影响。

## 1. 问题与假设

M1 只隔离"种群外壳语义"本身(池 + 锦标赛 + dc fitness + 池级刷新, 无重组无教育),
预期质量 ≈ base。假设:
- H0: 壳无负效 — population 臂每实例 3-seed 配对均值差不显著劣于 base
  (差 ≤ 同臂种子散布 → 中性)。
- H1: dc 臂(popdc)档案路由多样性 ≥ 纯成本臂(pop)— dc fitness 真的在选择,
  非空转; 且 popdc 不劣于 pop。
M1 通过 = H0+H1 成立 + 引擎级验收(测试)全绿 → 上 M2(判别实验)。
M1 负效显著(pop 显著劣 base)→ verdict 修壳, 不上 M2。

## 2. 口径(全部冻结)

- 实例 × seed: pr15 / pr16 / pr20 × s42 / s43 / s44(与既有 base 配对文件同集)
- 迭代口径: 6000 iter 固定迭代(确定性 → 并行安全)。**不跑 600s**(M5 才进)。
- 配置: BASE 逐键 = experiments/benchmark_population_ab.py 的 BASE
  (max_iterations=6000, max_time_seconds=1e9, stagnation_limit=6000,
  segment_size=100, reaction_factor=0.1, decay_factor=1.0,
  min_selection_prob=0.005, archive_capacity=20, parent_selection=crowding,
  initial_solution=nearest, k_min_ratio=0.10, k_max_ratio=0.40, sigma=3.0,
  local_search_max_iter=10, local_search_freq=5, local_search_on_accept=True,
  local_search_final=False) — 与既有 population_ab base 文件逐位可比。
- 臂(arm_keys):
  - base = {} (不重跑! 配对参照 = results/population_ab/{inst}_s{seed}_base.json)
  - pop = {population_mode: True, population_size: 8, population_tournament: 2,
           population_dc_kappa: 0.0}
  - popdc = 同上但 population_dc_kappa: 0.05
- runner: experiments/benchmark_population_m1.py, CLI `{inst} {seed} {arm}`
  (复用 benchmark_population_ab.py 骨架: 实例拷贝 dc_replace、selector 构造
  destroy_ops 全表 × greedy_cost_tw、rvnd_improve、on_archive_add 日志);
  输出 results/population_m1/{inst}_s{seed}_{arm}.json, 字段 = population_ab
  同构(best_feasible_cost/vehicles/tw_violations/served/num_customers/
  archive_entries/archive_distinct_routes/archive_mean_jaccard/cx_iterations/
  wall_s) + population 扩展: {population_stats, pool_final_spread}
  (pool_final_spread = population_final 成员两两路由集 Jaccard 距离均值)。

## 3. 跑批(判定前冻结)

- 18 组 = 2 臂 × 3 实例 × 3 seeds; 6000iter 确定性 → xargs -P9 并行安全
  (OMP/OPENBLAS=1 单线程; 预计 ~15-25 min, 与既有 6000iter 批同量级)。
- 命令模式(独立进程): python experiments/benchmark_population_m1.py pr15 42 pop
  (bash for 循环; 每组 python 独立进程)。
- 环境: D:/mdvrp/.venv/Scripts/python.exe; workdir = 仓库根。

## 4. 判定规则(预注册死值, 事后不调)

定义: 臂 X vs base 的配对差 d_i = mean(X_i − base_i) over 3 seeds(每实例独立算);
同臂散布 s = base 3-seed 的 std(每实例)。
1. H0 通过: 每实例 |d| ≤ s(差被同臂散布吞没 → 中性)。
   H0 否证: 任一实例 d > s(壳负效)→ 修壳 verdict;
   d < −s(壳正效, 意外)→ 复验后记入 verdict, 同样放行 M2。
2. H1 通过: popdc 的 archive_distinct_routes 与 archive_mean_jaccard 不低于
   pop(每实例 3-seed 均值), 且 popdc 质量判据同 H0。
3. 引擎验收: 全量 pytest 绿(记录数量); 确定性抽查(同 seed 重跑逐位同值,
   runner 层做 1 组); 池过程指标(population_stats)无异常(报告即可, 不设门槛)。
判定文档: docs/ma-alns-m1-verdict-20260907.md(Hermes 写)。

---
## 预注册增补 1 (2026-09-07, 裁定 2 后冻结; 判定规则不变)

1. runner JSON 增补字段(population 臂): population_stats(引擎直出, 含
   member_refresh_events)、pool_final_spread、pool_feasible_members
   (population_final 中可行成员数 / population_size)。
2. 池健康度为过程诊断字段, 报告不设门槛 — 但若 6000iter 后 pop 臂
   pool_feasible_members < μ 的 50%, verdict 必须记载"壳健康度不足"并作为
   M2 立项的前置修复项。
3. 冒烟/诊断口径以 docs/ma-alns-m1-adjudication2-20260907.md 第 2 条为准
   (60iter 硬门槛 = 不崩 + archive best 可行 + 池可行成员 ≥1 + 墙钟 ≤2×;
   250iter κ=0.05 s42 诊断跑只报告)。
