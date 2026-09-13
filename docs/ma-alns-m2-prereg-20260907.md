# M2 池内重组判别 A/B 预注册 (2026-09-07) — 轴2 Memetic-ALNS 里程碑 2

设计/实现规格: docs/ma-alns-m2-spec-20260907.md; M1 判定: docs/ma-alns-m1-verdict-20260907.md。
判定协议归 Hermes; 本文件冻结于跑批前; 事后判定写 docs/ma-alns-m2-verdict-*.md。

## 1. 问题与假设(轴2 灵魂问题)

M1 壳正效(pr16/pr20 超散布)。M2 在同一壳上装重组, 判别**重组单位**:
- H_A: load-bearing 单位 = 路由级 → route 臂(popcx_route)质量 ≥ partition 臂。
- H_B: load-bearing 单位 = 车场分配簇(结构差判定: 分配强吸引子/划分盆地不可达
  的主坐标是车场归属)→ partition 臂(popcx_part)质量 > route 臂。
- 附带: 两臂 vs pop(壳基线)— 重组在池内是增益还是白费(M1 cx 家族单轨迹史:
  增益预期低, 如实记录即可, 不设乐观预期)。
- 中间结构指标(报告性): partition 胜出时检验 best 解错配罚金是否向 pyvrp
  ~890 量级靠拢(结构差判定锚), 不设死值。

## 2. 口径(冻结)

- 实例 × seed: pr15 / pr16 / pr20 × s42-44(同 M1)。
- 迭代口径: 6000 iter 固定迭代(确定性 → xargs -P9 并行)。不跑 600s(M5)。
- 配置: BASE 逐键 = M1(experiments/benchmark_population_m1.py 的 BASE);
  臂 config:
  - pop(参照, **不重跑** — 配对 = results/population_m1/{inst}_s{seed}_pop.json,
    κ=0, M1 批已跑且确定性验证过)
  - popcx_route = pop + {population_cx_mode: "route", population_cx_rate: 0.2,
    population_cx_inherit_prob: 0.6}
  - popcx_part = pop + {population_cx_mode: "partition", population_cx_rate: 0.2}
  - 全臂 κ=0(κ 档位臂留待判别出方向后的跟进批, 见 §5)。
- runner: experiments/benchmark_population_m2.py, CLI {inst} {seed} {arm},
  输出 results/population_m2/{inst}_s{seed}_{arm}.json = M1 字段全集 +
  population_stats(含 cx_* 键)+ 结构字段(best 解):
  best_mismatch_penalty / best_n_mismatch / best_geo_consistent_frac —
  公式与 experiments/analyze_structure_diff.py analyze() 同款
  (rank = Σ_{d'} [d(c, d') < d(c, actual)]; extra = d_actual − d_near;
  mismatch_penalty = Σ 2×extra; n_mismatch = Σ(rank>0);
  geo_consistent = mean(rank==0); 客户全局号直引 distance_matrix)。

## 3. 跑批(冻结)

- 18 组 = 2 臂 × 3 实例 × 3 seeds; 每 arm 独立 python 进程; xargs -P9;
  OMP/OPENBLAS=1; 日志 results/population_m2/batch_m2_6000iter.log。
- 确定性抽查: pr15 s42 popcx_part 批后同配置重跑 1 组, 逐字段比对。

## 4. 判定规则(死值)

定义: 每实例配对差 d_{X−Y} = mean over 3 seeds (cost_X − cost_Y);
锚散布 s_pop = pop 臂该实例 3-seed std(M1 文件计算)。
1. **J1 判别(核心)**: d_{part−route} < −s_pop → H_B 胜(partition 稳健更优);
   d_{part−route} > +s_pop → H_A 胜; 否则两臂无差(判"重组单位无差别")。
   跨实例稳健: ≥2/3 实例同向且无反向实例 → 宣称; 否则"实例相关"。
2. **J2 安全网(对壳)**: 每实例每臂 d_{arm−pop} > +s_pop → 该臂负效 → 修臂
   verdict(不进入 M3 该臂); < −s_pop → 增益记录。
3. **J3 结构中间指标(报告性)**: 胜臂的 best_mismatch_penalty / n_mismatch /
   geo_consistent vs pop 与 vs pyvrp 锚(~890, 结构差判定文档), verdict 记录;
   若 H_B 胜且错配罚金未向锚靠拢 → verdict 注明"分配修复未走质量路径",
   机制解释留给后续 dump 分析, 不阻塞。
4. **J4 过程指标(报告)**: cx_attempts/children/duplicates/novel_route_keys、
   档案多样性、pool_final_spread、member_refresh_events — 供机制归因,
   不设门槛。
5. 引擎验收: 全量 pytest 绿; 确定性重跑逐字段一致。
判定文档: docs/ma-alns-m2-verdict-20260907.md(Hermes 写)。

## 5. 后续路径(预声明的分支决策)

- H_B 稳健胜 → 结论: 车场分配簇 = MDVRPTW 种群 load-bearing 重组单位(pr 基准
  新机制知识, 独立成文素材); κ 档位跟进批(0.10/0.20, 2 臂 × 3 inst × 3 seeds);
  M3 立项 = 胜臂 + 预算感知教育。
- H_A 胜或无差 → 结论: 路由/车场单位无差别, 多样性/教育为价值载体; M3 照常在
  route 臂基础上立项(教育), 结论写入 M2 verdict。
- 任一臂负效(J2)→ 该臂修复或关闭, M3 只带健康臂。
