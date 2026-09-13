# M5 600s 平台表 + 教育墙钟压缩 — 预注册 (2026-09-07) — 轴2 里程碑 5

背景: M3 verdict(popcxge 已合 main 80a32c8, 6000iter 全实例超壳) + M4 verdict
(罚域关闭, M5 只测 popcxge)。**M5 无引擎代码改动** — 压缩只动 config 值
(population_cx_rate / population_educate_ls_budget), 全部机制已在 main。
判定归 Hermes; 冻结于跑批前; 判定写 docs/ma-alns-m5-verdict-*。

## 1. 问题与目标

1. **教育墙钟压缩验证**: popcxge 6000iter 墙钟 982-1669s(教育 ~0.3-0.8s/次 ×
   rate 0.2)。600s 时间预算下 popcxge 仅 ~3000-4000 迭代(base 600s 为
   6000-7000)→ 教育成本必须压: popcxgeC = rate 0.2→0.1 + educate LS 预算
   50→15(教育次数减半 × 单次成本 ~1/3 ≈ 总教育成本 ~1/6)。
2. **600s 平台表**(pyvrp 对照口径, 论文用): base-600s(自产) / popcxge-600s /
   popcxgeC-600s vs pyvrp_wide 既有 600s 数据(pr15/16/20 × s42-44, float_cost
   口径) — 检验形态优势是否跨口径保持, 给出论文的时间预算对照表。
3. 口径红线: 时间预算实验**串行 + 独立进程**(并行污染时间预算结果, CLAUDE.md);
   600s base 配置 = scripts/probe_connectivity_dump.py CONFIG 同款
   (max_time_seconds=600, stagnation_limit=300, archive_capacity=20,
   local_search_max_iter=10/freq=5/on_accept=True/final=False, segment/reaction/
   decay/min_selection_prob 同 BASE) — 与既有 base600s_routes 数据同口径。

## 2. Phase A: 压缩安全性(6000iter 固定迭代, 并行安全)

- 臂: popcxgeC = popcxge 键 + {population_cx_rate: 0.1,
  population_educate_ls_budget: 15}; 参照 = population_m3 popcxge(不重跑)。
- 9 组(3 实例 × 3 seeds, 6000iter, xargs -P9, ~30-40 min)。
- runner: experiments/benchmark_population_m5.py CLI {inst} {seed} popcxgeC
  (mode=6000 默认), 输出 results/population_m5/。
- **判据 A(死值)**: 每实例 d(C−ge) > +s_ref(ge 3-seed std)→ 压缩退化, Phase B
  的 popcxgeC600 臂取消(只跑 base600/popcxge600); 无退化(中性或优)即放行。

## 3. Phase B: 600s 平台表(串行, ~5-7h, 后台过夜)

- 臂(全部 600s 时间预算, 独立进程串行):
  - base600 = 600s CONFIG(重跑 9 组, 与既有 base600s_routes 同口径可交叉核对)
  - popcxge600 = 600s CONFIG + popcxge 键(教育原参)
  - popcxgeC600 = 600s CONFIG + popcxgeC 键(压缩参; 仅 Phase A 放行时跑)
- 27 组 = 3 臂 × 3 实例 × 3 seeds; 串行 for 循环(禁并行); 每进程独占;
  bash 编排脚本 scripts/rerun_m5_600s.sh; 增量写 results/population_m5/600s/
  {inst}_s{seed}_{arm}.json(时间预算实验波动 1-2%, 记录 wall_s 与迭代数)。
- pyvrp 参照: results/pyvrp_wide/{inst}_s{seed}_600s.json 的 float_cost
  (pr16 用 int_cost 或 float 不可行注记 — 与 pyvrp-wide verdict 同口径处理)。
- 确定性: 时间预算口径不可逐位复现(已知红线), 不设确定性抽查; 每组记录
  iterations 实际数供归因。

## 4. 判定规则(死值)

1. **J1(压缩有效性, Phase A)**: 如上 §2 判据; 另报告 popcxgeC 的
   educate_attempts 与墙钟下降比例(预期教育成本 ~1/6, 整跑 ~1.5-2× base 量级)。
2. **J2(跨口径形态保持, Phase B 主判)**: 每实例配对差 d(popcxgeC600 − base600)
   与 d(popcxge600 − base600), 锚散布 s = base600 3-seed std:
   < −s → 形态 600s 增益; 稳健规则 = ≥2/3 实例增益且零反向 → **形态优势跨口径
   保持**; 无差但方向全负 → 记录"方向性, 幅度不足"(散布大时常见, 如实报告)。
3. **J3(pyvrp 平台表)**: 报告 3 臂 × 3 实例均值 gap vs pyvrp float_cost(3-seed
   均值)与 vs BKS(2433.15/…, 见 mdvrptw-bks-2026); 论文口径 = 平台表 + 差距
   收窄叙述(6-11pp → 目标 <6pp), 不宣称超越(引擎语言/实现效率差异如实注明:
   pyvrp = C++ 核)。
4. **J4(压缩 vs 原参 @600s)**: d(popcxgeC600 − popcxge600) 报告: C ≥ 原参 →
   压缩在时间预算下必要且有效; C < 原参且幅度 > s → 压缩过度, 记录中间档
   (rate 0.15 / LS 25)为后续选项(不自动加跑)。
5. **合并决策(J5' 类比)**: M5 分支(feat/axis2-m4m5)内容 = M4 负结论机制
   (gated 存档, 铁律 5 保留) + M5 runners/docs + 无引擎默认路径改动 → 行为中性。
   合并条件 = Phase B 完成且 J2 无"形态 600s 全面负效"(若 popcxge600 与 C600
   均 > +s 于 ≥2/3 实例 → 形态时间预算口径失效, verdict 如实记录, 分支留档
   暂不合, 论文按固定迭代口径定位); 否则 ff 合回 main。M4/M5 verdict 链
   全量保留。

## 5. 判定产物

docs/ma-alns-m5-verdict-20260907.md: Phase A/B 数据表、J1-J4 判定、600s 平台表
(自产 3 臂 vs pyvrp vs BKS, 论文 §实验 直接引用)、合并执行记录。
