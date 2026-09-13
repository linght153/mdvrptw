# M6 中间档教育压缩 A/B 预注册 (2026-09-08) — M5 J4 预声明候选兑现

背景: M5 Phase A 判否激进压缩(rate 0.1/LS 15, 6000iter 退化: 教育 LS 深度 =
质量载体); M5 verdict §5 剩余① = 中间档 (rate 0.15 / LS 25-30) 作为 600s 幅度
兑现路径, 且按 J4 预声明"需先在 6000iter 复验安全性再加跑"。**M6 无引擎代码
改动**(纯 config), 只加 runner 臂。

## 1. 假设

- H_M6a: 中间档(rate 0.15 + educate LS 25)在 6000iter 不显著退化
  (d(m−ge) ≤ +s_ref 每实例)→ 压缩存在安全操作点。
- H_M6b: 600s 口径下中间档迭代数 > popcxge600(单次教育成本 ~1/2-2/3)且
  单迭代质量损失 < 迭代增量收益 → popcxgem600 质量 ≥ popcxge600 且 ≥ base600
  (600s 幅度兑现: M5 的方向性优势从"幅度不足"变"幅度足够")。

## 2. Phase A: 6000iter 安全性(并行 P9, ~30 min)

- 臂: popcxgem = popcxge 键 + {population_cx_rate: 0.15,
  population_educate_ls_budget: 25}; 参照 = population_m3 popcxge(不重跑)。
- 9 组 (pr15/16/20 × s42-44); runner = benchmark_population_m5.py 加 popcxgem
  臂(experiments/ 小改, 提交)。
- **判据 A(死值)**: 每实例 d(m−ge) > +s_ref(ge 3-seed std)→ 退化 → M6 关闭,
  中间档不成立; 无退化(中性或优)即放行 Phase B。
- 记录: 墙钟比、educate_attempts 比(预期 ~0.7×)、质量方向。

## 3. Phase B: 600s 平台表补充(串行 ~2h, 仅 9 新组)

- popcxgem600 = 600s CONFIG + popcxgem; 参照(不重跑) = M5 Phase B 已产
  results/population_m5/600s/{inst}_s{seed}_{base|popcxge}.json。
- 串行 for(时间预算红线); 脚本 scripts/rerun_m6_600s.sh(镜像 m5 脚本, 9 组)。
- **判据 B(死值)**: 每实例 d(popcxgem600 − base600) 与 d(popcxgem600 −
  popcxge600): 锚散布 s = base600 3-seed std。
  - 若 ≥2/3 实例 popcxgem600 < popcxge600 且 ≤ 0 于全部(零反向) → **中间档
    600s 兑现**: 合并候选 (J5' 类比: 代码 gated 行为中性 + 判定通过)。
  - 若与 popcxge600 无差(幅度 < s)但 ≥ base600 方向全负 → 记录"中间档 ≈ 原参,
    无兑现", 合并仍可(无负效); 关闭压缩线。
  - 若 popcxgem600 > popcxge600 于 ≥2/3 且幅度 > s → 压缩方向负, 关闭。
- 迭代数与教育统计照记(归因)。

## 4. 产物

docs/ma-alns-m6-verdict-20260908.md(Phase A/B 表 + 判定 + 合并执行)。
