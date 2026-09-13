# M4 双空间罚域种群 A/B 预注册 (2026-09-07) — 轴2 里程碑 4

实现规格: docs/ma-alns-m4-spec-20260907.md; M3 判定: docs/ma-alns-m3-verdict-20260907.md
(popcxge 已合 main)。判定归 Hermes; 冻结于跑批前; 判定写 docs/ma-alns-m4-verdict-*。

## 1. 问题与假设

单轨迹罚域两族(容量/TW)判"无穿越"是在单轨迹+可行档案语义下; HGS 罚域 =
种群共进化(不可行成员以罚后 fitness 参与)。M4 在 popcxge 形态(壳+partition+
改进门+教育)上开 TW 罚域成员通道, 检验:
- H_M4: **罚域共进化在种群形态提供超 popcxge 的增益** — 分配/链组织盆地间存在
  需要"不可行走廊+教育修复"的连接(与单轨迹 softtw"无穿越"结论的对立面:
  共进化维持不可行多样性, 走廊由教育关闭而非轨迹穿越)。
- 附带: 罚域臂质量与 λ 量纲的关系(λ 静态, mult 臂扫) — 若高 λ 臂 ≈ popcxge
  (退化为全可行), 低/中 λ 臂无增益 → 罚域在 MDVRPTW 种群形态也无价值
  → FIS 形态格真·全满, 机制格收束叙事完整(论文负面矩阵终章)。

## 2. 口径(冻结)

- 实例 × seed: pr15 / pr16 / pr20 × s42-44。6000 iter。不跑 600s(M5)。
- 配置: BASE 逐键 = M1-M3; 全臂 κ=0, partition + gate=improve + educate
  (popcxge 底形):
  - 参照(不重跑): popcxge = population_m3/{inst}_s{seed}_popcxge.json;
    pop = population_m1 文件(背景列)。
  - popcxgep_l1 = popcxge + {population_penalty_mode: "tw",
    population_penalty_mult: 1.0}
  - popcxgep_l10 = 同上 mult 10.0(高罚: 预期退化逼近 popcxge — λ 灵敏度锚)
- runner: experiments/benchmark_population_m4.py CLI {inst} {seed}
  {popcxgep_l1|popcxgep_l10}, 输出 population_m4/ 目录, 字段 = M3 全集 +
  population_stats(含 penalty_*)。
- 规模: 2 臂 × 3 实例 × 3 seeds = 18 组; xargs -P9; 确定性抽查 pr15 s42
  popcxgep_l1 批后重跑逐字段比对。

## 3. 判定规则(死值)

定义同前: 每实例 d_{X−Y} = mean(3 seeds); s_ref = popcxge 臂该实例 3-seed std
(本里程碑锚臂散布)。
1. **J1(罚域价值)**: 每臂每实例 d_{arm−popcxge} < −s_ref → 罚域增益; 稳健规则 =
   ≥2/3 实例增益且零反向(d > +s_ref 计反向)。
2. **J2(λ 灵敏度)**: l1 vs l10 配对: 报告 3 实例 d_{l1−l10}; 若 |d| 全 ≤ s_ref →
   λ 不敏感(记录, 不设门槛); 若 l10 全面劣 → 与"高 λ 退化"预期一致(机制自洽
   证据)。
3. **J3(池健康)**: penalty_infeasible_final / penalty_member_iters 报告 —
   罚域通道真的在使用(非空转): 若 l1 臂 penalty_member_iters ≈ 0 且质量无差 →
   通道空转 verdict(罚域在种群形态也不构成可达通道); 若 >0 且质量增益 → 通道
   活跃且有效; >0 无增益 → 通道活跃但无价值(记录)。
4. **J4(引擎)**: pytest 绿 + 确定性重跑一致 + 档案纯净(18 组 archive 全可行)。
5. **结论分支**: J1 稳健增益 → M4 通过, M5(600s)带罚域臂; J1 无增益(通道空转
   或活跃无价值)→ 罚域形态格关闭, M5 只测 popcxge(600s 口径), M4 verdict 记为
   机制格终章。
判定文档: docs/ma-alns-m4-verdict-20260907.md(Hermes 写)。

## 4. 后续(预声明)

- 无论 M4 结果, M5(600s 口径 + 教育墙钟压缩)按 M3 verdict 技术债推进;
- M5 口径: 600s base 配置(archive 20 / LS 10-5 / stagnation 300, 参照
  scripts/probe_connectivity_dump.py CONFIG), 串行独立进程(时间预算红线),
  实例 pr15/16/20 × s42-44, 臂 = base-600s(重跑同口径)/ popcxge 压缩版 /
  (罚域若通过) popcxgep; pyvrp-600s 参照 = 既有 pyvrp_wide 数据。
