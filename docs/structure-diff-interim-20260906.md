# 结构性差异分析中间记录 (2026-09-06, s42 先行; 3-seed 全量待补跑)

衔接: docs/pyvrp-wide-control-verdict-20260905.md 待办 1。预注册:
docs/structure-diff-prereg-20260906.md。工具:
scripts/dump_base600s_routes.py (自产 base 600s dump, 3-seed 补跑中) +
experiments/analyze_structure_diff.py (五组结构指标 + 成本加性分解)。

## 数据口径

- pyvrp 600s dump: results/pyvrp_wide/{name}_s{seed}_600s.json (float 口径)
- 自产 base 600s: s42 = results/connectivity_{name}_s42_routes.json (既有,
  与 baseline_pr_120s/*_600s.json 逐位一致); s43/s44 = base600s_routes/
  (补跑中)。两侧 seq 均为全局节点号可直接对齐。
- 前提验证: Σ闭环距离 == dump best_cost 精确吻合 (pr20: 3177.86 vs
  24×132.41; pr15: 2526.17 vs 20×126.31) → 成本差可 100% 归因到弧段结构。

## s42/s43 先行结果 (pr15×2seed, pr20×1seed; 仅线索不定论)

### 成本加性分解 (参照 = 每客户独立往返 2×d_actual)

pr15 s42: Δ+102.5。twoway ours 13490.7 < pyvrp 13507.2 (−16.5, 自产分配
略优); chain_save 0.8127 vs 0.8206 → **Δ 几乎全部来自成链质量差** (~108)。
错配罚金 617.3 vs 633.8 (−16.5)。

pr15 s43: Δ+136.2。twoway 13802.5 vs 13464.4 (+338.1 = 罚金 929.2 vs
591.1); chain_save 0.8146 vs 0.8200 (≈73)。分配差贡献 ~63 + 链差 ~73 各半。

→ **pr15 成链差 0.54-0.79pp 跨 2 seed 同号 (pyvrp 系统性更优)**;
分配差 pr15 内部不稳 (s42 −16.5 / s43 +338.1), 待 s44。

pr20 s42: Δ+202.8。twoway 16738.9 vs 16185.6 (+553.3 = 错配罚金 1447.1 vs
893.8 之差); chain_save 0.8102 vs 0.8162 (Δ0.6pp ≈ 98)。
→ **Δ = 分配差贡献 ~105 + 成链差贡献 ~98, 各半**。分配差=自产挂非最近
车场 (66 vs 50 错配客户, mean_extra 10.96 vs 8.94); 成链差=两实例同号
(pyvrp chain_save 均高 0.6-0.8pp)。

### 分配差细查 (pr20 s42)

- 61/288 客户 (21%) 两方法分配不同; 主簇 = depot1↔depot5 之间 26 客户
  (两车场相距仅 7.2 = 孪生车场), depot3↔4 (35.4) 等近距车场对。
- 错配集合: ours-only 34 / both 32 / pyvrp-only 18。自产独有错配几乎全为
  "closer-to-nearest 且 pyvrp 挂最近" 的**纯错配** (近车场被无视, 无顺路/
  容量理由; 例 cust 61 距 depot0 4.2 却挂 depot1, cust 180/201/75 距
  depot3 14-16 却挂 depot1)。pyvrp 的错配更便宜 (罚金 894 vs 1447)。
- 错配客户需求/时间窗无规律 (demand 1-25, window 186-354 宽) → 不是
  TW 紧/大需求驱动的结构性错配, 疑似 repair 跨车场插座的路径依赖盲插。
- 微观 (cust 61): 自产把它捎进 depot1 一条横跨 depot0 门口的 13 客户
  越界路由 (该路由多客户距 depot0 仅 4-16); pyvrp 同客户在 depot0 近区
  路由 (max d0 29)。自产 depot0 4 条路由呈混带 (r0 近/r1-r2 中远混/r3
  远征), pyvrp 近区路由更纯 + 远征路由专扫远客户 (两级分化更清晰)。

### 宏观形态

- 车辆数全满且相等 (pr15 20/20, pr20 24/24) → 车辆数非差异源。
- route_len_std / route_util_std: pyvrp 更高 (pr15 2.26 vs 1.45;
  pr20 2.06 vs 1.61; util_std 0.130 vs 0.067 / 0.097 vs 0.052) → pyvrp
  路由长短/负载更异质, 自产被抹平 (每车几乎等满)。
- purity / geo_consistent: pyvrp 更高 (pr20: 0.831/0.826 vs 0.774/0.771)。
- tw_inversion: pr15 ours 0.377 vs pyvrp 0.349 (自产 TW 排序更乱 +2.9pp);
  pr20 反号 −2.2pp → 符号不稳待 3-seed。
- 共享骨架: pr15 15 对 ≥4 长共享段覆盖 112/240 客户 (47%);
  pr20 10 对/81/288 (28%) → 两方法收敛到显著公共主干, 差异在段边界与
  路由成员组成。

## 待办

1. 7 组自产 dump 完成后跑全量 3-seed 配对表 → 差异方向跨 seed 确认。
2. verdict: 判定哪些形态维度稳健差异 → 映射"形态跃迁组件优先级"
   (车场分配重平衡机制 vs 链/顺序教育机制)。
