# 方向①车场分配重平衡 A/B 预注册 (2026-09-06)

衔接: docs/structure-diff-verdict-20260906.md 判定 1/2 (pr20 错配罚金
1447/1490/2865 vs pyvrp ~890; 自产错配 = 纯错配 + seed 漂移) → 组件
优先级 ①。实现规格: docs/engine-upgrade-specs-20260906.md (CC 实现,
gated 默认关)。本文件只定义实验与判定, 不定义实现。

## 假设

H-R: 显式车场分配重平衡算子 (错配优先移除 + 周期重分派) 在固定迭代
口径下: (i) 机制生效 — 轨迹末端解的错配罚金/geo_consistent 优于 base;
(ii) 质量转化 — 6000iter best 成本低于 base, 配对差跨 seed 稳健。

## 两级判定协议 (硬屏障, 不事后调门槛)

**第 1 级 机制生效 (过程指标, 先于质量判定)**:
- 指标: 末端档案 best 解的错配罚金 Σ2×(d_实际 − d_最近) 与
  geo_consistent_frac (dump 解后由 analyze_structure_diff 的
  assignment 指标族计算, 只读)。
- 支持 = pr20 且 pr15 的 arm 罚金配对均值 ≤ base −10% 且符号一致
  (removal/redispatch/both 任一); 否定 = 罚金不降 → 机制未生效,
  探针关闭, 不进入第 2 级 (不与 09-05 结构生成三探针混淆: 那是
  生成类, 这是分配修复类, 过程指标可测)。
- 注意: 6000iter base 本身错配罚金小 (pr20 s42 6000iter 数字待 dump
  时一并算) — 若 base 6000iter 已几乎无错配, 第 1 级自动否定, 说明
  错配是 600s 长预算/重启下的现象, 转向 600s 口径复验 (见下)。

**第 2 级 质量转化 (结果指标)**:
- 主口径: 6000iter × 3 seeds (42-44) × pr15/pr20 × arm
  {removal, redispatch, both}, base 参照 = 既有 population_ab/
  {inst}_s{seed}_base.json (固定迭代引擎确定性, 同 seed 同值, 不重跑)。
- 支持 = 某 arm 在 pr15 且 pr20 配对均值差 < 0 (arm 更优) 且
  |均值差| > 同臂 seed 散布 (max-min); 符号混或幅度不足 = 否定。
- 若第 2 级否定但第 1 级生效: 判 "分配修复无质量转化" (与 E 机制
  教育列同构的结论), 记录并转向 ② 或收束。

**扩展 (非判定, 仅记录)**: 600s × pr20 × 3 seeds × 生效 arm 串行
复验 (~30 min/arm), 观察 s44 型分配崩坏 (83 错配) 是否被机制吸收 —
600s 口径数字受重启影响, 不作主判定。

## 执行

1. CC 实现完成 + 全量 pytest 后, Hermes 侧启动跑批: 串行独立进程
   bash 循环, 每组 python 独立进程 (6000iter pr15 ~200s / pr20 ~230s,
   12 组 ~45 min; both/removal/redispatch 若都跑 18 组 ~70 min)。
2. dump 各 arm 末端 best 解 routes → analyze_structure_diff 输出
   错配罚金/geo_consistent (第 1 级)。
3. 第 2 级用结果 JSON 对比 base。
4. 判定文档 docs/rebalance-ab-verdict-20260906.md。
