# 09-04 单 seed 臂间结论 3-seed 复核计划 (2026-09-05)

背景: score 解耦探针发现 6000 iter 口径下单 seed A/B 不可判定 (base 同臂
跨 seed 散布 2.3-3.5%)。09-04 全天档案侧臂间结论 (div/cx/admit/A2/cur/
d1/d2) 多数单 seed (s42) 判定 → 本复核以 3-seed 配对口径重估。

## 已有数据 (不重跑)

- pr15/pr16/pr20 × s42-44 × {base, div, cx} — 完整 3-seed
- pr15/pr20 × s42/s43 × admit
- 6000 iter 全臂单点统计 (probe_stats/weights) 存 results/population_ab/

## 补跑 (8 炮 6000 iter, 串行独立进程)

| run | 目的 |
|---|---|
| pr15 s44 admit, pr20 s44 admit | admit 宽窗 3-seed 配对补齐 |
| pr06 s43/s44 × {base, admit, div} | 窄窗 3-seed (昨天 pr06 仅 s42) |

## 判定条款 (预注册)

1. **admit 宽窗** (pr15/pr20, 3-seed 配对差): 符号一致且均值 ≤ −2pp →
   稳健效应 (恢复); 符号不一致或 |均值| ≤ 1.5pp → 无稳健效应 →
   09-04 "admit 6000iter 免费午餐/div 收益载体=admit" 作废确认。
2. **div 宽窗** (pr15/pr16/pr20, 已有 3-seed): 同上口径。
3. **cx**: 昨天 H5 否定 (无收益) → 3-seed 复核若仍无稳健正效应 = 结论
   存活但理由修正 (无基线改善前提下交叉无增量)。
4. **pr06 窄窗 div/admit**: 3-seed 全部 ≥ +0.5pp → "窄窗扰动负" 稳健;
   否则无效应 (s42 负为单点)。
5. **cur/d1/d2**: 6000 iter 逐位 = base 为机制性结论 (不涉 seed 散布),
   存活; 600s 单点略差 → 标注存疑, 不影响决策 (无后续候选依赖)。
6. **A2 fitness 裁剪**: 关闭结论存活 (连 div 都无稳健效应 → 裁剪无独立
   空间), 机制叙述 (div 收益 = admit 准入) 作废。
7. 判定落盘: docs/legacy-ab-reeval-verdict-20260905.md。

## 执行纪律

6000 iter 固定迭代; 串行; 结果文件命名含臂名 (既有规则); 全程不读
s43/s44 以外的新 seed 数据直到跑完 (无 holdout 需求, 纯复核)。
