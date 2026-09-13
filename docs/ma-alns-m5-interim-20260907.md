# M5 Phase A interim (2026-09-07) — 压缩安全性判定: 退化, popcxgeC600 取消

协议: docs/ma-alns-m5-prereg-20260907.md §2 判据 A(死值)。
数据: 9 组 6000iter(popcxgeC = rate 0.1 + educate LS 15)vs popcxge(M3 参照)。

## 判定: 压缩退化(判据 A 不通过)

| 实例 | d(C−ge) | s_ref(ge) | 判定 |
|---|---:|---:|---|
| pr15 | +40.3 | 20.7 | **退化** |
| pr16 | +22.4 | 34.8 | 中性 |
| pr20 | +66.1 | 64.6 | **退化** |

→ 按预注册, **popcxgeC600 臂取消**; Phase B 只跑 base600 + popcxge600
(18 组串行, ~3.5-4h)。

## 压缩效果与机制读数

- 墙钟 0.37-0.40×(352-624s vs 882-1669s), 教育次数 ~0.48×(rate 减半生效)。
- 但质量损失 pr15 +1.6% / pr20 +2.1% — **教育 LS 深度(预算 50)是质量的载体,
  教育频率不是**: 同样 6000 迭代下砍深度直接丢质量, 砍频率(rate 0.1 且教育数
  仍 441-521)只省时间。
- 推论(记 M5 verdict 用): 600s 口径下 popcxge600 原参的胜负取决于
  "迭代减半 vs 单迭代质量优势"的净效应 — Phase B 实测; 若原参 600s 也劣, 中间档
  (rate 0.15 / LS 25-30)是候选(J4 预声明), 需先在 6000iter 复验安全性再加跑,
  不自动执行。

## 状态

Phase B 串行脚本 scripts/rerun_m5_600s.sh 启动(后台过夜); 产物
results/population_m5/600s/{inst}_s{seed}_{arm}.json ×18; pyvrp 参照 =
results/pyvrp_wide(既有)。终判 = docs/ma-alns-m5-verdict-20260907.md。
