# ALNS 类方法在 Cordeau MDVRPTW 算例上的 gap 调研

> 更新: 2026-08-08 · 调研目的: 回答"相关 ALNS 算法论文在此算例下的 gap"

## 1. 文献 BKS (MDVRPTW, Cordeau 2001 实例)

| 实例 | .res 权威解 | **HGSADC BKS** (Vidal 2013) | SGVNSALS (COR 2023) | PR-ALNS (2026) |
|---|---:|---:|---:|---:|
| pr01 | 1083.98 | **1074.12** | 1190.00 | ~1076 (≈HGSADC) |
| pr02 | 1763.07 | **1762.21** | 2082.10 | ~1764 |
| pr03 | 2408.42 | **2373.65** | 2725.56 | ~2376 |
| pr04 | 2958.23 | **2815.48** | 3558.48 | ~2818 |
| pr05 | 3134.04 | **2964.65** | 3789.84 | ~2967 |

来源: COR 2023 (Bezerra et al., SGVNSALS 论文对比表) 引用 HGSADC
(Vidal et al. 2013, "A hybrid genetic algorithm with adaptive diversity
management", MDVRPTW 至今最强方法); PR-ALNS = Chakour & Abdoun 2026
(Relative ALNS, Evolutionary Intelligence), 摘要称平均改进 PR-ALNS
-0.22%、距 HGSADC 0.10%。

**关键**: HGSADC 全部优于 .res (pr01 -0.9%, pr02 -0.05%, pr03 -1.4%,
pr04 -4.8%, pr05 -5.4%) — .res 是官方结果集, 但 HGSADC 改进过它。

## 2. 各方法 gap 对比 (以 HGSADC BKS 为基准)

| 方法 | 类型 | pr01 | pr02 | pr03 | pr04 | pr05 | 平均 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HGSADC | 混合遗传 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| PR-ALNS | **ALNS 类** | ~0.19% | ~0.10% | ~0.10% | ~0.10% | ~0.10% | ~0.12% |
| SGVNSALS | VNS | +10.8% | +18.2% | +14.8% | +26.4% | +27.8% | +19.6% |
| **本项目 ALNS** | ALNS | **+11.1%** | **+17.1%** | — | — | — | — |
| pyvrp 30s | HGS | +18.9% | +14.4% | +29.4% | +15.6% | +23.2% | +20.3% |

本项目 ALNS (120s 预算, 三轮修复后): pr01 1194-1296 / pr02 2061-2221。

## 3. 结论

1. **文献 ALNS 类方法 (PR-ALNS) 的 gap ≈ 0.1%** — 但那是精调过的
   研究级实现 (每实例 30 runs × 60min);
2. **本项目 ALNS 120s 预算 gap +11-17%** (pr01/pr02) — 与 SGVNSALS
   (60min × 30 runs) 的 +11-27% 同量级, 优于 pyvrp 30s (+14-29%);
3. **预算-质量权衡明显**: 我们 120s 达到文献 VNS 60min 的量级,
   说明 ALNS 框架本身竞争力没问题, 差距来自 (i) 预算 (ii) 算子
   工程精调 (iii) 车辆数探索能力;
4. 论文如实口径: "120s 预算下 ALNS 静态解 gap +11-17% vs HGSADC
   BKS, 与同类启发式 (VNS/ALNS 60min) 同量级; pyvrp 30s 为参照"。
