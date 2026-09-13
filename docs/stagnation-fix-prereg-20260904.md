# stagnation 修复双向验证预注册 (2026-09-04)

衔接: A2 判定 + 600s admit 判别发现 archive.add 拒收耦合 stagnation_count —
满档拒收误计停滞 → 误重启; 6000 iter 口径重启吃预算 (base 差), 600s 口径
重启 = 有效多样性 (base 好)。修复候选: stagnation 计数改 **current 质量停滞**
(config stagnation_source="current", 默认 "archive" 逐位不变 126 PASS):
accepted 且 obj 严格改进 (vs 接受前) 才清零, 拒收/平台/等值接受均累计。
预期: 6000 iter 拿 admit 收益 (改进期不误重启) + 600s 保留重启多样性
(平台期 current 停滞 → 仍按阈值重启)。

## 假设

H-FIX: current 源在 6000 iter ≤ base − 2pp (≈ admit 水平或更好) 且
600s ≤ base + 0.5pp (不损重启多样性) → 两全成立 → 翻默认 + 全量重评。

## 设计 (串行独立进程)

| run | 口径 | 实例 seed | 臂 |
|---|---|---|---|
| 1 | 6000 iter s42 | pr15 | cur |
| 2 | 6000 iter s42 | pr20 | cur |
| 3 | 600s s42 | pr20 | cur |

参照 (既有不重跑): 6000 iter base/admit (pr15 2674.88/2580.10; pr20
3307.93/3185.89); 600s base 3177.86 / admit 3253.24 (pr20)。

## 判定

- **两全成立**: run1 ≤ 2620 (pr15 base −2pp) 且 run2 ≤ 3240 (pr20 base −2pp)
  且 run3 ≤ 3193 (pr20 600s base +0.5pp) → 翻默认, 锁存/文档/技能全量更新。
- **部分**: 6000 iter 达标但 600s 损 >0.5pp → current 源不可作默认, 保留
  gated (6000 iter 口径实验可用)。
- **否定**: 6000 iter 不达标 → 修复假设错 (停滞非重启主因), 归档。

## 执行纪律

600s run 后台串行 (避免并行污染); 结果落盘 population_ab/{inst}_s42_cur.json
+ sp_pool_pr20_s42_cur.json; 判定文档 docs/stagnation-fix-verdict-20260904.md。
