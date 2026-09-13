# stagnation 修复双向验证判定: v1/v2 双否定 + 真机制修正 (2026-09-04)

预注册: docs/stagnation-fix-prereg-20260904.md。实现 (gated, 默认逐位不变):
stagnation_source (v1: archive/current) + stagnation_recovery_divisor (v2,
默认 3 = 旧 //3)。测试 +2 → 126 PASS。

## 结果

| run | 口径 | 配置 | 结果 | vs 参照 |
|---|---|---|---|---|
| cur | 6000 iter pr15 | current 源 | 2674.8794 | = base 逐位 |
| cur | 6000 iter pr20 | current 源 | 3307.9305 | = base 逐位 |
| cur | 600s pr20 | current 源 | 3195.12 | base 3177.86 (+0.54% 差) |
| d1 | 6000 iter pr20 | 阈值 6000 (永不重启) | 3307.9305 | = base 逐位 |
| d2 | 6000 iter pr20 | 阈值 3000 | 3307.9305 | = base 逐位 |
| d2 | 600s pr20 | 阈值 150 | 3189.23 | base 3177.86 (+0.36% 差) |

参照: 6000 iter base 3307.93 / admit 3185.89; 600s base 3177.86 / admit 3253.24。

## 判定: 停滞/重启修复方向双否定

1. **v1 (current 源) 无效**: 6000 iter 两实例逐位 = base → 无温度接受下
   "accepted∧改进" ⇔ "added" (拒收迭代 = 未接受迭代), 两源计数序列恒等。
   600s 略差 (重启时机微移)。
2. **v2 (恢复阈值) 无效**: d1/d2 在 6000 iter 逐位 = base → **6000 iter 下
   stagnation 从未达任何阈值, 重启根本没发生**。d2 600s 略差 0.36% (重启
   减少略损多样性)。

## 真机制修正 (A2/判别判定的第二次修正)

6000 iter 下 base (3307.93) vs admit (3185.89) 差 3.7pp **与重启无关** (双方
零重启)。真机制链: **archive.add 成败 → selector score (engine:469,
score=2 if added) → 段权重更新 → destroy 选择分布 → 轨迹**。admit 放宽准入
→ added≈常真 → score 恒 2 → 权重平稳健康; base 平台期 added 常假 + accepted
常假 → score 0 → 权重震荡劣化 → 轨迹差。600s 下差异主轴 = 重启多样性
(base 频繁重启多轨迹好; admit 单轨迹卡 3253) — 两口径机制不同, 均与
"停滞语义"无关。

## 立即结论与方向

1. **停滞/重启修复方向关闭** (v1/v2 实证否定; 6000 iter 无重启事实)。
2. **6000 iter 口径的 admit 收益 = selector-准入耦合的副作用**: 修复候选 =
   score 与准入解耦 (score 度量解质量而非入档) — 未实现未验证, 属新探针;
   现成 workaround = 6000 iter 口径实验统一 config archive_admit_relaxed=True
   (不加多样性裁剪 — div/a2 已证裁剪无贡献)。
3. **600s 口径维持 base** (重启多样性必要)。
4. population_ab 30-run (6000 iter base/div/cx) 解读: base 臂 = score 震荡
   劣势态; div/cx 臂 = admit 语义的宽松准入 → 其收益 = selector 健康化,
   与多样性/交叉无关 (cx pr20 −1.63pp 需同框架重释)。

## 数据事故记录

probe_sp_collect 文件命名原不含 divisor → d2 600s 结果覆盖
sp_pool_pr20_s42.json (base 3177.86 → 3189.23)。已修命名 (divisor 入 tag)
+ 后台重跑恢复 base 文件。教育列池/曲线探针复用该文件 — 恢复完成前不得
重跑依赖脚本。
