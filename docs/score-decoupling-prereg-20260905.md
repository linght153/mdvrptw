# score 与准入解耦探针预注册 (2026-09-05)

衔接: stagnation 修复双向验证 (7358b70) 最终机制修正 — 6000 iter 口径下
base vs admit 差 3.7pp, 真链条 = archive.add 成败 → engine.py:492
`score = 2.0 if added else (1.0 if accepted else 0.0)` → SegmentReward 段权重
→ destroy 选择分布 → 轨迹。base 平台期 add/accepted 常假 → score 0 →
**权重倒置劣化** (常用算子在平台期被逐段压向地板, 冷门算子保留高权重 →
轮盘赌反转偏向历史劣算子); admit 放宽准入 → score 恒 2 → 权重平稳。
修复候选 (未验证) = **score 度量解质量/搜索活动而非是否入档**。

**6000 iter 口径归因干净性** (本探针前提, v2 已证): 该口径下 base 重启
阈值 2000 从未达、深破坏阈值 3000 从不达、cx 关 → admit 与 base 唯一差异
通道 = score 行; 档案内容 (admit 池内更差成员) 无任何消费者 → score 通道
是 admit 收益的唯一候选载体。本探针直接测它。

## 臂 (config 键 score_scheme, 默认 "admission" 逐位不变)

| 臂 | score 语义 | 准入 | 隔离对象 |
|---|---|---|---|
| base (参照) | 2 if added else (1 if accepted else 0) | 硬 (现默认) | — |
| admit (参照) | 同上 (added≈常真) | 放宽 | score 恒 2 + 档案污染 (已混) |
| **act** (新) | 可行且未崩溃 → 2.0, 否则 0.0 | 硬 | **只改 score 通道**: 复刻 admit 的 score 分布, 档案/stagnation/重启语义 = base |
| **qual** (新) | crashed→0; new<best→2.0; new<parent→1.5; accepted→1.0; 拒→0.0 | 硬 | 经典 P&R 改进阶梯: 只奖励质量改进, 平台期无反馈 |

act 的动机: admit 的 score 分布 ≈ "可行探索即反馈" (放宽准入下几乎一切
非重复可行解入档 → score 2); act 用硬准入复刻同一分布 → 若 admit 收益
真走 score 通道, act 必须复现 admit 于全部实例 (含 pr06 的负收益);
若 act ≠ admit → 归因链第三次修正 (通道另有其物)。

qual 的动机: 测"平台期需要活动反馈"叙事 — 若只按质量阶梯给分, 平台期
accepted 稀 → score 仍 0 → 倒置照旧 → 预测 qual ≈ base。

## 假设与判定条款 (6000 iter s42, 差异以相对 base 的 pp 计)

参照值 (既有 results/population_ab/, 不重跑):
pr15 base 2674.8794 / admit 2580.0975 (−3.54pp); pr20 base 3307.9305 /
admit 3185.8899 (−3.69pp); **pr06 (窄窗对照) base 3601.7952 / admit
3672.5057 (+1.96pp admit 负收益)**; 600s pr20 (阈值 100) base 3177.86 /
admit 3253.24 (−2.4pp admit 差)。

- **H-A (score 通道载体实锤)**: pr15 且 pr20 上 act ≤ base − 2.1pp
  (= admit 收益的 ≥60%) → act 复现 admit 收益 → 09-04 归因链确认,
  档案副作用在 6000 iter 无独立贡献。
- **H-A′ (完备性, pr06 控制)**: act ≥ base + 1.0pp (复现 admit 负收益
  ≥50%) → score 恒 2 语义本身是窄窗病因, 通道完备; 若 act ≈ base ±1pp
  → 通道不完备 (负收益另有载体), 归因待修正。
- **H-C (质量阶梯不足)**: 两宽窗上 |qual − base| ≤ 1.0pp → 平台期活动
  反馈必要性叙事支持; qual 任一实例 ≤ base − 2.0pp → H-C 否定。
- **H-B (两全, 600s)**: pr20 600s act ≤ 3177.86 × 1.004 且 < 3253.24
  (= base 水平, 重启多样性保留) → act = admit 收益 + base 重启 → 翻默认
  候选 (score_scheme 默认改 activity, 全量锁存重评)。

## 执行矩阵 (串行独立进程, 时间预算红线)

6000 iter (benchmark_population_ab.py, arm_tag 入名):
pr15 s42 {act, qual}, pr20 s42 {act, qual}, pr06 s42 {act, qual},
pr15 s43 act, pr20 s43 act = 8 runs。
600s (probe_sp_collect.py, stagnation_limit 300 = 阈值 100):
pr20 s42 act = 1 run。
参照不重跑。结果落盘 results/population_ab/{inst}_s{seed}_{arm}.json +
results/sp_pool_pr20_s42_scoreact.json (命名含全部非默认 config 标记)。

## 执行纪律

失败即停止; 判定文档 docs/score-decoupling-verdict-20260905.md;
实现 gated 默认逐位不变 (126 PASS 锁存); 引擎确定性 (同 seed 同 config
逐位复现) 由 v1/v2 验证史背书, act/qual 短预算行为契约测试 + 默认锁存。
