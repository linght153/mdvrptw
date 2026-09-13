# score 解耦探针中间分析 (2026-09-05, 等矩阵跑批中, 非最终判定)

预注册: docs/score-decoupling-prereg-20260905.md。实现: engine score_scheme
(admission/activity/quality, 默认 admission 逐位锁存) + probe_stats 观察计数
+ on_archive_add 回调 (全部行为中性, base/admit pr15 s42 重跑逐位复现历史
2674.8794/2580.0975 ✓)。

## 已确立事实 (pr15 s42 6000 iter, 确定性引擎 → 臂间差非采样噪声)

| 臂 | best | adds | accepts | equal_acc | stagn_peak | 终权重 |
|---|---|---|---|---|---|---|
| base | 2674.88 | 299 | 548 | 167 | 287 | 倒置: [0.12, 0.01×6, 0.08] 地板+3幸存者 |
| admit | **2580.10** | **5777** | 443 | 79 | 2 | 平坦 ~1.9-2.0 |
| act | 2664.99 | 311 | 446 | 81 | 195 | **平坦 ~1.9-2.0** |

- act (score = 可行即 2, 准入 = base) 复刻了 admit 的**权重形态** (平坦 2)
  却没复现 admit 收益 (2664.99 vs 2580.10, 差 85 = 3.3pp) → **H-A 首点即败**:
  score→权重→destroy 混合通道不是 admit 收益载体 (若 09-04 \"selector-准入
  耦合\" 归因成立, act 必须 ≈ admit)。
- base 终权重倒置 (地板 + random/worst_emission/repair 幸存) vs act 平坦 —
  两者结果几乎相同 (2674.88 vs 2664.99) → **权重形态本身 (倒置 vs 健康) 也
  不是结果的决定量**。
- 改进事件轨迹: 三臂改进算子构成相似 (random+worst_emission 主导), 但 admit
  在相同成本区间的单步下探幅度更大 (−10.6/−6.1 vs base/act 的 −0.4~−5.4),
  admit 末次改进 @3710 (比 base @4015 更早), 优势来自\"同算子下更大的结构
  跳跃\"而非\"改进持续更久\"。
- 机械归约 (代码审查): 6000 iter 口径下 cx 关、重启/深破坏阈值不可达、
  SearchContext 无消费者、crowding 父代选择在单目标退化 = 恒取档案最优
  (entries[0])、档案内容 (top-20 不同成本) 在 base/admit 间集合恒等 →
  臂间唯一机械差异 = score 行本身。act ≈ base 且 admit 差异巨大 ⇒
  要么 admit 收益经 rng 混沌放大一条 ~4% 迭代的 score 细节差异 (待 s43
  检验), 要么存在未发现的档案侧消费者 (待查)。

## 待矩阵数据 (后台串行中)

pr20/pr06 × {act, qual, admit} + pr15 qual + pr15/pr20 s43 × {act, admit}
+ 600s pr20 act。判定点:
1. s43: admit 的 −3.5pp 是否复现 → 若否 = 单 seed 混沌假象, 09-04
   \"6000 iter admit 免费午餐\" 需重新定性。
2. pr06 (窄窗): admit 已知 +1.96pp 负 — act 是否 ≈ base (不复制负收益)?
3. qual: 与 base 差异 ≤1pp 与否。
4. 600s act: 重启保留 (H-B)。
