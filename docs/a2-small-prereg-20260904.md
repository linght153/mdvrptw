# A2 小规模实验预注册: 多样性贡献进 fitness (κ 扫描) (2026-09-04)

衔接: 种群 A/B div 臂 (裁剪侧阈值, −0.70pp 主实例均值) 部分支持 → A2 =
HGS 式 **fitness = cost − κ×best_cost×dc** 连续权衡裁剪 (dc = 解到档案其余
解的平均路由 Jaccard 距离), 替换 div 的硬阈值 (阈值丢"成本好但相似度
>0.2"的解; κ 给它们按多样性税定价的机会)。实现: archive fitness_dc/
fitness_kappa (默认关, 与 div 同享准入放宽 — 单变量 = 裁剪策略), 引擎
config 键 archive_fitness_dc / archive_fitness_kappa。

## 假设

H-A2: 存在 κ 使 fitness 裁剪的 6000 iter 质量 ≤ base − 0.5pp 且 ≤ div − 0.3pp
(赢过已有最佳臂) → 方向可行, 值得扩 3 seed + 600s 复验。
若赢 base 但输 div → fitness 连续权衡不敌阈值 (κ 尺度问题或 dc 度量维度错);
若全 κ 与 base 打平/更差 → 裁剪侧多样性已到极限, A2 关闭。

## 设计 (小规模, 1 seed 扫 κ)

- 臂: a2 × κ ∈ {0.02, 0.05, 0.10}; 实例: pr15, pr20 (宽窗紧车队主战场) +
  pr06 (窄窗对照 — div 在此 +1.5pp 负贡献, 检查 A2 是否同样负 → 条件化必要
  性证据)。
- 固定 6000 iter, s42, 单进程串行 (与 population_ab 同口径)。
- 参照 (不重跑, 复用 population_ab 既有结果): base/div 的 pr15/pr20 s42。
- 判定指标: best_feasible_cost gap vs BKS + 相对 base/div; 档案机制指标
  (distinct routes / mean Jaccard) 报告。

## 判定阈值 (s42 单 seed, 方向性)

- **支持**: pr15/pr20 上 ∃κ: a2 ≤ base − 0.5pp 且 ≤ div − 0.3pp
- **弱**: ∃κ: a2 ≤ base − 0.3pp (但未赢 div)
- **关闭**: 全部 κ 与 base 差 < 0.3pp 或更差
- pr06 窄窗: a2 若 ≥ base + 0.5pp → 确认条件化必要性 (与 div 同向, 佐证
  "多样性非免费午餐" 跨机制成立)。

## 执行纪律

- 串行独立进程 (bash for), results/population_ab/{inst}_s42_a2_k{k}.json;
  确定性 rng; 完成后分析 + 判定文档; 引擎默认行为由既有 120 锁存测试保证
  (新测试 +4)。
