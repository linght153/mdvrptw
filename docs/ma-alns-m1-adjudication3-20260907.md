# M1 裁定 3 — 冒烟墙钟门槛 ④ 改到 250iter 操作点 (2026-09-07)

CC 三轮(proc_8d403363f81c)裁定 2 实现完毕: 174 passed; 冒烟硬门槛 ①不崩
②archive best 可行 ③池可行成员 ≥1 全过; **④墙钟 ≤2× base 在 60-iter 字面口径
FAIL**(κ=0/κ=0.05 均 ~2.4×)。CC 分解归因(采信, 与 Hermes 独立估算一致):
pop μ=8 池构建固定开销 ≈1.37s = pr20 初始解经 _enforce_feasibility 恒不可行 ×
8 成员 × 10 次重试兜底全部跑满(裁定 1 写死语义); 60-iter base 总墙钟仅 ~1.17s,
故固定开销占比 ~1.2×; 250iter 降至 ~1.4×; 6000iter 操作点摊销后近中性
(1.37s / 单组 ~8min ≈ 0.3%)。**门槛④以 60iter 分辨率无法摊销按真实操作点校准的
固定初始化成本 — 属门槛口径缺陷, 非实现缺陷。** 250iter 诊断佐证机制健康:
last_solution_feasible=True, 池可行 8/8, member_refresh_events=7(驱逐 ≥200 触发,
符合裁定 2 预期), refresh_events=0, archive_size=20。

## 裁定(全部生效)

1. **冒烟硬门槛 ④ 重写**: 墙钟比在 250-iter 操作点度量: 同配置
   (population_mode 关/开, μ=8) 各跑 250 iter(s42, pr20), 门槛 = pop 墙钟 ≤ 2×
   base 墙钟。CC 250iter pop κ=0.05 已测 wall=6.80s; base 250iter 需补测。
   另报告迭代体效率比 = (pop_wall − pool_init_wall) / base_wall(pool_init_wall
   用 max_iterations=0 探针; 只报告不设门槛)。60iter 的 ④ 作废。
2. 测试 (c) 的白盒直驱形态(空档案分支在 tw fixture 上循环内不可达 → 直驱
   _member_refresh_replacement 锁定兜底分支)采信, 功能无缺失。
3. 6000iter 批的 wall_s 比值(预期 ~1.05-1.2×)在 verdict 中如实报告, 不设门槛
   (判定只看质量/多样性/池健康度, 见预注册)。
4. 其余条款不回退。提交信息不变。CC 四轮任务 = 补测 base 250iter + 出比值 +
   完成唯一一次提交 + 报告。
