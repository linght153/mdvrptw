# 方向②破坏性教育 A/B 预注册 (2026-09-06)

衔接: 结构差异 verdict 判定 4 (差异 = 路由划分/成员组合盆地不可达; 纯 LS
教育无法迁移划分 — LS 收敛对照 6/6+6/6 近锁定) + 方向① verdict (单客户
粒度分配修复空转, 组合级迁移是唯一未测杠杆)。实现规格:
docs/engine-upgrade-specs-20260906.md 2a (CC 已实现, commit 5a680cd,
educate_mode gated 默认 none, 全量 154 PASS)。

## 假设

H-E: cx 后代经破坏性教育 (随机移除 20% → 重建 → LS) 后, 在固定迭代
口径下: (i) 机制生效 — 教育产生划分变化 (教育后解产生档案外新路由键,
educate_improved 频率 > 0 且 novel 键产出不随轮次枯竭); (ii) 质量转化
— 6000iter best 优于 cx 与 base, 配对差跨 seed 稳健。

## 臂 (6000iter × pr15/pr20 × s42-44; cx 与 base 用既有确定性文件不重跑)

- base: 既有 population_ab/*_base.json (pr15 2674.88/2585.62/2600.94;
  pr20 3307.93/3244.55/3325.29)
- cx: 既有 population_ab/*_cx.json (div+population_crossover 配置,
  cx_iterations 已记录)
- educate: cx 同配置 + educate_mode="deep" (burst_ratio 0.20 /
  max_iter 1 / ls_budget 300 — 规格默认档, 单参数不调优)

## 判定 (两级, 硬屏障)

第 1 级 机制生效 (过程): educate 组 cx_stats: educate_improved 触发
比例 > 0 (6000iter 内教育至少成功若干次) 且档案路由键 (archive_
distinct_routes) ≥ cx; 否定 = 教育从未改进 (实现问题或后代不可教育)
→ 关闭不判质量。
第 2 级 质量转化: educate 配对差 vs cx 与 vs base: 同实例 3 seed
符号一致且 |均值差| > 同臂散布 (取 cx/base 中较大散布) 才支持; 否则
否定 → "教育有结构产出但无质量转化" (与 09-05 教育列/09-06 removal
同构), 单轨迹形态升级线收束, 转向论文定位。

过程记录: 每组存 educate 统计 (educate_improved 次数/比例、delta 均值、
novel 键、archive_distinct_routes、archive_mean_jaccard) 供机制解读。

## 执行

runner experiments/benchmark_educate_ab.py (模式同 benchmark_rebalance_
ab: engine 直连 + 末端 dump); 6 组 × ~210s ≈ 25 min 串行独立进程;
结果 results/educate_ab/。判定文档 docs/educate-ab-verdict-20260906.md。
