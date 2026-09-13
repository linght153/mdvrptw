# TW 硬约束修复记录 (重大 bug 修复)

> 发现: 2026-08-08 · 修复完成: 2026-08-08

## 1. 发现过程

用户提供 Cordeau 官方 .res 权威解 (C:\Users\13395\Downloads\C-mdvrptw-sol)。
验证发现: ALNS 静态解 861.32 比权威解 1083.98 **还低 20%** → 不合理。

深入检查: **861.32 的解有 31 个 TW 违规** (客户 42 迟到 148 分钟等)。
861.32 实为 **MDVRP 无时间窗口径** 的距离最优, 不是合法 MDVRPTW 解。

## 2. 根因链

1. `Solution.is_feasible()` 只检查容量 + 客户覆盖, **不含 TW**;
2. ALNS 默认 repair 算子集合是纯距离插入 (greedy_cost 等), TW-aware 的
   greedy_cost_tw 存在但**从未被默认使用**;
3. `_tw_insertion_delta` 在 `tw_penalty_weight <= 0` (默认 0) 时**退化为
   纯距离增量**, 等于没用;
4. 局部搜索 (two_opt) 纯距离改进, 可能**破坏 TW** 且无事后修复;
5. 修复算子找不到 TW 可行位置时**硬塞进违规位置**而不是开新车
   (权威解策略是"多用车、少装货"满足 TW: 8 路由、每车 1-10 客户)。

## 3. 修复内容 (4 处)

| 文件 | 修复 |
|---|---|
| src/core/solution.py | `is_feasible()` 增加 TW 硬约束 (实例含 TW 时违规→不可行) |
| src/operators/repair.py | `_tw_insertion_delta`: TW 实例下即使权重 0 也用回退权重 1000 |
| src/operators/repair.py | `_insert_cheapest_cost_tw`: 只接受 TW 可行位置, 无可行位置则开新车; 新增 `_route_tw_excess` 增量检查 (性能) |
| src/solvers/standard_alns.py | TW 实例强制用 greedy_cost_tw (传入 engine 的 repair_ops) |
| src/alns/engine.py | 初始解/每次迭代 LS 后/接受时 LS/final LS 四处强制 `_enforce_tw` |

## 4. 验证

- 新增 `tests/test_tw_hard_constraint.py` (3 用例): is_feasible 含 TW /
  ALNS 默认 TW 可行 / 成本接近权威解
- 修复后 ALNS: TW 零违规, 成本 1423.77 (vs 权威 1083.98, 40s 预算 gap 31%)
- 修复前: 861.32 含 31 违规 (非法)
- 增量 TW 检查性能: 定向测试 45s → 13s

## 5. 影响范围 (需要重跑)

**所有使用 ALNS 的实验结果均基于 TW 违规解, 必须重跑**:

- [ ] M1 静态双目标基线 (benchmark_static_green.py)
- [ ] M4/M5 动态对比 (benchmark_scpo.py / benchmark_scpo_multi.py)
- [ ] M9-M10 绿色双目标 (benchmark_green_multi.py)
- [ ] 论文 §5 全部数值表 + 摘要数字

**论文口径更正**:
- ~~"最优成本 861.32 = 公开参考值"~~ → 861.32 是 MDVRP 值, 非法
- 权威 BKS: pr01-05 = 1083.98 / 1763.07 / 2408.42 / 2958.23 / 3134.04
  (官方 .res, 已逐位复现验证)
- 修复后 ALNS gap: pr01 40s → 1423.77 (vs 1083.98, +31%) 需按新基准报告
