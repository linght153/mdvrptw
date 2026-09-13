# 全仓代码验收报告 (2026-09-08) — main @ 8dd7d5c 之后

范围: 全部 18,072 行 Python (src/ 17 模块 + tests/ 33 文件 + experiments/ + scripts/)
+ 2 个 bash 编排脚本。方法: 主代理精读 engine.py 全 2671 行 + 核心地基抽查; 3 个
并行审查子代理分片深审 (core+alns / operators+solvers / tests+experiments+scripts);
交叉核验全部 high/medium 发现 (逐条读码复核); compileall + 17 模块 import 冒烟;
全量回归 ×2 (仓库根 + 异 cwd tests/ 目录), 均 205 PASS。

## 一、核验为误报的发现 (子代理报告已排除)

1. **RVND "单邻域即退"** (报告: local_search L508 内层 break) — **误读缩进**:
   `if not improved_any: break` 与 `for idx in order:` 同级 (while 体内), 一轮内
   全部邻域试完才退出, 任一改进即重洗重启 — 标准 RVND, 实现正确, 无改动。
2. shaw_removal/cluster_removal 空解崩溃 — 引擎主循环 destroy 只作用于全覆盖解,
   不可达 (低危 API 面已记录, 未修)。

## 二、已修复 (全部行为中性; 提交见下)

| 文件 | 修复 | 类型 |
|---|---|---|
| tests/test_objectives.py | `test_cost_unit_is_km` 恒真断言 (成本与自身比) → 手工期望距离和断言 | 测试补真 |
| tests/test_soft_capacity.py | `X or X` 复制笔误 → `X and 车场0保持原状` | 测试补真 |
| tests/test_instance_tw.py / test_mdvrptw_parser.py + 8 文件 | 12 处 `open("data/…")` 相对路径 → `Path(__file__)` 锚定 (共 10 文件) | 鲁棒性 |
| src/alns/engine.py | `archive_capacity < 1` 显式 ValueError 守卫 (原空档时 add() 崩) | 防御 |
| src/core/objectives.py | COPERT 两分支惰性 import (ModuleNotFoundError 迷雾) → 显式 NotImplementedError 指向归档 tag pre-reorg-20260902 | 归档诚实 |
| src/alns/engine.py | σ 键注释: 确认从未接线, shaw 恒 σ=10.0; **禁止接线修复** (全 runner 显式传 3.0, 接线即全实验口径迁移) | 文档锁定 |
| src/operators/repair.py | 增量 TW helper 注释: "与全重算一致" → "判定等价已实证, 数值为截断界" | 文档纠偏 |
| src/core/objectives.py | removal_cost_delta 注释: 抵消仅 α=0 成立, MEET α>0 低估 33-54% (公式未动, 保历史口径) | 文档纠偏 |
| scripts/rerun_m5_600s.sh | 对齐 official10: 断点续跑 skip-if-exists | 工具一致 |

## 三、已知未修 (行为改变/死线资产, 记录在案)

- [中] 停滞重启解无 `_enforce_feasibility` (init 路径有) — 历史行为, 修 = 行为改变
- [中] `instance` 经 setattr 传 soft 标志/λ — 多引擎共享实例串扰; runner 单实例不触发 (API 陷阱)
- [中] Solution.copy 共享 customers/assigned_depot_index — 入库前 _sync 兜底, 结构性风险已注释
- [低] `stagnation_recovery_divisor=0` 零除; restart 无 enforce; is_feasible 覆盖检查用 set (理论漏重复服务, 链上不可达)
- [低] ParetoArchive.save/load 零调用方 + frozenset 不可序列化 (启用前须修); 双目标 add 返回 True 但新解可被裁
- [低] analyze_structure_diff 数据溯源混用 (s42 回退旧批无标记) + shared_segments 死参数
- [低] scripts/rerun_official10_600s.sh 续跑不校验 JSON 完整性
- [低] pyvrp_solver: 车辆数取 depots[0] + int() 截断 — 语料全整数值/上限均一, 不触发
- [低] destroy 空解崩溃; capacity=0 前守卫已补; shaw σ 恒 10.0 (见上); 双目标/vehicle_start/COPERT 相关 = 已归档死线 (2dc0c97, tag pre-reorg-20260902)
- [低] 测试弱点: test_tw_hard_constraint 仅下界门; test_pareto 仅 ≥0 断言; test_score_scheme 阈值无机制锚定 — 均保留 (质量门补上限易脆)

## 四、口径结论 (论文相关, 验收附带确认)

1. **全历史实验 destroy 池实际 σ = 10.0** (config 键 σ=3.0 从未生效) — 论文方法节
   须写 σ=10.0 或删 σ 提法; 影响范围: 所有 shaw 臂 A/B 配对 (双侧同 σ, 配对差仍有效)。
2. COPERT/绿色/动态线 = 已归档死线 (tag pre-reorg-20260902), 不在论文口径内。

## 五、回归

- 仓库根 pytest: 205 passed (4:31)
- 异 cwd (cd tests) pytest: 205 passed (4:28) — CWD 无关性验证
- compileall 全绿; src 17 模块 import 全通

提交: 待列 (git log -1 后补)。
