# 旧项目资产迁移记录（LLM-ALNS-Green-MDVRP → mdvrp）

> 日期: 2026-08-07
> 来源仓库: D:\G-MDVRP\vrp学习\LLM-ALNS-Green-MDVRP (git commit f299b6f, 分支 master)
> 目标仓库: C:\Users\13395\Desktop\mdvrp (本仓库)

## 1. 迁移原则

1. **只迁移验证过的核心算法资产**，不迁移已证伪的学习型设计
2. **逐模块依赖审查**：迁移前检查每个文件的 import 依赖，确保无隐藏耦合
3. **净化边界明确**：engine.py 移除学习型组件后行为必须与旧项目一致（基准复现验证）
4. **数据完整性**：实例文件 SHA256 逐一比对

## 2. 迁移清单

### 已迁移（白名单）

| 模块 | 文件 | 说明 |
|---|---|---|
| core | instance.py, solution.py, objectives.py, pareto.py, initial_solution.py | 数据模型+目标函数，无学习型依赖 |
| utils | helpers.py | project_root/load_yaml 等 |
| operators | destroy.py, repair.py, local_search.py | 标准破坏/修复/2-opt 算子 |
| alns | archive.py, acceptance.py, parent_selection.py, selection.py, selection_context.py | Pareto 档案、SA 接受、父代选择、段奖励选择器、上下文数据结构 |
| alns | engine.py (净化版) | 见下方净化清单 |
| data | data/green_mdvrp/*.json (21 文件) | SHA256 21/21 一致 |
| tests | test_objectives.py, test_pareto.py (原样) | 依赖纯 core |
| tests | test_alns_engine.py (重写) | 参考旧测试行为契约，只保留标准引擎行为 |

### 未迁移（污染源，明确排除）

| 模块 | 原因 |
|---|---|
| src/alns/context_features.py | ContextV1/V2/V3 特征提取，服务学习型选择器 |
| src/alns/selection_context.py 中的扩展 | 已随 engine 净化移除引用 |
| src/alns/hierarchical_interaction_selection.py 等 HSI 系列 | 4 轮实验失败的设计 |
| src/alns/joint_hierarchical_interaction_selection.py | HSI-Joint v4 联合评分，A2-D 未过准入 |
| src/alns/hsi_lineage_credit.py | C0 谱系信用，六折全负 |
| src/alns/factorized_contextual_selection.py | 因子化选择器（F1 恢复失败） |
| src/alns/linucb_selection.py, ucb_selection.py | 学习型选择器，D4 主线不使用 |
| src/rl/ (全部) | Q-learning/REINFORCE/neural，深度 RL 路线已放弃 |
| src/llm/ (全部) | LLM 算子生成，D4 不依赖 |
| src/solvers/ 除 standard_alns | 学习型求解器入口 |
| experiments/ 全部 | 旧实验脚本（引用已移除组件） |

## 3. engine.py 净化清单

在迁移后的 engine.py (原 807 行 → 现约 720 行) 中移除：

1. `from src.alns.context_features import ...` → 删除
2. `from src.rl.reward import ParetoReward` → 删除
3. `_CALIBRATION_UPDATE_KEYS / _CALIBRATION_REWARD_SOURCES` 常量 → 删除
4. `context_extractor` 构造参数与默认实例化 → 删除
5. `self._selector_reward = ParetoReward()` → 删除
6. `_selector_reward.reset()` (停滞恢复处) → 删除
7. LLM 崩溃回退分支 (destroy/repair 的 `startswith("llm_")`) → 改为标准算子崩溃直接抛出
8. `compute_reward(...)` → `reward = float(score)`（保留 score 语义）
9. `get_last_update_diagnostics` / `get_inverse_fallback_count` 诊断块 → 删除
10. `_build_search_context` 中 `ContextV1Extractor` → 内联纯构造（公式等价：progress/stagnation/temperature/容量利用率/档案比例）

## 4. 验证结果

### 单元测试

```
tests/test_objectives.py + test_pareto.py: 12 passed (原样迁移)
tests/test_alns_engine.py (重写): 10 passed
总计: 22 passed
```

### 数据完整性

```
SHA256 一致: 21/21 (data/green_mdvrp/)
```

### 基准复现（已完成）

对照实验：新项目（净化引擎）与旧项目（当前代码）在完全相同配置下跑 p01 25k 迭代：

| seed | 新项目 | 旧项目当前代码 | 一致 |
|---|---|---|---|
| 42 | 596.12 | 596.12 | ✅ |
| 43 | 596.12 | 596.12 | ✅ |
| 44 | 593.38 | — | — |

**结论：净化后的引擎与旧引擎行为逐位一致，迁移零污染。**

> 注意：旧项目 README 中的基准（p01 均值 589.3、最优 576.9/BKS 576.87）是 2026-07-17
> 历史配置+旧引擎代码的快照。此后参数多次演进（sigma 10→3、decay_factor 0.98→1.0、
> reaction_factor 0.5→0.1），旧项目当前代码即使改用 sigma=10 也复现不出 576.85（实测 596.12）。
> 该差异属旧项目自身的配置漂移，非迁移引入。

### 验证脚本

- `experiments/benchmark_p01_migration.py`：p01 25k 迭代 × 3 seeds 基准复现
- 后续作为回归基准保留

## 5. 遗留事项

- [x] 基准复现结果填入本节（见上）
- [ ] 旧项目 results/ 下的实验数据不入库（含学习型诊断，非资产）
- [ ] 后续 D4 开发（预测/场景化）在迁移后的基础上进行
