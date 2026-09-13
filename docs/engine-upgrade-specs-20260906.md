# 引擎升级工程规格 (方向①车场重平衡 + 方向②破坏性教育, 2026-09-06)

给 Claude Code 的实现规格 (HEAD cc9b824)。背景与判定:
docs/structure-diff-verdict-20260906.md + docs/structure-diff-prereg-20260906.md。
纪律: 项目根 CLAUDE.md。只做实现/测试/提交; 实验批与判定由 Hermes 侧
按 docs/rebalance-ab-prereg-20260906.md 执行 (本规格不含判定协议)。

客户/节点 id 约定: 路由 seq 为全局节点号, 客户号 i = 全局号 − num_depots;
客户最近车场 = argmin_k inst.distance_matrix[k][全局号] (k = 车场全局号,
0..nd−1)。实例容量 inst.max_vehicle_capacity; 车场车辆上限
inst.depots[di].vehicles_available; 可行插入用现有 greedy_cost_tw 修复
路径 (TW 实例), 禁止新造 TW 检查。

═══════════════════════════════════════════════════════
方向①: 车场分配重平衡 (rebalance) — 机制实现规格
═══════════════════════════════════════════════════════

动机: 结构分析判定 1/2 — pr20 自产 600s 解错配客户 66/63/83 vs pyvrp
50/51/51, 错配罚金 1447/1490/2865 vs ~890, 且 seed 漂移 2 倍; 自产独有
错配多为"纯错配" (最近车场被无视)。单轨迹缺显式扰动/修复分配的算子。

## 1a. 错配优先移除算子 (destroy 侧, 新文件 src/operators/destroy.py 内新增)

`def mismatched_removal(solution, k, rng)` 加入 DESTROY_OPERATORS:
- 计算当前解所有已服务客户的错配度 = d(实际路由车场, 客户) − d(最近车场,
  客户) (全局号索引 distance_matrix; 最近车场允许用容量可行最近 —
  repair.py 已有 _nearest_depot_with_capacity 可参考其语义, 但移除算子
  本身只做移除, 不检查容量)。
- 错配度 > 1e-9 的客户按错配度降序移除, 直到移除 k 个或错配客户耗尽;
  不足 k 用 random_removal 语义补齐。
- 返回 (partial, removed) 与其它算子同接口。移除后 repair 走现有全车场
  greedy_cost_tw (TW 实例), 客户自然获得跨车场重插机会。

## 1b. 周期重分派 (engine 侧, src/alns/engine.py, gated)

config 键 (默认关, 默认行为逐位不变):
- `rebalance_mode`: "none" | "removal" | "redispatch" | "both" (默认 none)
- `rebalance_interval`: int 默认 0 (=关; >0 且 mode != none 才生效)
- `rebalance_ratio`: float 默认 0.20 (removal 模式的 k 比例,
  仅当该迭代 destroy 走 rebalance 路径时用)

engine 主循环内 (参照 population_crossover 分支的位置与风格):
- mode 含 "removal": 每隔 rebalance_interval 迭代, destroy 阶段以概率
  1.0 用 mismatched_removal (k = rebalance_ratio × n) 替代正常 destroy
  选择 (不更新 selector 权重, 同 cx 分支惯例); 其余迭代行为不变。
- mode 含 "redispatch": 每隔 rebalance_interval 迭代, 对当前 parent_sol
  做一次错配修复 pass: 对错配客户按错配度降序, 逐个尝试移动到最近车场
  的可行位置 — 用现有插入机制 (TW 实例 greedy_cost_tw 语义: 只接受
  TW/容量可行位), 仅接受成本严格下降的移动; pass 完成后该迭代照常走
  destroy/repair (重分派是独立后处理, 不进 selector 统计)。修复 pass
  必须保覆盖/容量/TW 可行 (结束时 is_feasible() 断言或 _enforce 兜底
  语义一致)。
- mode "both" = removal + redispatch 同开 (间隔共享 rebalance_interval)。

## 1c. 测试 (tests/test_rebalance.py, TDD)

- mismatched_removal: 构造已知错配解 (手工把客户塞进非最近车场路由),
  断言移除序列优先含错配客户; k > 错配数时补足到 k; 空解/无错配回退
  不崩。
- redispatch pass: 注入错配解 → pass 后错配罚金 (Σ 2×(d_实际−d_最近)
  近似, 或 geo 一致客户数) 不增且成本不增; 默认 none 模式 engine 行为
  逐位不变 (关键回归: 现有全套测试全过即证明 gated 生效, 另加一个
  config 缺省 == none 的断言)。
- engine 集成: rebalance_interval=1 + mode removal 跑 20 iter 不崩、
  解可行; mode redispatch 同。

═══════════════════════════════════════════════════════
方向②: 破坏性教育 (educate) — 机制实现规格
═══════════════════════════════════════════════════════

动机: 结构分析判定 4 — 差异 = 路由划分/成员组合盆地不可达; LS 收敛对照
证明纯 LS 教育 (深 RVND) 无法迁移划分 (两侧解均近锁定)。故教育必须是
"破坏 + 重组" (HGS 教育思想): 后代先小规模破坏再重建, 改变路由成员
组合, 而非只做邻域精化。旧 cx/pdiv 探针 = route_copy 保序复制 + 短 LS,
不改变划分 → 与判定 4 一致地无收益。

## 2a. 机制 (src/alns/engine.py + src/operators/crossover.py, gated)

config 键 (默认关, 默认行为逐位不变):
- `educate_mode`: "none" | "deep" (默认 none; deep = 对 cx 后代做
  破坏性教育再准入)
- `educate_burst_ratio`: float 默认 0.20 (后代被破坏的客户比例)
- `educate_burst_max_iter`: int 默认 1 (破坏-重建轮数, 1 = 单轮)
- `educate_ls_budget`: int 默认 300 (教育内 LS 的 rvnd_improve
  max_iterations 预算; 现 ls_on_accept 短 LS 预算 = ls_max_iter 10-30)

调用点: engine 主循环 population_crossover 分支 (child_cx 生成后,
_postprocess_after_repair 前) — educate_mode=deep 时:
1. 从 child_cx 随机移除 educate_burst_ratio × n 客户 (用 random_removal
   语义或 route_removal 语义, 实现选一即可, 注释说明);
2. 用现有 repair (TW 实例 greedy_cost_tw) 重插 → 教育后解 child_educ;
3. child_educ 走 _postprocess_after_repair + LS (ls_on_accept 逻辑不变),
   若 child_educ 可行且成本 < child_cx, 用 child_educ 作为本迭代
   new_sol; 否则退回 child_cx (教育不劣化当前迭代轨迹);
4. cx_stats 扩展: educate_improved (bool), educate_delta (成本差) —
   engine 已有 cx_stats 记录 novel_route_keys 的机制可复用;
   教育本身不更新 selector 权重 (同 cx 分支惯例)。

## 2b. 验收要点 (实现者自检, 写进测试)

- 教育后解必须可行 (容量/TW/覆盖) — 断言。
- 教育产生"划分变化"证据: 小规模对照 (20 iter × 固定 seed) 中
  educate_mode=deep 的 novel_route_keys 计数不显著低于 none —
  防止实现退化成纯 LS (纯 LS 已知无法改划分)。
- 默认 none 逐位不变 (全套测试回归)。
- 教育成本可控: 单次教育 ≤ 1 次 destroy/repair + 1 次 LS 的量级
  (不得引入每迭代全量重优化)。

═══════════════════════════════════════════════════════
执行顺序与提交 (全部完成后按此提交)
═══════════════════════════════════════════════════════

1. 读 CLAUDE.md + 本规格 + src/alns/engine.py (主循环 380-470 行,
   _crossover_child 1006-1070 行) + src/operators/destroy.py +
   src/operators/repair.py (greedy_cost_tw/_nearest_depot_with_capacity)
   + tests/ 既有结构探针测试风格 (tests/test_structure_probes.py)。
2. 方向①: RED → GREEN → 全量 pytest → 提交 "feat: 车场分配重平衡
   (mismatched_removal + 周期重分派, gated 默认关)"。
3. 方向②: RED → GREEN → 全量 pytest → 提交 "feat: 破坏性教育
   (educate_mode=deep, cx 后代破坏-重建, gated 默认关)"。
4. 不得自行跑 600s/6000iter 长实验批 (协议外), 不得动 results/ 与
   docs/ 下 verdict/prereg 文档; 新机制冒烟允许 (pr01/pr15 短预算
   ≤60s 验证不崩)。
