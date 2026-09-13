# 最大路线时长约束修复 — 实现规格 (2026-09-11)

背景: 2026-09-10 GPT 外审第 5 条(约束核验)触发的全链排查证实: **实例定义的最大路线
时长 D 从未在引擎与 pyvrp 桥接中实现** (src 无 duration 逻辑; 桥接 shift_duration=1e6
=关闭), 自产全部批次与 pyvrp 解在标准问题下不可行 (popcxge pr15 超 79.6 / pr16 超 58.2 /
pr20 超 54.4 / pr04 超 118.7; base600s pr20 超 117; pyvrp pr16 最高超 191.5)。
用户拍板 = 实现约束 + 全量重跑重审计 (方案 B)。

证据锚 (权威):
- VRP-REP 官方数据包 readme (可复取): "D = maximum duration of a route";
  方案文件 "l k d q list" 中 "l = 车场号, k = 车辆号, d = duration of the route";
  成本 = "total duration excluding service time"。
- 官方 .res (pr01-05, 79 条路线): 逐路线 duration 全部 ≤ D 且 3 例贴界
  (pr01 499.68/500、pr04 439.07/440、pr05 419.39/420) → 该约束在标准问题中有效且紧。

## 1. 时长口径 (唯一定义, 实现不得自行解释)

对于路线 r = [车场 d, c1, ..., ck, d], 按"车场 0 时刻出发"前向排程:
- 逐弧推进 t; 到达客户 b 时 start_b = max(t, e_b), wait_b = start_b − t,
  t = start_b + s_b; 末弧返回车场得 return_time。
- **duration(r) = return_time − wait_{c1}** (即首客等待不计入; 其余等待计入)。
- 等价形式 (实现参考): duration = Σ弧行驶 + Σ服务 + Σ_{j≥2} wait_{c_j}。
- 边界: k = 0 → duration = 0。k ≥ 1 且 wait_{c1} = 0 → duration = return_time。
- 约束: duration ≤ D (实例定义 D 时); 另: return_time ≤ 车场最迟返回 l_depot
  (现 pr 实例 l=1000, 非绑定, 一并入检查以完整化约束集)。

唯一验收基准: 对官方 .res pr01 的 8 条路线, 用本实现计算 duration 与 .res 列逐位一致
(|diff| < 0.01, 官方仅 2 位小数)。

⚠️ pyvrp 内置 duration 语义与上述不同 (它把可吸收的中间等待也通过延迟出发消掉, 值更小;
实测 pr01: 401/131/342/47/243/442/26/423 vs 官方 410.12/…/499.68)。pyvrp 侧处理见 §5。

## 2. 数据模型改动 (src/core/instance.py, scripts/parse_cordeau_mdvrptw.py)

- `Instance.route_duration_limit: float | None = None` (新字段, 默认 None = 不约束)。
- `instance_from_parsed`: 从 `parsed.route_duration_limit` 填充。
- `load_instance` (green_mdvrp JSON): 可选键 `route_duration_limit`; 缺失 → None
  (旧数据行为逐位不变)。
- `Depot` 增加可选字段 `tw_late: float | None = None` (最迟返回); `instance_from_parsed`
  从 parsed.depots[i]["due_time"] 填充; JSON 侧可选。无该字段 → 不检查。

## 3. 可行性语义改动 (src/core/solution.py)

- 新增 `Solution.route_duration(depot_index, route, route_index=0) -> float`
  (基于既有 route_schedule + 末弧回场; 复用 _route_start 的车辆起始状态, 静止场景 = 0)。
- 新增 `Solution.duration_violations() -> list[dict]` (仿 tw_violations; 实例无 D 或
  无路线 → 空)。返回 [{depot, route_index, duration, limit, excess}]。
- `is_feasible()` 追加两项 (在现有 TW 检查后):
  a. 存在 D 时: duration_violations() 非空 → False;
  b. depot.tw_late 非 None 时: 任一 return_time > tw_late → False。
- 性能: 允许新增一次 schedule 扫描 (is_feasible 当前已有 1 次); 若冒烟超标, 合并
  tw/duration 为单遍诊断 (优化项, 不改变语义)。

## 4. 引擎改动 (src/alns/engine.py, src/operators/repair.py, src/operators/local_search.py)

### 4.1 可行化链
- `_enforce_feasibility`: 在 `_enforce_capacity` / `_enforce_tw` 之后追加
  `_enforce_duration`。
- 新增 `_enforce_duration(solution)`: 对每条超长路线: 反复移除"贡献最大"客户
  (启发: 移除后 duration 降幅最大/或最远客户), 用既有 TW 修复通道重插
  (受 capacity/TW/duration/车辆数全约束); 无可行位置且该车场仍有空闲车辆 →
  开单车新路线; 仍失败 → 放弃该客户? **禁止**: 未分配客户必须保持客户总数守恒
  (由外层 `_repair_unassigned` 兜底链路负责), `_enforce_duration` 迭代上限
  (如 50 步) 后返回当前解 — 上层照常拒绝不可行解 (与 _enforce_tw 同语义)。
- `_postprocess_after_repair`: 链尾接入 duration 强制 (TW 强制已在内);
  `_postprocess_tw` (M4 罚域, gated 默认关): 同步接入 duration 强制
  (罚域只松弛 TW, 不松弛 duration)。

### 4.2 修复热路径 (repair.py)
- `_insert_cheapest_cost_tw` / `greedy_cost_tw_insertion` / `_repair_unassigned`:
  候选插入位置须同时满足 duration ≤ D (无 D 实例跳过); 无可行位置 → 既有"开新车"
  候选逻辑 (每车场剩余车辆>0 时)；新车候选同样检查 duration。
- 增量计算 (性能关键, 复用既有 `_route_tw_excess` / `_insert_incremental_excess` /
  `_route_timeline` 的增量骨架):
  - 插入位置 p > 0 (首客不变): duration 增量 = (新 return − 旧 return) − 0
    (首客等待不变)。由于插入只影响 p 之后, 从 p 向前传播一次 (O(len−p))。
  - p = 0 (新首客): 全量重算 (O(len)), 低频, 可接受。
  - 移除客户: 同理向前传播。
  实现形态: 让现有增量辅助函数同时返回 (tw_excess, dur_excess), 一处维护两种约束,
  禁止两套重复 propagate。
- 单一事实源: 增量未覆盖的路径 (疑似候选/调试/测试) 一律回落到 Route/Solution 层
  的 duration() 实现, 不得复制公式。

### 4.3 局部搜索 (local_search.py)
- `_move_tw_ok` 升级为对受影响路线同时检查 TW 与 duration (改名或增加 `_move_dur_ok`,
  任一失败即拒绝该候选移动)。
- `rvnd_improve` 的候选终检 `candidate.is_feasible()` 保持 (duration 已纳入)。
- relocate/swap/two_opt/two_opt_star 的容量内联检查保持不变; duration 走上述过滤。

### 4.4 统计与输出
- 结果 JSON (runners) 增加字段: `max_route_duration_excess` (float, 0 = 全合规) 与
  `max_return_time` (float)。既有字段语义不变。
- `population_stats` / `probe_stats` 不改键集 (避免扰动)。

## 5. pyvrp 桥接 (src/solvers/pyvrp_solver.py)

- `instance_to_pyvrp`: 当 `inst.route_duration_limit` 非 None → 每个 VehicleType
  `shift_duration = int(limit)` (原为 1_000_000); 无 D → 维持 1_000_000 (行为不变)。
- **语义差异处理 (待泄漏探测结果)**: pyvrp 的 duration (延迟出发口径) ≤ 官方 duration,
  即 shift_duration=D 时 pyvrp 可接受官方口径下超时的路线 (泄漏)。
  泄漏探测 (probe_durfix_*, 2026-09-11 凌晨运行中) 数据决定处置:
  - 若 shift_duration=D 下 pyvrp 解全部满足官方 duration ≤ D → 采用 + 外部复核;
  - 若仍有泄漏 → 备选: (i) 每实例收紧档校准 (记录口径); (ii) 后处理修复;
    (iii) 其他方案须经判定文档记录, 不得静默采用。
- 独立重算器与 `probe_pyvrp_wide.py` 的输出增加官方口径 duration 复核。

## 6. 独立核验器 (新, scripts/verify_solution_constraints.py)

零引擎依赖 (只 import parse_cordeau_mdvrptw + math/numpy), 输入: 实例名 + 解文件
(支持 {depot: [[客户序列]]} 的 dump JSON 与 .res), 输出逐项核验报告:
覆盖完整/重复访问、逐路线容量、每车场车辆数、往返同车场、客户 TW、车场最迟返回、
最大路线时长 (官方口径)、原始欧氏距离与总成本。任一失败非零退出。
用途: ①本次全部重跑解的事后核验; ②pr04/pr06 等争议解的标准约束核验;
③GPT 第 5 条清单的落地工具。

## 7. 测试 (TDD, 先写失败测试)

1. duration 公式: 手工 2-3 条固定路线 (含首客等待/中间等待/零等待三型),
   断言与手工值一致; pr01 .res 8 路线断言 (需 data 存在, 可标 skip)。
2. is_feasible: 构造超长路线 (已知超 5) → False; 恰好 == D → True。
3. repair: 修复算子在"只剩超长路线位置"时不选择该位置; 开新车候选可用。
4. LS: 会产出超长的候选移动被拒 (构造 fixture)。
5. _enforce_feasibility/_enforce_duration: 人为注入超长 → 修复后 duration 合规
   (或保持客户守恒且返回未修复态由上层拒绝 — 断言客户守恒)。
6. 回归: 全部既有测试; 无 D 实例 (green_mdvrp fixture) 逐位行为不变。
7. 确定性: 同一 seed 两次运行 (含新字段) 逐位一致。

## 8. 验收清单 (全部满足才提交)

- [ ] 全量 pytest 绿 (基线 208 + 新增), 含异 cwd 运行。
- [ ] pr01 官方 .res duration 逐位核对通过 (核验器)。
- [ ] 冒烟: pr15/pr20 各 500-1000iter × (base, popcxge): 产出解 `verify_solution_constraints` 全过;
      墙钟对比: 新增开销 ≤ 25% (超标 → 增量优化后复测)。
- [ ] 确定性重跑逐字段一致。
- [ ] `git diff --numstat`: 改动全部在预期文件; 无删除行丢失 (除非有据)。
- [ ] 结果命名: 本批全部输出到 `results/durfix/{batch}/...`; 旧 results 不动。
- [ ] 不 push; 不在本任务内跑长批 (长批由 Hermes 编排)。

## 9. 边界 (硬性)

- 不引入 config "默认关" 假开关: 本修复是**语义修复** (TW/容量修复先例), 由数据
  (实例是否有 D) 驱动; 无 D 实例行为逐位不变是唯一兼容承诺。
- 不删旧代码; 不动 results/ 既有文件; 不改既有生成脚本的默认输出路径。
- 歧义停下报告, 不自行扩展设计。

## 10. 交付物

engine + operators + tests + scripts/verify_solution_constraints.py + runner 字段更新;
冒烟证据 (日志/JSON); 一份实现报告 (commit hash / 测试数 / 冒烟结果 / 出入清单)。
