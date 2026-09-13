# M1 种群壳实现规格 — 给 Claude Code (2026-09-07)

背景与设计: docs/ma-alns-design-20260907.md(先读)。纪律: 项目根 CLAUDE.md。
只做实现/测试/本地提交; 实验批与判定由 Hermes 侧按 docs/ma-alns-m1-prereg-20260907.md
执行(本规格不含判定协议)。仓库: D:\mdvrptw, 分支 feat/axis2-memetic-alns,
基线 HEAD 308e846(本规格文档已提交在其上)。

## 必读文件(动手前)

1. CLAUDE.md(纪律: venv = D:/mdvrp/.venv/Scripts/python.exe; pytest 全量命令;
   gated 默认关 + 行为中性铁律; TDD 顺序; 提交纪律)
2. docs/ma-alns-design-20260907.md(动机与约束表)
3. src/alns/engine.py — __init__ config 解析区(参照 199-250 行 educate/rebalance
   块的 gated 风格); run() 主循环 293-815 行(语义逐项镜像对象); _count_novel_keys
   与 cx_stats 记录(1355-1375 行附近, route key 约定来源); _crossover_child 区
   1312-1432 行(不需要实现交叉, 只需确认 M1 不触碰这些分支)
4. src/alns/archive.py(ParetoArchive API: add/capacity/single_objective)
5. src/alns/acceptance.py(ParetoAcceptance: accept/cool/reset/_auto_calibrate)
6. src/core/solution.py(apply_destroy/apply_repair 是否原地修改 — M1 池成员
   禁止原地修改, 必须确认语义并写测试锁定)
7. tests/conftest.py(fixture: tw_instance 6 客户 2 车场合成 TW 样本) +
   tests/test_structure_probes.py / tests/test_rebalance.py(测试风格参照)

## 机制语义(逐条照做, 不留设计空间)

### config 键(全部默认 = 旧行为, 默认路径逐位不变)

- `population_mode`: bool 默认 False — 总开关
- `population_size`: int 默认 8 — 池成员数 μ(≥1, 1 = 单成员退化, 合法)
- `population_tournament`: int 默认 2 — 锦标赛规模(≥1, 1 = 随机选)
- `population_dc_kappa`: float 默认 0.0 — dc 权重(0 = 纯成本锦标赛; >0 时
  fitness = cost − κ×池内当前最优成本×dc, 镜像 A2 fitness 量纲)

### 非法组合守卫(在 __init__ 内, population_mode=True 时)

以下组合 M1 未测禁开, 抛 ValueError(默认关路径不触发, 无行为影响):
population_crossover=True / educate_mode != "none" / rebalance_mode != "none" /
soft_capacity=True / soft_tw=True。文案: "population_mode M1 forbids <键>".

### 接入点(唯一允许触碰 base 路径的一行)

run() 内 `run_started = time.perf_counter()` 之后、既有初始化代码之前插入:

    if self.population_mode:
        return self._run_population_mode(run_started)

_base 路径其余代码零改动。_run_population_mode() 新方法放 run() 之后。

### _run_population_mode 语义(逐项镜像 base, 差异只来自"多成员")

1. **初始化池**: 依次从 engine.rng 取随机源调 build_initial_solution(init_method)
   建 μ 个解 → 每个 _enforce_feasibility → (obj = calculate_objectives); 失败兜底
   语义与 run() 相同。池成员存 list[tuple[Objective, Solution]], 每次存入前
   .copy()(防共享实例/路由别名)。_sync_depot_assignments 对齐。
2. **SA 校准**: acceptor._auto_calibrate 时 calibrate(池内最优成员 obj,
   max_iterations)(base 用初始解 obj — 语义一致化到池最优)。
3. **档案初始播种**: 只加池内最优可行成员(iteration 0, type "population_init"),
   与 base 单点播种语义最接近(档案统计可比)。
4. **成员路由键缓存**: 每成员缓存 route-key set(键约定 = _count_novel_keys 的
   route key 同构; 读该函数后复用同构键)。仅当该成员被替换/接受新解时更新缓存
   (禁止每迭代全量重算 — 6000iter × μ=8 必须保持与 base 同量级墙钟)。
5. **主循环** (for iteration in 1..max_iterations):
   a. 时间检查 / 池级停滞检查(break): 语义 = base(stagnation_count >=
      stagnation_limit break; recovery_threshold = max(30, stagnation_limit //
      stagnation_recovery_divisor); 达到阈值 → 刷新最劣成员(fitness 定义见下,
      并列取序号小者)为新初始解(engine.rng 取源, _enforce_feasibility),
      restart_clear_archive=True 时 archive.clear(), acceptor.reset(),
      pool_stagnation = 0, population_stats["refresh_events"] += 1;
      与 base 不同点: 只换 1 个成员, 不重置其余成员与轨迹。
   b. **锦标赛选成员**: 从池中无放回抽 min(tournament, μ) 个不同成员(engine.rng),
      取 fitness 最小者 = 本轮成员 m*。fitness = obj[0] − population_dc_kappa ×
      best_pool_cost × dc(m); dc(m) = min 对其它成员 (1 − |S_m ∩ S_m'| / |S_m ∪
      S_m'|), 池唯一成员时 dc = 1.0。population_dc_kappa=0 → 退化纯成本选择。
      抽取消耗 rng 的次序必须固定(先抽子集再比 fitness), 保证同 seed 确定性。
   c. 迭代体(全部镜像 base 语义, 逐项):
      - 父代选择/档案父代 _select_parent 逻辑: M1 不用(池即父代源), 直接以 m*
        为演化对象 — 与 base 的 parent_entry 路径无关, 不调用 _select_parent。
      - destroy/repair: parent_sol = m* 解的副本; _compute_k 不变; selector.
        select_pair(context, rng) 选 (d, r)(context 用 _build_search_context(
        iteration, pool_stagnation, m*_sol, m*_obj)); apply_destroy/apply_repair;
        cx/rebalance 分支全部不执行(M1 禁开)。
      - _postprocess_after_repair(公共可行化链)。
      - LS 调度: 与 base 完全一致(iteration % ls_freq / ls_on_accept 分支,
        软容量/TW 分支不执行)。
      - 接受: acceptor.accept(m*_obj, child_obj, rng); 接受 → 该成员解/obj 替换
        (整体替换为新 Solution 副本); 拒绝 → 池不变(测试锁定)。
      - 档案: _sync_depot_assignments(child) 后可行才 archive.add(obj, {…,
        "destroy":…, "repair":…, "solution": child.copy(), "type": "population"});
        入档成功 added=True(score 语义 admission 分支与 base 逐位一致)。
      - score/selector.update: M1 无 cx → 每迭代都 update(action, outcome,
        next_context)(镜像 base 非 cx 路径); outcome 构造与 base 相同字段。
      - 降温: acceptor.cool() 每迭代一次。
      - 停滞: added → pool_stagnation=0(回调 on_new_archive 同 base); 否则 +1。
      - 轨迹: 每迭代追加一条(base 同构字段: context/decision/action/outcome,
        outcome 至少含 accepted/archive_added/feasible/crashed/discrete_score/
        normalized_reward/elapsed_seconds/cost/emission/archive_size; action 含
        destroy_name/repair_name 等; decision 可为全 None 键(同 base 结构,
        学习型选择器字段除外)); on_iteration 回调同 base。
6. **收尾**: 终局 soft 修复块不执行(soft 禁开); final LS 块镜像 base(ls_final 且
   档案非空 → _apply_final_local_search, TW 跳过逻辑在内); checkpoint 间隔逻辑
   镜像 base(如有配置)。self.last_solution/last_obj = 循环结束时最后被选成员
   的解/obj 副本(base 语义 = 轨迹末端解)。self.population_final = [(obj[0],
   route 结构摘要)]? — 不, 暴露为 self.population_final: list[tuple[Objective,
   Solution]] | None(mode on = 池终态副本列表, off = None; 供 runner 算池分散度,
   行为中性, 参照 last_solution 暴露先例)。
7. **统计**: self.population_stats: dict | None — mode off 时 None(默认路径键集
   不变); mode on 时 {"init_members": μ, "refresh_events": int,
   "member_advances": {i: 次数}, "dc_min_mean": float(每迭代所选成员 dc 均值)}。

### 确定性纪律

全部随机取自 engine.rng 链; 禁内部 new Generator; 禁遍历 set 决定搜索路径
(route-key set 只用于 dc/统计); 同 (seed, config) 两次运行逐位同值。

## 测试(tests/test_population.py, TDD: 先写失败断言再实现)

1. 默认关: config 无 population 键 → engine.population_mode False 且
   population_stats is None 且 population_final is None(键集断言);
   现有全套 pytest 回归 = gated 生效证据。
2. 非法组合守卫: population_mode=True 分别叠加 educate/rebalance/soft_capacity/
   soft_tw/population_crossover → ValueError; 默认关不抛。
3. 锦标赛纯成本: μ=3, obj 成本 [10, 20, 30] 的合成成员, 固定 seed → 选取 =
   抽中两成员中成本低者(单元级直接测选择函数)。
4. 锦标赛 dc: 两成员路由完全相同(低成本) + 一成员不同路由(成本略高),
   κ 取大 → 选不同路由成员; κ=0 → 选低成本成员(单元级)。
5. 池构建: tw_instance fixture, μ=4 → 全部可行; 路由集互异(固定 seed 断言);
   档案含初始播种 1 条可行。
6. 集成小跑: μ=4, max_iterations=40, tw fixture → 完成; 档案条目全部可行;
   轨迹长度 = 40; population_stats 键齐; last_solution 可行;
   同 seed 重跑 → 档案最优成本与轨迹成本序列逐位相同(确定性)。
7. 拒绝不改池: 注入全拒 acceptor(accept 恒 False, 无 _auto_calibrate),
   stagnation_limit=6000, 40 iter → 池成员与初始逐位相同(证明 destroy/repair
   链无原地修改); population_final 同。
8. 刷新路径: 全拒 acceptor + stagnation_limit=200 + max_iterations=300 →
   refresh_events ≥ 1, 运行完整不崩, 档案非空(初始播种在)。
9. Solution.apply_destroy/apply_repair 原地性确认(先读代码; 若原地修改需在
   测试 7 前先与 Hermes 沟通, 不得自行改 Solution 语义)。

## 验收清单(实现完成自检, 全部满足才提交)

- [ ] DESTROY_OPERATORS / REPAIR_OPERATORS 公共注册表零改动(与 HEAD 逐键一致;
      M1 不新增任何算子注册)
- [ ] engine.py base 路径仅 +1 行 gated 分支; __init__ 只新增 config 解析
- [ ] cx_stats 键集默认不变; population_stats/population_final 默认 None
- [ ] 随机源全部 engine.rng
- [ ] 全量 pytest 通过(记录基线数量与通过数)
- [ ] 冒烟: pr20 s42, population_mode=True(μ=8, κ=0 与 κ=0.05 各 60 iter)不崩、
      解可行、墙钟 ≈ base 同迭代量级(≤2×)
- [ ] git 提交 1 条, message 原文: "feat: 种群壳模式 (population_mode, μ成员池+
      锦标赛+多样性贡献适应度+池级刷新, gated 默认关)"

## 边界(显式禁止)

- 不 push / 不改写历史; 不动 results/ 与 docs/ 下既有 prereg/verdict/specs;
  不删既有 gated 机制; 不改既有 config 默认值; 不新增依赖; 不跑 6000iter/600s
  长实验批(冒烟 ≤60 iter 允许, 不写 results/); 不自行扩展机制(设计空间归
  Hermes); 完成报告给出: commit hash / 新增测试数 / pytest 通过数 / 冒烟输出 /
  规格出入清单(有出入先停下报告, 不自行裁决)。
