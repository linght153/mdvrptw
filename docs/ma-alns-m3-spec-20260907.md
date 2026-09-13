# M3 池内重组预算竞争修复 — 实现规格 (2026-09-07)

背景: docs/ma-alns-m2-verdict-20260907.md(M3 立项方向: part 臂 + 预算竞争修复)。
纪律: 项目根 CLAUDE.md。只做实现/测试/本地提交; 实验批与判定归 Hermes
(协议 = docs/ma-alns-m3-prereg-20260907.md)。仓库: D:\mdvrptw, 分支
feat/axis2-memetic-alns, HEAD 749fa05, 工作区必须干净。

## 设计决策(Hermes 定, 零自由空间)

M2 判定: part 重组 @rate0.2 在 pr20 负效(+38.8 vs pop), 归因 = cx 迭代占 20%
预算且子代多不优于 destroy/repair 路径产物。M3 用**改进门**消灭预算竞争:
cx 子代经(可选)教育后, 若成本 < m* 才替换(跳过本迭代 destroy/repair);
否则**回退执行正常 destroy/repair 迭代**(cx 尝试变成纯增量, 不占迭代预算)。
教育 = 池内版破坏性教育(partition 子代 burst 破坏-重建 + 封顶 LS), 镜像 legacy
_educate_child 机制, 首次在池内检验。rate 维度被 gate 吸收: 门控下 rate 只影响
cx 尝试次数, 不再决定迭代去向 — gate 正效则无需 rate 扫(预注册声明)。

## config 键(全部默认 = M2 行为逐位不变)

- `population_cx_gate`: "none" | "improve" (默认 "none" = M2 的 SA+可行两级门)
- `population_cx_educate`: bool 默认 False
- `population_educate_burst_ratio`: float 默认 0.20
- `population_educate_ls_budget`: int 默认 50(legacy 300 过重; 池内按次封顶)
- 复用既有: population_cx_mode("partition") / population_cx_rate(0.2) /
  population_cx_inherit_prob(0.6, partition 不用)

守卫(__init__, 追加到既有组合守卫):
- population_cx_gate="improve" 且 population_cx_mode ≠ "partition" → ValueError
  (gate 只与车场簇重组配对测过)
- population_cx_educate=True 且 (population_cx_gate ≠ "improve" 或
  population_cx_mode ≠ "partition") → ValueError
- population_cx_gate ≠ "none" 且 population_mode=False → ValueError(并入既有)

## 机制语义

### gate="improve" 主循环改造(_run_population_mode 的 cx 分支; M2 路径为底)

M2 cx 触发 → partition_crossover 生成 child → _postprocess_after_repair →
(SA 接受 & 可行)门。M3 改判据并加回退:

1. cx 触发(rate 命中, mode=partition): 双亲 = m* + 随机异键集成员(12 次尝试,
   M2 同款) → child = partition_crossover(m*, mB, rng)。
2. education(population_cx_educate=True): child 先教育再进门 —
   a. 从 child 随机移除 population_educate_burst_ratio × n 客户(random_removal
      语义, 参照 legacy _educate_child 1851-1910 行的实现结构, 但用新键);
   b. 既有 repair(greedy_cost_tw, TW 实例)重插;
   c. local_search_fn(child, max_iterations=population_educate_ls_budget, rng);
   d. _enforce_feasibility 兜底 + calculate_objectives;
   e. 统计: educate_attempts += 1; 教育后成本 < 教育前 → educate_success += 1,
      educate_delta_sum += (educate 前成本 − 教育后成本)。
3. 门: gate="improve" → accepted = child.is_feasible() AND child_obj[0] <
   m_obj[0] − 1e-9(纯成本严格改进, 不走 SA); gate="none" → 沿用 M2
   pool_accept_replace(sa_accepted, feasible) 语义。
4. **门过**: 替换 m*(member_advances += 1, member_reset), 本迭代结束 —
   不执行 destroy/repair、不更新 selector(cx 惯例), 档案/score/停滞/轨迹记账
   与 M2 cx 路径一致; cx_gate_success += 1。
5. **门不过(gate=improve)**: cx_gate_fail += 1; **回退**: 本迭代继续走正常
   destroy/repair 路径(破坏/修复/LS/接受/selector 更新/档案 — 与 M1 非 cx 迭代
   逐位一致), 即该迭代当作 cx 没发生过(仅多付了一次 cx 尝试的墙钟)。
   gate="none" 门不过 → 维持 M2 行为(迭代结束, 成员不变)。
6. 统计键(population_stats, 仅相应 mode 开启时存在, 累计):
   - gate="improve": cx_gate_success / cx_gate_fail(int)
   - educate=True: educate_attempts / educate_success(int), educate_delta_sum
     (float 累计)
   - 既有 M2 键(cx_attempts/cx_children/cx_duplicates/cx_novel_route_keys)语义
     不变: cx_attempts = 触发且双亲齐备; cx_children = 子代生成成功
     (gate/educate 不影响这两个键的计数口径)。

### 教育实现约束

- 全部随机 engine.rng; burst 移除与 repair 复用既有算子函数(分支内 import,
  禁注册公共表); 教育只作用于 cx child, 不作用于 destroy/repair 产物;
- 教育后必须可行(公共链兜底); 教育本身不更新 selector;
- 墙钟约束: 单次教育 ≤ 1 次 destroy/repair + 1 次 LS(预算 50)的量级 —
  不得引入每迭代全量重优化(镜像 legacy 验收 2b)。

## 测试(tests/test_population.py 追加, TDD 先红)

1. gate 纯函数化: 模块级 pool_gate_improve(child_cost, m_cost, feasible) ->
   bool(feasible AND child < m − 1e-9); 三用例锁定; 实现必须经它, 禁内联。
2. gate="improve" 集成(tw fixture, μ=4, 40 iter, rate=1.0 强制): 门过替换/门不过
   回退路径都执行过(用可断言统计: cx_gate_success + cx_gate_fail ≥ 1 且
   cx_gate_fail 时该迭代确有 destroy/repair 痕迹 — 以 selector 权重/usage 计数
   或 trajectory 该迭代 action 非空断言); 档案可行; last_solution 可行;
   同 seed 重跑逐位同(含全部统计)。
3. education 集成(同上 + educate=True): educate_attempts ≥ 1; 教育后解可行;
   educate_success ≤ attempts; 同 seed 重跑逐位同。
4. 守卫: gate="improve"+mode=route → ValueError; educate=True+gate="none" →
   ValueError; educate=True+mode=route → ValueError; 默认 none/none/False 全不抛
   且 population_stats 无新键(键集断言)。
5. 默认回归: population_cx_gate 缺省 = "none"、educate 缺省 False → M2 行为
   逐位不变(键集断言 + 全套 pytest)。
6. 冒烟: pr20 s42 60 iter × (popcxg, popcxge): 不崩、last_solution 可行、
   cx_attempts ≥ 1、popcxg 墙钟 ≤ 6s、popcxge 墙钟 ≤ 12s(教育 12 次 × 封顶
   LS50 的量级); 临时脚本跑完即删。

## 验收清单(自检, 全满足才提交)

- [ ] 公共注册表零改动; 教育/门逻辑全部在 _run_population_mode + 模块级纯函数
- [ ] run()/base/legacy 分支与 M2 population 路径(gate="none", educate=False)
      零行为变化 — 全套 pytest + 键集断言
- [ ] 随机全 engine.rng
- [ ] 新统计键累计、仅对应 mode 开启时存在
- [ ] 全量 pytest 绿(基线 187 + 新增)
- [ ] 唯一提交, message 原文: "feat: 池内重组改进门+教育 (population_cx_gate
      improve + population_cx_educate, 预算竞争修复, gated 默认关)"

## 边界(硬性)

不 push; 不改 docs/; 不动 results/; 不跑 6000iter/600s 长实验; 不删既有机制;
不改既有 config 默认值; 不新增依赖; 不自行扩展设计(歧义停下报告)。报告模板:
commit hash / 新增测试数 / pytest / 冒烟 / 验收清单 / 出入清单。
