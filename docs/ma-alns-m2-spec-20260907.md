# M2 池内重组实现规格 — 给 Claude Code (2026-09-07)

背景: docs/ma-alns-design-20260907.md(M2 = 判别实验)、docs/ma-alns-m1-verdict-20260907.md
(M1 通过: 壳正效; 本规格吸收其 §3 观察)。纪律: 项目根 CLAUDE.md。只做实现/测试/
本地提交; 实验批与判定归 Hermes(协议 = docs/ma-alns-m2-prereg-20260907.md)。
仓库: D:\mdvrptw, 分支 feat/axis2-memetic-alns, HEAD bbe8b2d, 工作区必须干净。

## 设计决策(Hermes 定, 零自由空间)

**判别问题**: MDVRPTW 上池内重组的 load-bearing 单位 = 路由级(A 臂)还是车场
分配簇(B 臂)。A 臂 = 既有 route_copy_crossover(逐路由继承); B 臂 = 新车场簇
继承算子 partition_crossover(整车场路由组继承)。

**形态**: M1 种群壳内, 每迭代以 population_cx_rate 概率触发重组(池 ≥2 恒真):
- 双亲: m* = 当轮锦标赛选中成员 + m' = 池内随机另一成员(路由键集与 m* 不同,
  12 次尝试, 参照 engine._crossover_child 1942-1951 行 legacy 语义); 键集用
  ParetoArchive._route_keys 约定。
- 子代 → _postprocess_after_repair(公共可行化链)→ SA 接受时 ls_on_accept 精化
  → pool_accept_replace(sa_accepted, 精化后可行)才替换 m*(与 M1 完全同链)。
- cx 触发迭代不更新 selector 权重(legacy cx_used 惯例); 停滞/score/轨迹记账
  与 M1 相同。
- 未触发 → M1 原 destroy/repair 路径逐位不变。

## 机制语义

### config 键(全部默认 = M1 行为逐位不变; 独立新键, 不碰 legacy crossover 键)

- `population_cx_mode`: "none" | "route" | "partition" (默认 "none")
- `population_cx_rate`: float 默认 0.2 (mode ≠ none 才生效; 每迭代触发概率)
- `population_cx_inherit_prob`: float 默认 0.6 (仅 route 臂传给
  route_copy_crossover 的 inherit_prob; partition 臂不用)

守卫(__init__, 参照 M1 组合守卫风格): population_cx_mode ≠ "none" 且
population_mode=False → ValueError; population_cx_mode ≠ "none" 与
(population_crossover 或 educate_mode≠none 或 rebalance_mode≠none 或
soft_capacity 或 soft_tw 或 stagnation_source=current)→ ValueError(M2 未测禁开)。

### A 臂: route 模式

调用既有 route_copy_crossover(sol_a=m*, sol_b=m', self.rng,
self.population_cx_inherit_prob)(src/operators/crossover.py, 零改动)。

### B 臂: partition 模式 — 新算子(放 src/operators/crossover.py 内, 函数导出,
**禁止注册任何公共算子表**)

`def partition_crossover(a_sol, b_sol, rng, ...) -> Solution`:
1. 对车场 d 按固定序 0..nd−1 逐车场掷硬币: rng.random() < 0.5 → 父本 A 否则 B
   (每车场独立; 硬币必须每个车场都抽 — rng 消耗与父本结构无关, 确定性)。
2. 继承: 该车场在所选父本的全部路由逐条复制入子代(路由序保持); 已覆盖客户
   (先前车场已复制)跳过, 跳过的客户进修复池; 复制不拆路由(整条进, 过滤后
   非空仍整条保留 — 过滤重复客户后的子路由保持原序)。
3. 覆盖补全: 修复池客户用既有 repair(与 route_copy_crossover 完全同款:
   TW 实例 greedy_cost_tw_insertion, 否则 greedy_cost_insertion)重插 —
   允许跨车场(与 route 臂 repair 语义公平对照)。
4. _sync_assigned_depots(child)(route_copy 同款收尾)。
5. 可行性: 复制路由 = 可行父本子集 → 容量/TW/车辆数继承可行; repair 后经
   engine 公共链兜底。
6. 返回完整解(可能带重复跳过后的修复池, 无未覆盖)。

设计注(实现注释写明): 修复池大小 = 跨父本重复客户数 — 两父本分配结构越接近,
重复越少(继承纯度越高); A/B 车场偏好相反的父本组合重复多 → cx_duplicates
累计计数如实记录, verdict 用。

### 统计(population_stats 键, 仅 population_cx_mode ≠ none 时存在 — 累计整数,
教训: 机制过程统计必须累计, 禁只留最近值)

- cx_attempts: 触发且双亲齐备的次数(rate 命中 + 找到异键集 m')
- cx_children: 子代成功生成的次数(双亲 None 回退不算)
- cx_duplicates: 累计修复池客户数(partition 臂; route 臂恒 0? — 不, route_copy
  内部 pool 不可见; 仅 partition 臂记录, route 臂该键写 0)
- cx_novel_route_keys: 子代路由键 − 池全体成员路由键并集 的新键累计数
  (镜像 legacy 1963-1968 的 novel 计数语义, 池成员键集 = pool_keys 快照)

M1 键(init_members/refresh_events/member_refresh_events/member_advances/
dc_min_mean)与语义零改动。

## 测试(tests/test_population.py 内追加, TDD 先红)

1. partition_crossover 单元: tw fixture 上手工构造 A(客户全 depot0 布局)、
   B(depot1 布局)两可行解 → 子代覆盖全集且可行; 固定 seed 确定性(两次同值)。
2. 重复计数: 构造已知 k 个跨父本重复客户 → cx_duplicates 语义正确(算子层
   暴露修复池长度即可测)。
3. 继承纯度: A 全 depot0 + B 全 depot1 → 硬币全偏 A 时(用固定 seed 或注入
   选择?)子代 depot0 路由 = A 的 depot0 路由逐条同序(过滤后)— 直接断言
   子代 depot0 路由条数与成员 ⊆ A 的 depot0 客户集。硬币随机 → 断言每个
   车场的客户集 ⊆ (A_d ∪ B_d 中该车场路由客户并集 ∪ 修复池) — 用轻断言:
   子代每车场路由数 ≤ 两父本该车场路由数之和。
4. 守卫: population_cx_mode=route 且 population_mode=False → ValueError;
   与 legacy population_crossover/educate/soft 各组合 → ValueError;
   默认 none 不抛且 population_stats 无 cx 键(键集断言)。
5. 集成: pop+route 与 pop+partition 各 40 iter(tw fixture, μ=4): 完成、档案
   可行、last_solution 可行、cx_attempts ≥ 1、cx_stats 键齐; 同 seed 重跑逐位
   同(含 population_stats 全等)。
6. 默认回归: population_cx_mode 缺省 = none → 与 M1 行为一致(键集断言 +
   全套 pytest)。
7. destroy/repair 不原地改成员(既有测试)对 cx 子代路径同样成立 — 追加:
   cx 触发但子代被拒(pool_accept_replace False)→ 池成员不变。

## 验收清单(实现者自检, 全满足才提交)

- [ ] 公共注册表(DESTROY/REPAIR_OPERATORS)零改动; partition_crossover 只函数导出
- [ ] run()/base 路径与 legacy cx/educate/rebalance/soft 分支零改动(M2 全部在
      _run_population_mode 内扩展; git diff 应显示 engine.py 改动仅限
      __init__ 新键区 + _run_population_mode 区 + crossover.py 新函数 + 测试)
- [ ] population_cx_mode 默认 "none" → M1 行为逐位不变(全套 pytest + 键集断言)
- [ ] 随机全 engine.rng; 硬币序固定(每车场必抽)
- [ ] cx 统计键累计整数、仅 mode≠none 时存在
- [ ] 全量 pytest 绿(基线 174 + 新增)
- [ ] 冒烟: pr20 s42 60 iter × (route, partition): 不崩、last_solution 可行、
      cx_attempts ≥ 1; 墙钟 ≤ 2× M1 同 iter(60iter)
- [ ] 唯一提交, message 原文: "feat: 池内重组判别 (population_cx_mode route|partition,
      路由继承 vs 车场簇继承, gated 默认关)"

## 边界(硬性)

不 push; 不改 docs/ 任何文件; 不动 results/; 不跑 6000iter/600s 长实验; 不删
既有机制; 不改既有 config 默认值; 不新增依赖; 不自行扩展设计(歧义停下报告)。
报告模板: commit hash / 新增测试数 / pytest 通过数 / 冒烟输出 / 验收清单逐项 /
出入清单。
