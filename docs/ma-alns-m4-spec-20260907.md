# M4 双空间罚域种群 — 实现规格 (2026-09-07)

背景: docs/ma-alns-m3-verdict-20260907.md(M3 = popcxge 3/3 超壳已合 main 80a32c8;
M4 = 双空间罚域, 唯一未验的 FIS 形态)。纪律: 项目根 CLAUDE.md。只做实现/测试/
本地提交; 实验批与判定归 Hermes(协议 = docs/ma-alns-m4-prereg-20260907.md)。
仓库: D:\mdvrptw, 分支 feat/axis2-m4m5, HEAD 80a32c8, 工作区干净。

## 设计决策(Hermes 定, 零自由空间)

M3 形态 popcxge(壳 + partition 重组 + 改进门 + 教育)在 6000iter 全实例超壳, 但
距 pyvrp-600s 锚仍 2.5-4.5pp。单轨迹罚域史(softtw/soft_capacity)判"无穿越", 但
那是**单轨迹+可行精英档案**语义; HGS 的罚域是**种群共进化**(不可行成员以罚后
fitness 参与选择/替换/教育, 不靠单轨迹穿越)。M4 = 在 popcxge 形态上允许
**TW 罚域成员共存**(容量/车辆数仍硬, 镜像 softtw 的"TW 转罚"边界)。

关键设计约束(吸收教训):
1. **λ 静态不自适应**(softtw 教训: λ 管理路径依赖 + 量纲病态) — λ = mult ×
   池内初始最优可行成本 / max(1, 初始不可行候选的 TW excess 中位数); mult 为
   预注册臂变量, 静态贯穿全程。
2. **接受门 = 罚后 fitness 严格改进**(确定性, 不走 SA; 与 M3 improve-gate 同构,
   只是比较量从真实成本换成罚后成本, 且不再要求子代可行)。
3. 教育仍是转化组件(burst 破坏 + greedy_cost_tw 重插只收 TW 可行位 → 教育天然
   向可行域修复); 教育后比较也用罚后成本。
4. 档案仍只收可行解(纯净性不变); 驱逐刷新源 = 档案可行精英(M1 语义不变) —
   保证池里永远有可行精英回流。
5. TW 可行化链在罚域模式**跳过**(否则不可行子代永远活不到评估); 容量/车辆数/
   覆盖兜底保留。LS 只精化可行解(违规解 RVND 空转, 镜像 legacy soft_tw 分支)。

## config 键(默认 = popcxge 行为逐位不变)

- `population_penalty_mode`: "none" | "tw" (默认 "none")
- `population_penalty_mult`: float 默认 1.0 (λ0 乘子; 臂变量)
守卫: population_penalty_mode="tw" 且 population_mode=False → ValueError;
"tw" 与 (soft_capacity 或 soft_tw 或 educate_mode≠none legacy) → ValueError;
"tw" 要求 population_cx_gate="improve" 且 population_cx_mode="partition"(M4 只
测 popcxge 形态上的罚域; 其它组合未测禁开); 默认全不抛。

## 机制语义(全部在 _run_population_mode 内扩展)

1. **λ0 校准**(run 开头, 池构建后): best_feas = 池内最优可行成员成本; 初始池中
   不可行成员(10 次重试全败的兜底候选)的 _tw_excess 集合, excess_med = 中位数
   (无不可行成员时取 1.0); λ = population_penalty_mult × best_feas /
   max(1.0, excess_med)。成员对象扩展存 (obj, penalized_cost, sol); 可行成员
   penalized = 真实成本; 不可行成员 penalized = cost + λ×_tw_excess(sol)。
2. **罚后接受门**(tw 模式, 替换 M3 的 pool_gate_improve 调用点): 模块级纯函数
   pool_penalty_gate(child_pen, m_pen) -> bool = child_pen < m_pen − 1e-9。
   可行性不再进门(子代可行与否都按罚后成本比); 门不过 → 回退 destroy/repair
   (M3 语义不变); 门过 → 替换 m*(成员可变为不可行)。
3. **fitness 语义**(锦标赛/驱逐/池级刷新选最劣): tw 模式统一用 penalized cost;
   κ=0 臂(M4 全臂)即纯罚后成本。dc/fitness 纯函数签名扩展传 penalized 列表。
4. **可行化链分流**: tw 模式下, destroy/repair 与 cx 子代的公共链 =
   repair_unassigned + 容量/车辆数兜底, **跳过 TW 强制**; LS(周期与 ls_on_accept)
   仅在子代可行时执行(TW 违规子代跳过, 镜像 legacy soft_tw 分支); 教育(如有)
   照常执行(burst + TW-aware repair, 教育后子代可能转可行)。
   实现提示: _postprocess_after_repair 内部是 repair_unassigned →
   _enforce_capacity/vehicle → _enforce_tw 的链; tw 模式需一条不加 _enforce_tw
   的变体 — 禁止改 _postprocess_after_repair 本体(base/legacy 路径), 在
   _run_population_mode 内按需调链内各步(读源码后以最小重复实现, 注释说明)。
5. **档案/score/停滞/驱逐/轨迹**: 档案只收可行(不变); score 语义 = base admission
   (added 2 / accepted 1 / 拒 0 — 此处 accepted = 罚门过); 池级停滞 = 档案 add
   驱动(不变); 成员级停滞驱逐 = 档案可行精英(不变, 不可行尸体照常被驱逐 —
   与 M1 驱逐机制天然兼容); 轨迹 outcome.cost 记真实成本, 增记 penalized_cost
   与 tw_excess(诊断)。
6. **统计键**(tw 模式存在, 累计): penalty_member_iters(选中成员不可行的迭代数),
   penalty_infeasible_final(population_final 中不可行成员数), lambda0(float)。
7. 确定性: 全部随机 engine.rng; λ/罚后成本纯计算。

## 测试(tests/test_population.py 追加, TDD 先红)

1. pool_penalty_gate 纯函数三用例(child 更优可过/劣拒/等值拒)。
2. λ0 校准单测: 构造含不可行成员的池 → λ = mult×best_feas/max(1, excess 中位)
   (手工数值断言)。
3. 守卫: tw+mode 关/gate 非 improve/mode 非 partition/legacy soft 组合 →
   ValueError; 默认全不抛且 population_stats 无 penalty 键。
4. 集成(tw fixture, μ=4, 40 iter, tw 模式): 完成; 档案全可行(纯净门禁); 统计键
   齐; 同 seed 重跑逐位同。构造断言: 至少观察到罚门路径生效
   (penalty_member_iters 或不可行成员出现过 — 若 fixture 上无法自然出现不可行
   子代, 用白盒注入不可行成员断言其按罚后成本参与锦标赛不被驱逐前参与选择)。
5. 默认回归: penalty_mode 缺省 none → popcxge 逐位一致(键集断言 + 全套 pytest)。
6. 冒烟: pr20 s42 60 iter, tw 模式(mult=1): 不崩、档案全可行、墙钟 ≤ 8s;
   临时脚本跑完即删。

## 验收清单(自检, 全满足才提交)

- [ ] 公共注册表零改动; 全部扩展在 _run_population_mode + 模块级纯函数;
      _postprocess_after_repair 本体与 run()/legacy 零改动
- [ ] penalty_mode=none → popcxge(M3)逐位不变(全套 pytest + 键集断言)
- [ ] 随机全 engine.rng; 新统计键累计且仅 tw 模式存在
- [ ] 全量 pytest 绿(基线 195 + 新增)
- [ ] 唯一提交, message 原文: "feat: 双空间罚域种群 (population_penalty_mode tw,
      罚后成本门+λ0 静态校准, popcxge 形态扩展, gated 默认关)"

## 边界(硬性)

不 push; 不改 docs/; 不动 results/; 不跑 6000iter/600s 长实验; 不删既有机制;
不改既有 config 默认值; 不新增依赖; 不自行扩展设计(歧义停下报告)。报告模板:
commit hash / 新增测试数 / pytest / 冒烟 / 验收清单 / 出入清单。
