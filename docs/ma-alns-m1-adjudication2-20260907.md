# M1 裁定 2 — 成员级停滞刷新(池健康度修复) + 冒烟口径纠正 (2026-09-07)

CC 二轮(proc_e5b439d52f1b)按裁定 1 实现后再次停摆: 裁定 6 的冒烟断言
"κ=0.05 60 iter last_solution 必须可行"在 pr20/μ=8 下不可达。CC 证据(采信,
Hermes 复核逻辑自洽): ①pr20 初始解经 _enforce_feasibility 永不可行(独立验证
nearest 0/10、random 0/60 全败)→ 池必然全员不可行起步, 只能演化自愈;
②μ=1/base 单轨迹 60 iter 可自愈(每迭代都在治伤), μ=8 时锦标赛让首个自愈的
低成本成员垄断选择, 其余成员被冷温饿死(不可行成本 ~5790 vs 可行 ~4050,
κ×best×dc ≤ ~200 折扣远不能补偿 → 永不入选 → 永不自愈); ③池级全局刷新
(recovery_threshold = max(30, stagnation//3) = 2000)在 6000iter 内都难触发,
且即便触发也只换 1 个成员 — 尸体积压是结构性病理, 非冒烟门过严。
裁决 1 的"成员只允许可行解"因此无法靠 init+接受两级守住, 必须加第三级: 驱逐。

## 裁定(全部生效, 替代/补充裁定 1 的相应条款)

1. **成员级停滞刷新(新增机制, 驱逐路径)**: 每成员带 member_stagnation 计数
   (连续"未成功替换"的迭代数; 成功替换 = 可行子代替换成员 或 自身被刷新 → 清零)。
   - 触发: 每迭代开头(时间/池停滞检查之后, 锦标赛之前)扫描全体成员, 凡
     member_stagnation ≥ member_refresh_threshold 的成员全部刷新。
   - 阈值(不加 config 键, 公式写死): member_refresh_threshold =
     max(50, stagnation_limit // 30)。6000iter BASE 配置(stagnation_limit=6000)
     → 200。
   - 刷新动作: 新成员 ← 档案精英副本: 调 self._select_parent()(既有 crowding
     选择, 无 rng 消耗)得 entry, 非 None → _reconstruct_solution(entry).copy();
     None(档案空)→ _build_pool_member()(裁定 1 的 10 次重试逻辑)兜底。
     替换后该成员 obj 重算(calculate_objectives), member_stagnation=0,
     population_stats["member_refresh_events"] += 1。
   - 不重置 acceptor; 不改 pool_stagnation(全局池停滞仍只由 archive add 驱动,
     语义不变); 不动其它成员。
   - 设计意图: 尸体(不可行/长期停滞成员)被档案可行精英驱逐, 池保持"活成员
     为主" — 这是 M1 作为 M2 重组地基的健康度前提; 档案精英经 crowding 选择
     天然偏稀疏区, 防止全池同化为档案最优附近(池重复度由预注册的
     pool_final_spread 字段观测)。
2. **冒烟口径纠正(裁定 1 第 6 条作废重写)**: "60 iter 后 last_solution 必须
   可行"不是确定性保证性质(依赖自愈+刷新时序), 降级为工程自检口径:
   冒烟硬门槛 = ①不崩 ②archive best 可行 ③population_final 中可行成员 ≥1
   ④墙钟 ≤ 2×base。另加 250 iter 诊断跑(κ=0.05, s42): 报告 last_solution
   可行性、池可行成员数、member_refresh_events(≥200 iter 应触发驱逐, 只报告
   不设门槛)。质量与池健康度的真正门槛在预注册 6000iter 批(见 prereg 增补)。
3. **测试补充(tests/test_population.py)**:
   - member_refresh_threshold 公式单测(max(50, stagnation//30) 三值);
   - 驱逐集成测: 构造 archive ≥2 可行条目的引擎, 白盒把某成员
     member_stagnation 拨到 ≥ 阈值 → 跑 1 迭代 → 断言该成员被档案精英替换
     (obj 等于某档案条目 obj)、member_stagnation=0、member_refresh_events=1;
   - 档案空兜底测: archive 清空 + 停滞成员 → 刷新走 _build_pool_member 不崩。
4. **统计键**: population_stats 增 "member_refresh_events": int(默认关 None,
   键集不变)。既有 refresh_events(池级全局)保留, 语义不变。
5. 其余(裁定 1 的第 1/2/3/4 条与第 5 条既有测试)维持原样, 不重开。
6. **提交不变**: 唯一一次提交, message 原文: "feat: 种群壳模式 (population_mode,
   μ成员池+锦标赛+多样性贡献适应度+池级刷新, gated 默认关)"。
   报告模板: commit hash / 测试数 / pytest / 冒烟与 250iter 诊断输出 /
   裁定 1+2 逐条落实 / 出入清单。
