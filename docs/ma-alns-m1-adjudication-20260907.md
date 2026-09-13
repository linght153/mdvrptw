# M1 规格出入裁定 (2026-09-07) — CC 实现停摆点裁决

CC(proc_896b56145242)按 docs/ma-alns-m1-spec-20260907.md 实现完毕, 全量测试通过,
但冒烟发现 pop κ=0.05 的 last_solution(池末端解)不可行(8 成员中 6 个终态带 TW
违规), 停摆请求裁定。根因(CC 定位 + Hermes 复核一致): pr20 nearest 初始解经
_enforce_feasibility 仍有 8 处 TW 违规(base 同款起点); μ=1 池退化与 base 一致自愈,
μ=8 时锦标赛共享迭代 → 不可行成员被饿死(池级停滞被其它成员入档清零, 永不刷新),
且"接受不可行子代"让不可行状态可延续。设计文档 §3 明示"每成员 = 完整可行解",
规格第 5.1 条"镜像 base init"与 5.5c 条"镜像 base 接受"在硬 TW 实例上相互冲突 —
裁决以设计语义为准。

## 裁定(全部生效, CC 照此改代码后完成唯一一次提交)

1. **成员可行性策略(替代规格 5.5c 的"接受即替换")**: 池成员只允许可行解。
   - 接受判定拆两级: SA 接受(acceptor.accept) AND child.is_feasible() 同时成立
     → 替换该成员; SA 接受但子代不可行 → 成员不变, 该迭代按"拒绝"记账
     (score=0, outcome.accepted=False/feasible=False, selector.update 照常,
      pool_stagnation += 1)。
   - 决策点抽成纯函数供测试锁策略: src/alns/engine.py 模块级
     `pool_accept_replace(sa_accepted: bool, child_feasible: bool) -> bool`
     = sa_accepted and child_feasible。实现用该函数, 禁内联绕过。
2. **init / 刷新重建的重试兜底**: build_initial_solution → _enforce_feasibility 后
   仍不可行 → 重新 build(engine.rng 链取源), 至多 10 次; 10 次全败取成本最低的
   候选入池(代码注释说明: pr11 类残余违规实例不得阻塞运行; 该成员仍按裁定 1
   不可被替换链选中? — 否: 该成员可被锦标赛选中并演化, 其可行子代一旦出现即
   替换它; 这是与 base 同构的自愈路径, 允许)。重试常数 10 写死, 不加 config 键。
3. **组合守卫补一条**: population_mode=True 且 stagnation_source="current" →
   ValueError(M1 池级停滞只实现 archive 语义, 镜像 base 的 current 语义未测禁开)。
4. **出入 2/3/5 采纳 CC 解读**: member_advances = 该成员经可行子代替换的次数
   (refresh 不计); refresh 时 acceptor.reset() 无条件、archive.clear() 仅
   restart_clear_archive=True(与 base 377-384 行一致); conftest 不存在 →
   test_population.py 本地定义 6 客户 2 车场 TW fixture(语义与规格一致)。
5. **测试补充**: (a) pool_accept_replace 纯函数三用例 (T,T)=True/(T,F)=False/
   (F,T)=False; (b) 既有"拒绝不改池"测试(规格第 8 节测试 7)保留, 另加: 注入
   accept 恒真 acceptor 时, 若构造使子代不可行(tw fixture 上难以自然构造 →
   用 pool_accept_replace 单测锁定策略即可, 集成层不强造); (c) 冒烟断言升级:
   pop κ=0.05 60 iter 后 last_solution.is_feasible() 必须为 True。
6. **提交不变**: 上述改动完成后仍为唯一一次提交, message 原文:
   "feat: 种群壳模式 (population_mode, μ成员池+锦标赛+多样性贡献适应度+池级刷新,
   gated 默认关)"。报告模板不变(commit hash / 测试数 / pytest / 冒烟 /
   出入清单), 追加本裁定逐条落实情况。
