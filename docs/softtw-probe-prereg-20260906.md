# TW 罚域探针 (softtw) 预注册 + 实现规格 (2026-09-06)

衔接: 探索史最后一个未测机制格 — 全部 09-04/05/06 机制 (软容量/连通性/
段级/SP/教育/交叉/重平衡) 均保持 TW 硬约束; 引擎无 TW 软罚 (仅容量软罚
soft_capacity, 镜像对象)。动机: 09-04 涟漪实证 (单客户修复前 300 位不可
行 = TW 协同传播, 剩余 gap 需"多客户协同+重定时") + pr16 int/float 边界
解 + 结构分析 (pyvrp/HGS 的 time-warp 类机制我们无等价物)。H-F 探针
(09-05) 否的是**容量**罚域单轨迹; TW 罚域正交未测。

## 假设

H-TW: TW 软罚轨迹 (接受标准 = cost + λ×Σexcess, 周期硬修复播种) 能穿越
TW 硬锁盆地 → 6000iter best 配对差 vs base 同实例 3/3 符号一致且 >
base 散布 (pr15 89.3 / pr20 80.7 / pr16 ~68)。否定 = 符号混或幅度不足
→ TW 罚域单轨迹无穿越, 探索史机制格全部填完, 收束。

## 机制实现规格 (CC 执行, 镜像 soft_capacity 全套分支)

config (默认关, 默认行为逐位不变):
- `soft_tw`: bool False
- `soft_tw_lambda_mult`: float 1.0 → λ0 = mult × obj0 / num_customers
  (每客户平均成本 ≈ 每单位时间违规的罚金量级; 播种失败 ×1.5 上调 /
  成功 ÷1.5 回落, 上限 λ0×100, 完全镜像 soft_capacity 的 λ 管理)
- `soft_tw_repair_interval`: int 100 (>0 周期修复播种; 镜像 soft_repair_
  interval 语义)

engine 改动 (src/alns/engine.py, 逐处镜像 soft_capacity 分支):
1. __init__: 读 config → self.soft_tw / soft_tw_lambda_mult /
   soft_tw_repair_interval; 初始化 λ 字段 (self._soft_tw_lam0/_lam);
   self.soft_tw_stats = {"violation_iters": 0, "repair_events": 0,
   "repair_feasible": 0, "seed_adopted": 0, "lam": 0.0}
2. run() 初始: if soft_tw: lam0 = mult×obj0/n; 设 λ。
3. 主循环 LS 分支 (do_ls): if soft_tw and new_sol.tw_violations():
   跳过 LS (违规解上 RVND 空转) — 镜像 soft_capacity 的 not feasible。
4. 接受目标: new_obj = _soft_tw_penalize(new_sol, new_obj) if soft_tw
   — 罚 = λ × Σ(v["excess"] for v in tw_violations()); violation_iters+=1。
5. ls_on_accept 分支: 同样跳过违规解 LS + 罚后目标。
6. 周期播种: if soft_tw and interval>0 and iter%interval==0:
   _soft_tw_seed(current_sol): 修复 = self._enforce_tw(current_sol.copy())
   (镜像 _soft_seed 的采纳规则: 修复解可行且 (轨迹违规 或 修复真实成本
   < current 罚后目标) → 锚回 current_sol; repair_events/feasible/
   seed_adopted 计数; 采纳时 stagnation_count=0)。
7. _soft_tw_penalize(sol, obj): obj 深拷贝改 [0] += λ×Σexcess。
8. 档案规则不动 (只收可行 — 已是全局)。
9. cx_stats 等既有键集不动; soft_tw 默认 False 时全部新分支不可达。

测试 (tests/test_soft_tw.py): ① 默认 none: engine 全套既有测试过即可
(逐位不变)。② _soft_tw_penalize 罚计算单测 (构造带违规的合成解, 断言
罚 = λ×Σexcess)。③ soft_tw=True 在 tw 小实例跑 60 iter 不崩、stats 键
存在、档案只收可行 (违规解不入档)。④ 周期播种: 注入违规轨迹 →
interval 触发后 repair_feasible/seed_adopted 计数语义正确。⑤ config
缺省 == soft_tw False。

## 实验 (并行执行, 6000iter 确定性 → 并行安全)

- 臂: softtw (soft_tw True, interval 100) vs base (既有 population_ab
  base JSON, 不重跑)。
- 实例 × seed: pr15/pr20/pr16 × s42-44 = 9 组新跑。
- **执行**: 每组独立 python 进程, xargs -P9 并行 (固定迭代 → 结果与
  串行逐位一致, 引擎确定性已验证); 每进程 OMP_NUM_THREADS=1 等限单
  线程。单组 200-370s → 全批 ~8-10 min 墙钟。
- runner: experiments/benchmark_softtw_ab.py (模式同 benchmark_educate_
  ab: engine 直连, 6000iter, 输出 best + soft_tw_stats + 结构指标 +
  dump)。

## 判定 (两级, 硬屏障)

第 1 级 机制生效: softtw 组 violation_iters > 0 (违规轨迹真实发生) 且
repair_feasible ≥ 1 (周期修复至少部分成功)。否定 (轨迹从未违规 = λ 过
大/修复即时; 或修复全败) → 记录 λ 病态诊断, 关闭。
第 2 级 质量: pr15/pr20/pr16 各自 3 seed 配对差 (softtw − base) 符号
一致且 |均值| > base 散布 → 支持; 否则否定 → TW 罚域单轨迹无穿越,
探索史收束 (论文转轨)。

## 产出

判定 docs/softtw-probe-verdict-20260906.md。结果 results/softtw_ab/。
