# 结构生成三探针预注册简案 (F/E/P, 2026-09-05)

衔接: "结构生成稀缺" 平台定位 (600s 9.28% pr20)。三机制级候选方向在投入
种群工程前先做成分探针 — 全部 6000 iter 固定迭代 × 3 seeds (42/43/44) ×
pr15/pr20 (宽窗紧车, 结构生成主战场), base/cx 参照 (s42-44) 既有不重跑。

## F: 罚域盆地存在性 (FIS 种群化前提)

假设 H-F: 罚域 (容量软罚) 轨迹演化 6000 iter 后, 末端解经硬修复
(_soft_repair_hard) 可回到 **低于 base best** 的可行解 → 不可行域藏着
单轨迹可行搜索到不了的盆地 → FIS 种群化 (HGSADC 式多轨迹+罚函数) 有原料。
09-04 软容量否定只测了 300/2000 iter 播种修复口径; 纯罚 (不播种锚回,
soft_repair_interval=0) × 6000 iter × 末端深度修复是未测组合。

- 臂: softfix (soft_capacity + interval=0; repair 罚入成本, λ0 固定)
- 判定: 支持 = pr15 且 pr20 修复成本配对均值 ≤ base −2pp 且符号一致;
  否定 = 修复不可行或 ≥ base (罚域无货, FIS 方向关闭 — 不再追文献形态)。
- 参照: base 6000 iter 3-seed (pr15 2674.88/2585.62/2600.94; pr20
  3307.93/3244.55/3325.29)。

## E: 在线教育生成器 (诊断, 不单独判定)

交叉后代 (route_copy + RVND-教育链) 的档案外新路由键产率 (novel_route_keys
/ children) — 离线教育列实验对照: 292 列产 13/20 intra 新列, 破界列
~1/840。产率数据支持/反对 "在线教育能规模化生成新结构" 机制叙述, 并入
P 判定。

## P: 互补双亲交叉 (cx 失效归因的修复候选)

假设 H-P: 09-04 cx 无效应归因 = 双亲全客户覆盖冲突 (B 路由全冲突 → 后代
≈ A 子集)。第二父代改选 "与 A 路由 Jaccard 距离最大且成本 ≤ A×1.10" 的
档案成员 (离线教育列实验的选择在线化) → 后代结构新颖度高 → 教育质量翻盘。

- 臂: pdiv = cx 同配置 (archive_diversity+crossover rate 0.2 inherit 0.6)
  + crossover_pair="diverse"; 对照 = cx 臂 (random, 既有 s42-44)。
- 判定: 支持 = pr15 且 pr20 pdiv ≤ cx −2pp 且 ≤ base −2pp (符号一致);
  否定 = |pdiv − cx| ≤ 1pp 或符号混 → 双亲互补不是失效主因, 交叉教育线
  关闭。

## 执行

12 炮串行独立进程, 结果 population_ab/{inst}_s{seed}_{softfix|pdiv}.json
(softfix 含 repaired/traj 字段; pdiv 含 cx_stats.novel_route_keys)。
判定文档 docs/structure-gen-probes-verdict-20260905.md。
