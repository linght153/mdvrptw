# durfix 后续执行清单 (2026-09-11 起, 供接续会话直接使用)

规格: docs/durfix-spec-20260911.md · 预注册: docs/durfix-prereg-20260911.md
实现: commits 7f914ec (引擎/核验器/测试) + f7e3068 (脚本) + 02cd291 (number-sheet §19)

## 批次进度
- [x] B1-B5 并行 (195 组): 2026-09-11 03:30 启动 `bash scripts/durfix_run_all.sh parallel`
      (B1 54 → B2 60 → B3 36 → B4 9 → B5 36; xargs -P9; 断点续跑; 预计 6-8h)
- [ ] B6 等墙钟 (校准 3 + BASE 9, 串行独占): 在 B1-B5 完成后, 机器安静时
      `bash scripts/durfix_run_all.sh serial` (或分开跑 b6/b7)
- [ ] B7 600s (pyvrp s1-10 + popcxge600 s1-10 + base600/popcxge600 s42-44, 78 组串行 ~13h)
- [ ] B8 取样 dump: B2 完成后按结果挑代表 seed; 至少每实例 1 个 (pr15/16/20 + pr04/pr06)
      **种子已定 (09-11 17:10, 各实例最优 popcxge seed)**: pr15 s4 (2481.33) · pr16 s4 (2908.19)
      · pr20 s10 (3121.95) · pr04 s44 (2899.61) · pr06 s44 (3836.70); 已挂串行衔接自动执行
      (等 DURFIX_SERIAL_DONE → b8_dump 5 组 → results/durfix/ma_routes/)
      用法: `bash scripts/durfix_b8_dump.sh pr04 42 pr06 43 pr15 <seed> ...`
      (dump 会重跑引擎复现并断言与结果 JSON 成本一致, 配置漂移即拒绝写盘)

## 批次后核验
- 全量: 扫 results/durfix/**/*.json 的 `max_route_duration_excess == 0` (运行期已经
  引擎 is_feasible 门控, 此扫描防回归) + `max_return_time` 抽查
- 路线级独立核验 (抽样): 对 B8 dump 的全部文件跑
  `python scripts/verify_solution_constraints.py <inst> <dump.json>` (须全过)
- pyvrp 侧: B7 输出含 d_col_violations / std_cost 字段 — 表内一律用 std_cost,
  native (float_cost) 保留审计; 任何修复不可行的 seed 如实单列

## 汇总脚本 (待写, 数字冻结前的第一件事)
- scripts/aggregate_durfix.py: 读 results/durfix/** →
  ① 组件链表 (B1: base/pop/part/route/g/ge × 3 实例 × s42-44, 配对差 vs base)
  ② 主判别 10-seed (B2: popcxge vs base, 配对差/负种子数/分布)
  ③ 扩展 6 实例 + BKS gap (B3, BKS 锚 docs/mdvrptw-bks-2026.md)
  ④ C4 (B4) / 参数 (B5) / 等墙钟 (B6) / 600s 平台 (B7 双侧) / pyvrp std_cost 对照
  ⑤ 与旧值对照列 (旧值来源: results/population_* 旧文件, 只读)
- 显著性: 沿用旧口径 (Wilcoxon signed-rank 或配对 t, 标注上标 ᵃ/ᵇ/ᶜ, 表内不列 p)

## 判定文档 (按批, 参照 J1-J7 预注册)
- docs/durfix-b1-verdict + b2 (J1 核心) + b3 (J7) + b4/b5 (记录性) + b6 + b7 (含
  pyvrp 泄漏处置记录: 实测 pr15 4/20 max +110.8、pr20 4/24 max +45.8)

## 论文改写清单 (数字冻结后统一执行)
- [ ] §2.1 加约束口径句 (五类硬约束全实现); 摘要/§5/§6 数字全换 durfix 口径
- [ ] 表 4-9 / 附录 A/B 重算; "低于文献参照" 表述按新事实重写 (旧 pr04/pr06 观察作废)
- [ ] pyvrp 对照换 std_cost 口径 + 泄漏处置脚注
- [ ] 机制修正 (独立于数字, 可先做): §3.2 重写 (先强化后门/ρ 不存在/固定 20%/
      最便宜可行插入) + 算法 1 重排 + 图 1 重画 (make_ma_figs)
- [ ] 表 7 七条 + 其他文字项 (见 number-sheet 与 GPT 裁决记录)

## 发现: 墙钟超标 (B6 校准, 09-11 17:05) — 待处置
- 独占 6000iter 墙钟 旧→新: **pr15 596.3→703.0 (1.18×)** · **pr16 1224.9→1554.3 (1.27×)**
  · **pr20 931.7→1648.5 (1.77×)** — pr16/pr20 超 spec §8 验收的 "≤25%" 指引 (pr15 达标)。
- 结构: MA/BASE 每迭代成本比 旧≈1.0 → 新 **1.28 / 1.62 / 1.89** (实例越大越显著;
  由等墙钟 BASE 迭代数 ÷ 校准墙钟 反推)。疑因: MA 侧多跑的种群构建 (8×init) 与交叉子代
  经时长修复通道 (候选预算 15k) 的触发率随规模上升; BASE 只付核心路径。
- 影响面: **结果不受影响** (全批同引擎; 等墙钟设计 = BASE 预算取 MA 实测墙钟, 公平性反而
  更强 — BASE 拿到比旧版更长的让时仍输); 影响 = 论文墙钟披露行 + "两侧墙钟相当" 类表述。
- 处置 (09-12 用户确认 = 两修 + 重跑): **已完成** —
  ① Bug1 预算锚点覆盖初始化 + ② Bug2 timefix 防饥饿引导 (commit 3e4cc31; 230 测试
  全绿; 位中性已验: pr15 s42/s5 6000iter 重跑逐字段一致, 仅 +2 统计键);
  48 组时间预算运行重跑 09-12 13:15 启动 (~8.5h); 旧文件归档
  `results/durfix/archive_pretimefix/`; 完整审计见 docs/timefix-20260912.md。
- 归因待做 (排 B7 后): pr20 热点拆分 (种群构建 / 交叉子代修复触发率 / LS 时长门频率)。

## 投稿线 (挂起待令)
- 代码公开决策: 挂起 (用户未决)
- 物流科技投递版 (submission-main-wlkj.md): 数字更新后需从 v3.x 主稿同源重压缩,
  勿单独改一份; 联系渠道核实+作者简介待用户执行
