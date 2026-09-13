# M3 改进门+教育 A/B 预注册 (2026-09-07) — 轴2 里程碑 3

实现规格: docs/ma-alns-m3-spec-20260907.md; M2 判定: docs/ma-alns-m2-verdict-20260907.md。
判定协议归 Hermes; 本文件冻结于跑批前; 判定写 docs/ma-alns-m3-verdict-*.md。

## 1. 问题与假设

M2 判 pr20 上 part 重组 @rate0.2 负效, 归因预算竞争(cx 占 20% 迭代且子代多不优)。
M3 假设:
- H_G: **改进门消除预算竞争后, part 重组价值显现** — popcxg(门+回退, 无教育)
  对 pop 不劣于中性, 且对 M2 的 popcx_part@0.2 在 pr20 稳健更优
  (d(g−part02) < −s_pop, 预算竞争实锤的逆验证)。
- H_E: **池内教育转化 cx 后代** — popcxge(门+教育)对 popcxg 有增益
  (≥2/3 实例 d(ge−g) < −s_pop 零反向)。
- 质量里程碑(合并判据, 见 §4.3): popcxg 或 popcxge 对 pop 达成 ≥2/3 实例
  d < −s_pop 零反向 → 重组形态超壳 → 合并候选成立。

## 2. 口径(冻结)

- 实例 × seed: pr15 / pr16 / pr20 × s42-44。6000 iter 固定迭代。不跑 600s(M5)。
- 配置: BASE 逐键 = M1/M2; 臂(κ=0 全臂):
  - pop(参照, 不重跑 = population_m1/{inst}_s{seed}_pop.json)
  - popcx_part@0.2(参照, 不重跑 = population_m2/{inst}_s{seed}_popcx_part.json;
    即 gate="none" 的 M2 行为)
  - popcxg = partition + rate 0.2 + gate "improve"
  - popcxge = popcxg + educate True(burst 0.20, ls_budget 50)
- runner: experiments/benchmark_population_m3.py CLI {inst} {seed}
  {popcxg|popcxge}, 输出 results/population_m3/{inst}_s{seed}_{arm}.json =
  M2 字段全集(含结构指标, 公式不变)。
- 规模: 2 臂 × 3 实例 × 3 seeds = 18 组; xargs -P9 独立进程; 日志
  results/population_m3/batch_m3_6000iter.log。
- 确定性抽查: pr15 s42 popcxge 批后同配置重跑, 逐字段比对。

## 3. 判定规则(死值)

定义同 M1/M2: 每实例 d_{X−Y} = mean(3 seeds)(cost_X − cost_Y); s_pop = pop 臂
该实例 3-seed std。
1. **J1(gate 修复验证)**: 每实例 d_{g−part02} < −s_pop → 门消除了预算竞争负效
   (pr20 必测点); d_{g−pop} > +s_pop → popcxg 负效(门未救回, 修门); 否则中性。
2. **J2(重组价值)**: d_{g−pop} 与 d_{ge−pop}: < −s_pop 计该实例"超壳"; 稳健规则 =
   ≥2/3 实例超壳且零反向实例(反向 = d > +s_pop)。
3. **J3(教育增益)**: d_{ge−g} < −s_pop 于 ≥2/3 实例零反向 → H_E 支持; 否则教育
   无池内增益(如实记录, educate_delta_sum/成功率作归因)。
4. **J4(过程, 报告)**: cx_gate_success/fail 比例、educate_attempts/success 率与
   delta_sum 均值、novel 键、pool_final_spread、member_refresh_events、墙钟
   (教育臂允许 ≤ 2.5× pop 墙钟, 记录不设质量门槛)。
5. **J5 合并判据(预注册的合并决策规则, 判定后执行)**:
   - (i) 若 popcxg 或 popcxge 达 J2 稳健超壳 → **合并主线候选成立**: M3 verdict
      + 全量 pytest 后合回 main(全部机制 gated 默认关, 合并行为中性; 最优臂
      config 写进 verdict 供后续 600s/论文实验启用);
   - (ii) 若两臂均未超壳但无负效(全部中性或 pr15 类方向性增益) → 合并候选 =
      **M1 pop 壳本身**(质量证据: pr16/pr20 对 main base 超散布正效已判);
      cx/educate 机制 gated 保留作探索记录, 合并后默认关;
   - (iii) 若任一新臂负效(J1/J2 反向)→ 该臂修复或关闭后重判, 暂不合并。
   - 合并执行 = ff 合回 main(主线 12 个 commit 期间零移动), 不 squash(保留
      verdict/裁定链), 合并后分支删除与否由用户定。
6. 引擎验收: 全量 pytest 绿; 确定性重跑一致。
判定文档: docs/ma-alns-m3-verdict-20260907.md(Hermes 写)。

## 4. 后续(预声明)

- M3 判超壳 → 合并后继续 M4(双空间罚域种群, 分支新建)或先跑 600s 对照(M5);
- M3 判壳本身为最佳 → 合并 pop 壳, M4/M5 在主线后新分支推进;
- κ 档位批在任何超壳结论后作为参数精调(双变量解耦, 维持 M2 verdict 决策)。
