# 结构性差异分析预注册 (pyvrp/HGS vs 自产 ALNS, 2026-09-06)

衔接: pyvrp 宽窗对照判定 (docs/pyvrp-wide-control-verdict-20260905.md) 待办 1
— "差距定位第一步: pyvrp 解结构 vs 我们的 (routes dump 已存): 车场分配/
路由形态/负载分布差异, 找'它凭什么破'的结构线索 (决定形态跃迁的具体组件
优先级)"。本分析只读 dump + 几何/时间窗事实计算, 不改引擎, 无随机性,
单臂枚举类 (09-04 探针方法学: 单臂枚举类底座成立, 不涉臂间差判定)。

## 数据

| 侧 | 来源 | 实例 × seed | 口径 |
|---|---|---|---|
| pyvrp 600s | results/pyvrp_wide/{name}_s{seed}_600s.json | pr15/16/20 × s42-44 | float 重算可行 (pr15/20); **pr16 int 口径 float 不可行, 仅形态参考并披露** |
| 自产 base 600s | results/base600s_routes/{name}_s{seed}_routes.json (补跑) + connectivity_{name}_s42_routes.json (既有, s42) | pr15/16/20 × s42-44 | 引擎 base 配置 600s 最优可行解 (CONFIG = probe_connectivity_dump, 与 baseline_pr_120s/*_600s.json 逐位一致已验证) |

两侧 seq 均为全局节点号 (客户 i 的全局号 = i + num_depots; pyvrp
route.visits() 与项目 Solution 同约定), 可直接对齐。gap 参照:
pr15 BKS 2433.15 (pyvrp −0.19% vs 自产 600s +3.8% s42), pr20 BKS
2983.78 (pyvrp −0.29% vs 自产 +6.5~11.4%), pr16 BKS 2836.67。

## 指标组 (每解全量计算, 逐实例 3-seed 配对报告)

1. **宏观形态**: 路由数 (车辆数, vs 每车场 vehicles_available)、每车场
   路由数分布、路由客户数分布 (均值/离散)、路由负载分布 (利用率 =
   load/capacity)、每车场总负载 vs 车场总容量 (veh×cap)。
2. **车场分配几何一致性**: 每客户实际车场 vs 最近车场的 rank (0=最近);
   rank=0 比例 (几何一致率)、rank≥2 比例、d_actual/d_nearest 均值。
   错配客户空间特征: 错配集 (rank>0) 的最近车场距离分布。
3. **路由区带紧凑性**: 路由内客户最近车场=路由车场的比例 (区带纯度,
   客户侧); 路由客户到路由车场均距 vs 到各自最近车场均距的比值 (装载
   "别区" 客户的代价); 路由绕行效率 = 闭环行驶距离 / Σ(客户到车场
   往返) — 扇区式路由≈2 倍往返下限附近, 绕圈路由显著更高。
4. **TW-顺序结构**: 路由内客户 due_time 逆序对比例 (TW 排序强度);
   路由末客户完成时刻相对其 due 的裕度 (时间紧迫度); 窄窗客户 (窗宽
   ≤ 全实例 P25) 在路由内位置分布 (是否被排在路由后段/优先衔接)。

## 判定口径 (探索性线索, 非假设检验)

- 差异方向成立 = 该指标两侧配对均值差跨 3 seed 符号一致 (pr15 与 pr20
  各自独立看, 不强求跨实例一致 — 形态差异可实例特异)。
- 输出 = 差异表 + 每条差异的"形态跃迁组件"映射建议 (verdict 阶段),
  不预注册组件优先级结论。

## 执行

1. 补 dump: scripts/dump_base600s_routes.py × 7 (pr15 s43/44, pr16
   s42-44, pr20 s43/44; s42 既有), 串行独立进程 600s 预算 (后台批已启)。
2. 分析: experiments/analyze_structure_diff.py (只读 dump + 实例文件,
   输出 JSON + 终端表)。
3. 判定文档: docs/structure-diff-verdict-20260906.md。
