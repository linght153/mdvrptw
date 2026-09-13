"""MA-ALNS 论文插图生成 (v3.3, 2026-09-13: 按 nature-figure 技能做投稿级改造)。

用法: python experiments/make_ma_figs.py [fig1|fig2|fig3|fig4|all]
输出: docs/paper/figs/  (图号按论文正文首次引用顺序)
  fig1_ma_flow.png/.pdf       — 图 1: MA-ALNS 单次迭代流程 (对应正文算法 1)
  fig2_component_chain.*      — 图 2: 组件叠加链逐级累计配对差 (线条 = 3 种子
                                均值, 小点 = 各种子累计值; 数据源 = 表 5/10 同批
                                JSON: population_ab base、population_m1 pop、
                                population_m2 popcx_part、population_m3 popcxg/popcxge;
                                脚本内做逐级累计闭合断言)
  fig3_final_routes.*         — 图 3: 主算例最终解路线图 (数据 = results/ma_routes/
                                dump_ma_routes.py 确定性重跑 dump; 双面板共享图例)
  fig4_dual_budget_gap.*      — 图 4: 双口径对照 (a) 双口径对 BKS 偏差 (10 种子
                                均值 ± 标准差) + (b)(c) 固定迭代收敛轨迹

v3.3 设计口径 (nature-figure 投稿级):
  1) 画布 = 刊印插入宽 1:1 (fig1 15.0 / fig2 10.5 / fig3 15.0 / fig4 14.0 cm;
     对应 scripts/make_paper_docx.py 的 fig_w), 图内字号即印后字号 (7-8.5 pt),
     不再大画布缩小 (旧版 fig1-3 印后仅 2-4.5 pt)。
  2) 多面板对齐门 (nature-figure 技能要求): fig3/fig4 生成时调用
     require_matplotlib_panel_alignment (需将技能 scripts 目录加入 PYTHONPATH,
     如 PYTHONPATH=~/ai-skills/nature-skills/skills/nature-figure/scripts);
     对齐报告落 docs/paper/figs_qa/ (gitignore)。不可导入时跳过并提示。
  3) 渲染后 QA (技能要求, 手动):
     audit_pdf_text.py figs/figN.pdf --min-pt 5
     audit_figure_collisions.py figs/figN.pdf --json-out figs_qa/figN.collision.json
  4) 图内中文渲染 (Windows: Microsoft JhengHei); 流程图为灰度安全设计;
     路线图按车场"颜色+线型/线宽+标记"三重编码并附图例 (灰度印刷下凭线型区分);
     组件链用线型+标记区分, 小点按实例左右分开。

图 1 布局 (v3.2 机制修正基础上): 重组子代先无条件执行池内强化, 再进改进门;
"找不到第二父代"与"改进门未通过"两条回退路径均为虚线 (回退迭代不占用额外预算)。
"""
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
QA = ROOT / "docs" / "paper" / "figs_qa"
# v3.2: 结果根可切换 (MDVRPTW_FIG_RESULTS=results/durfix 时使用修复后口径数据)
RES = Path(os.environ.get("MDVRPTW_FIG_RESULTS", str(ROOT / "results")))

CM = 2.54  # cm per inch

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "KaiTi", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 300
plt.rcParams.update({
    "svg.fonttype": "none",   # 矢量文本保持可编辑 (SVG)
    "pdf.fonttype": 42,       # PDF 内嵌 TrueType 可编辑文本
    "ps.fonttype": 42,
})
plt.rcParams["font.size"] = 8

# 主算例 BKS (文献[2,3], 双精度口径, 与表 11/10B 同源)
BKS = {"pr15": 2433.15, "pr16": 2836.67, "pr20": 2983.78}
INSTS = ["pr15", "pr16"]
DEPOT_COLORS = ["#c1272d", "#0050b3", "#1e8a34", "#8e44ad", "#e67e22", "#2c9eb3"]
# 灰度安全三重编码: 线型 + 线宽 + 沿线标记形状 (颜色仅供屏幕阅读)
DEPOT_LS = ["-", (0, (6, 2)), ":", "-.", (0, (3, 1, 1, 1, 1, 1)), (0, (1.5, 1.5))]
DEPOT_LW = [1.15, 0.85, 1.0, 0.75, 0.9, 0.75]
DEPOT_MK = ["o", "^", "s", "D", "v", "P"]

# ── 印宽 (cm) 与文档构建器 fig_w 保持 1:1 ──
W_FIG1, H_FIG1 = 15.0, 11.0
W_FIG2, H_FIG2 = 10.5, 6.6
W_FIG3, H_FIG3 = 15.0, 7.5
W_FIG4, H_FIG4 = 14.0, 11.0


def load_inst(name: str):
    sys.path.insert(0, str(ROOT))
    from scripts.parse_cordeau_mdvrptw import parse_mdvrptw
    from src.core.instance import instance_from_parsed
    text = (ROOT / f"data/mdvrptw_raw/{name}.txt").read_text(encoding="utf-8")
    return instance_from_parsed(parse_mdvrptw(text), name=name)


def _save(fig, name: str, w_cm: float, h_cm: float):
    """按画布原尺寸存 PNG (300 dpi) + PDF (矢量), 并核对像素尺寸。"""
    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    from PIL import Image
    im = Image.open(png)
    exp = (round(w_cm / CM * 300), round(h_cm / CM * 300))
    assert abs(im.size[0] - exp[0]) <= 2 and abs(im.size[1] - exp[1]) <= 2, \
        f"{name}: 像素 {im.size} != 预期 {exp} (画布 {w_cm}x{h_cm}cm)"
    plt.close(fig)
    print(f"{name} → {png.name} ({w_cm:.1f}x{h_cm:.1f} cm, "
          f"{im.size[0]}x{im.size[1]} px, 1:1 印宽设计)")


def _alignment_gate(fig, key: str):
    """多面板渲染对齐门 (nature-figure 技能; 需技能 scripts 目录在 PYTHONPATH)。"""
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        print(f"[qa] {key}: 对齐门跳过 — audit_panel_alignment 不可导入 "
              f"(运行前置 PYTHONPATH=.../nature-figure/scripts)")
        return None
    QA.mkdir(parents=True, exist_ok=True)
    report = require_matplotlib_panel_alignment(
        fig,
        json_out=str(QA / f"{key}.alignment.json"),
        overlay_svg=str(QA / f"{key}.alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    print(f"[qa] {key}: 多面板对齐门通过 (报告 {key}.alignment.json)")
    return report


# ─────────────────────────── 图 1 流程图 ───────────────────────────
# 画布 15.0x11.0 cm, 坐标单位 = cm, y 轴向下为正。
# 主列 (常规迭代) / 重组列 (触发后进入), 底部公共收尾条, 右缘回环。

def _flow_box(ax, xc, y_top, w, h, label, kind="proc", fs=8.0):
    if kind == "decision":
        cy = y_top + h / 2
        hw, hh = w / 2, h / 2
        ax.fill([xc - hw, xc, xc + hw, xc], [cy, y_top, cy, y_top + h],
                fc="#f5f5f5", ec="black", lw=0.9, zorder=3)
        ax.text(xc, cy, label, ha="center", va="center", fontsize=fs,
                zorder=4, linespacing=1.3)
        return
    if kind == "term":
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch((xc - w / 2, y_top), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    fc="#fdfdfd", ec="black", lw=0.9, zorder=3,
                                    mutation_aspect=1.0))
    else:
        ax.add_patch(plt.Rectangle((xc - w / 2, y_top), w, h,
                                   fc="#f0f0f0", ec="black", lw=0.9, zorder=3))
    ax.text(xc, y_top + h / 2, label, ha="center", va="center",
            fontsize=fs, zorder=4, linespacing=1.35)


def _flow_arrow(ax, pts, dashed=False, label=None, fs=7.5, lx=None, ly=None,
                ha="center"):
    """沿 pts 折线画实线/虚线, 末端画箭头。"""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color="black", lw=0.7, ls=(0, (3, 2)) if dashed else "-",
            zorder=2)
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7,
                                mutation_scale=7.0, shrinkA=0, shrinkB=0),
                zorder=2)
    if label:
        ax.text(lx if lx is not None else pts[-1][0],
                ly if ly is not None else pts[-1][1],
                label, fontsize=fs, ha=ha, va="center",
                bbox=dict(fc="white", ec="none", pad=0.5), zorder=5)


def fig1_flow():
    fig = plt.figure(figsize=(W_FIG1 / CM, H_FIG1 / CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_FIG1)
    ax.set_ylim(H_FIG1, 0)
    ax.axis("off")
    MX, CX = 3.4, 11.35            # 主列 / 重组列中心
    FC = 7.15                      # 回退虚线竖向通道
    LC = 14.55                     # 右缘回环通道
    MW, CW = 5.8, 6.0              # 两列框宽

    # ── 主列 (自上而下) ──
    _flow_box(ax, MX, 0.25, 4.6, 0.58, "第 t 次迭代开始", kind="term")
    _flow_box(ax, MX, 1.10, MW, 0.92,
              "锦标赛选择成员 a\n（适应度最小，式(1)）")
    _flow_box(ax, MX, 2.42, 5.0, 1.44, "触发重组？\n均匀随机数 < p_c",
              kind="decision")
    _flow_box(ax, MX, 4.55, MW, 1.0,
              "常规破坏—修复（ALNS 主路径）\nchild ← destroy/repair(a)")
    _flow_box(ax, MX, 5.95, MW, 1.0,
              "对 child 执行 VND 局部搜索与 SA 接受；\n更新当前解与算子权重 w")

    # ── 重组列 (自上而下) ──
    _flow_box(ax, CX, 4.55, CW, 0.94,
              "选第二父代 b：与 a 的路由键集不同\n（至多 12 次；找不到则回退）")
    _flow_box(ax, CX, 5.69, CW, 1.0,
              "簇级重组（a, b）：按车场掷硬币继承 b 的\n路由组；未覆盖客户重插 → child")
    _flow_box(ax, CX, 6.89, CW, 0.94,
              "池内强化 child（无条件执行）：\n移除比例 0.20，深度上限 B")
    _flow_box(ax, CX, 8.03, 5.2, 1.44,
              "改进门：child 可行\n且 c(child) < c(a)？", kind="decision")

    # ── 底部公共收尾条 ──
    _flow_box(ax, 7.4, 9.85, 13.6, 0.95,
              "迭代收尾：可行改进解插入档案；成员停滞计数累计\n"
              "（连续 T_r 次未置换者由档案精英副本驱逐替换）", fs=7.5)

    # ── 连线 ──
    _flow_arrow(ax, [(MX, 0.83), (MX, 1.10)])                    # 开始→选 a
    _flow_arrow(ax, [(MX, 2.02), (MX, 2.42)])                    # 选 a→◇p_c
    _flow_arrow(ax, [(MX, 3.86), (MX, 4.55)])                    # ◇p_c 否→破坏修复
    _flow_arrow(ax, [(MX + 2.5, 3.14), (CX, 3.14), (CX, 4.55)])  # ◇p_c 是→选 b
    _flow_arrow(ax, [(MX, 6.95), (MX, 9.85)])                    # 主列→收尾
    _flow_arrow(ax, [(CX, 5.49), (CX, 5.69)])                    # 选 b→重组
    _flow_arrow(ax, [(CX, 6.69), (CX, 6.89)])                    # 重组→强化
    _flow_arrow(ax, [(CX, 7.83), (CX, 8.03)])                    # 强化→◇门
    _flow_arrow(ax, [(CX, 9.47), (CX, 9.85)])                    # ◇门 是→收尾
    # 回退路径 (虚线): ①选 b 找不到第二父代 → 直退回破坏修复;
    #                 ②◇门 未通过 → 经通道 x=FC 上溯回破坏修复。
    _flow_arrow(ax, [(CX - CW / 2, 5.02), (MX + MW / 2 + 0.08, 5.02)],
                dashed=True)
    _flow_arrow(ax, [(CX - 2.6, 8.75), (FC, 8.75), (FC, 5.35),
                     (MX + MW / 2 + 0.08, 5.35)], dashed=True)

    # ── 出口/路径标注 ──
    ax.text(3.05, 4.20, "否", fontsize=7.5, ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=0.4), zorder=5)
    ax.text(6.30, 2.90, "是", fontsize=7.5, ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=0.4), zorder=5)
    ax.text(11.75, 9.62, "是", fontsize=7.5, ha="left", va="center",
            bbox=dict(fc="white", ec="none", pad=0.4), zorder=5)
    ax.text(8.45, 8.55, "否", fontsize=7.5, ha="right", va="center",
            bbox=dict(fc="white", ec="none", pad=0.4), zorder=5)
    ax.text(6.95, 7.55, "虚线：门未过或无第二父代时\n回退为常规破坏—修复\n"
            "（不额外消耗预算）", fontsize=6.8, ha="right", va="center",
            bbox=dict(fc="white", ec="none", pad=0.6), zorder=5,
            linespacing=1.4)

    # ── 右缘回环: 收尾 → 下一迭代开始 ──
    ax.plot([14.2, LC], [10.325, 10.325], color="black", lw=0.7, zorder=2)
    ax.plot([LC, LC], [10.325, 0.54], color="black", lw=0.7, zorder=2)
    ax.plot([LC, 5.90], [0.54, 0.54], color="black", lw=0.7, zorder=2)
    ax.annotate("", xy=(5.72, 0.54), xytext=(5.90, 0.54),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7,
                                mutation_scale=7.0, shrinkA=0, shrinkB=0),
                zorder=2)
    ax.text(14.22, 2.0, "下一迭代  t ← t+1", fontsize=7.5, rotation=90,
            rotation_mode="anchor", ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)

    _save(fig, "fig1_ma_flow", W_FIG1, H_FIG1)
    return OUT / "fig1_ma_flow.png"


# ─────────────────────────── 图 2 组件叠加链 ───────────────────────────

def fig2_chain():
    # 与表 5/10/10B 同批数据源 (臂 → 目录 + 文件名)
    stages = [
        ("base", "population_ab", "base"),
        ("pop", "population_m1", "pop"),
        ("cx", "population_m2", "popcx_part"),
        ("gate", "population_m3", "popcxg"),
        ("ge", "population_m3", "popcxge"),
    ]
    x_labels = ["BASE", "+种群池", "+簇重组", "+改进门", "+池内强化"]
    seeds = (42, 43, 44)
    per_seed = {inst: {sd: {} for sd in seeds} for inst in INSTS}
    mu = {inst: {} for inst in INSTS}
    for inst in INSTS:
        for key, sub, arm in stages:
            vals = []
            for sd in seeds:
                p = RES / sub / f"{inst}_s{sd}_{arm}.json"
                if not p.exists():
                    raise SystemExit(f"[fig2] 缺 {p}")
                c = json.loads(p.read_text(encoding="utf-8"))["best_feasible_cost"]
                vals.append(c)
                per_seed[inst][sd][key] = c
            mu[inst][key] = float(np.mean(vals))
    # 闭合断言: 逐级均值差累计 = 端到端均值差 (浮点精确)
    keys = [k for k, _, _ in stages]
    for inst in INSTS:
        cum = sum(mu[inst][keys[i]] - mu[inst][keys[i - 1]]
                  for i in range(1, len(keys)))
        e2e = mu[inst]["ge"] - mu[inst]["base"]
        assert abs(cum - e2e) < 1e-9, f"{inst}: 累计 {cum} != 端到端 {e2e}"
        print(f"[fig2] {inst}: 累计闭合 = 端到端 {e2e:.1f}")
    # 逐级累计: 均值 (线条) 与各种子 (小点)
    cum = {inst: [0.0] for inst in INSTS}
    cum_seed = {inst: {sd: [0.0] for sd in seeds} for inst in INSTS}
    for i in range(1, len(keys)):
        for inst in INSTS:
            cum[inst].append(cum[inst][-1] + mu[inst][keys[i]]
                             - mu[inst][keys[i - 1]])
            for sd in seeds:
                cum_seed[inst][sd].append(
                    cum_seed[inst][sd][-1]
                    + per_seed[inst][sd][keys[i]]
                    - per_seed[inst][sd][keys[i - 1]])
    # 一致性: 各种子累计的均值 = 均值累计
    for inst in INSTS:
        m = np.mean([cum_seed[inst][sd] for sd in seeds], axis=0)
        assert np.allclose(m, cum[inst], atol=1e-6), f"{inst}: 种子均值不闭合"

    fig = plt.figure(figsize=(W_FIG2 / CM, H_FIG2 / CM))
    ax = fig.add_axes([0.168, 0.105, 0.782, 0.865])
    styles = {"pr15": ("o-", "#111111"), "pr16": ("s--", "#4d4d4d")}
    xs = np.arange(5)
    # 各种子小点 (先画, 位于线条之下; 按实例左右分开, 保留种子可见)
    jit = {"pr15": {42: -0.12, 43: -0.07, 44: -0.02},
           "pr16": {42: 0.02, 43: 0.07, 44: 0.12}}
    for inst in INSTS:
        mk = "o" if inst == "pr15" else "s"
        col = "#111111" if inst == "pr15" else "#5a5a5a"
        alpha = 0.55 if inst == "pr15" else 0.65
        for sd in seeds:
            ax.plot(xs + jit[inst][sd], cum_seed[inst][sd], ls="none",
                    marker=mk, ms=1.9, mfc=col, mec=col, alpha=alpha,
                    zorder=2)
    # 均值折线
    for inst, (ls, col) in styles.items():
        ax.plot(xs, cum[inst], ls, color=col, lw=1.6, ms=4.2,
                mfc="white", mec=col, mew=0.7, label=inst, zorder=3)
        ax.annotate(f"{cum[inst][-1]:.0f}", (4, cum[inst][-1]),
                    textcoords="offset points", xytext=(9, 0), fontsize=7.5,
                    va="center", zorder=4)
    ax.axhline(0, color="black", lw=0.8, zorder=1)
    ax.set_xlim(-0.28, 4.85)
    ax.set_ylim(-360, 135)
    ax.set_yticks([-300, -200, -100, 0, 100])
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, fontsize=7.5)
    ax.set_ylabel("相对 BASE 的累计配对差\n（负值 = 更优）", fontsize=8,
                  linespacing=1.5)
    ax.tick_params(labelsize=7.5, width=0.7, length=2.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    seed_proxies = [
        Line2D([0], [0], ls="none", marker="o", ms=2.4, mfc="#111111",
               mec="#111111", alpha=0.8),
        Line2D([0], [0], ls="none", marker="s", ms=2.4, mfc="#5a5a5a",
               mec="#5a5a5a", alpha=0.8),
    ]
    ax.legend(h1 + seed_proxies,
              ["pr15 均值", "pr16 均值", "pr15 种子", "pr16 种子"],
              fontsize=7, frameon=False, ncol=2, loc="upper right",
              handlelength=1.8, handletextpad=0.5, columnspacing=1.2,
              labelspacing=0.35, borderaxespad=0.25)
    _save(fig, "fig2_component_chain", W_FIG2, H_FIG2)
    return OUT / "fig2_component_chain.png"


# ─────────────────────────── 图 3 最终解路线图 ───────────────────────────

def fig3_routes():
    best = {}
    for inst in INSTS:
        hits = sorted((RES / "ma_routes").glob(
            f"{inst}_s*_popcxge_routes.json"))
        if not hits:
            print(f"[fig3] 缺 {inst} 的 dump (等 scripts/dump_ma_routes.py 完成)")
            return None
        best[inst] = min(
            hits, key=lambda p: json.loads(p.read_text(encoding="utf-8"))
            ["best_cost"])
    fig, axes = plt.subplots(1, 2, figsize=(W_FIG3 / CM, H_FIG3 / CM))
    fig.subplots_adjust(left=0.048, right=0.988, top=0.855, bottom=0.145,
                        wspace=0.10)
    proxies = None
    for idx, (ax, inst) in enumerate(zip(axes, INSTS)):
        d = json.loads(best[inst].read_text(encoding="utf-8"))
        inst_obj = load_inst(inst)
        nd = inst_obj.num_depots
        custs, depots = inst_obj.customers, inst_obj.depots
        ax.scatter([c.x for c in custs], [c.y for c in custs], s=2.6,
                   c="#6e6e6e", zorder=2)
        for di0, routes in d["routes"].items():
            di = int(di0)
            col = DEPOT_COLORS[di % len(DEPOT_COLORS)]
            dp = depots[di]
            for r in routes:
                xs = [dp.x] + [custs[g - nd].x for g in r] + [dp.x]
                ys = [dp.y] + [custs[g - nd].y for g in r] + [dp.y]
                ax.plot(xs, ys, ls=DEPOT_LS[di % len(DEPOT_LS)],
                        lw=DEPOT_LW[di % len(DEPOT_LW)], color=col,
                        marker=DEPOT_MK[di % len(DEPOT_MK)], markevery=2,
                        ms=2.2, mew=0.5, mfc="white", mec=col,
                        alpha=0.95, zorder=3)
        if proxies is None:
            # 共享图例: 车场编号 → 颜色/线型/标记 (灰度印刷下凭线型区分归属)
            proxies = [Line2D([0], [0],
                              color=DEPOT_COLORS[d2 % len(DEPOT_COLORS)],
                              ls=DEPOT_LS[d2 % len(DEPOT_LS)],
                              lw=DEPOT_LW[d2 % len(DEPOT_LW)],
                              marker=DEPOT_MK[d2 % len(DEPOT_MK)], ms=2.8,
                              mew=0.55, mfc="white", label=f"D{d2 + 1}")
                       for d2 in range(nd)]
        for di, dp in enumerate(depots):
            ax.scatter([dp.x], [dp.y], marker="s", s=52, c=DEPOT_COLORS[di],
                       edgecolors="black", linewidths=0.8, zorder=4)
            # 编号置于方块上方+白底光晕 (车场相距近时方块内文字会重叠)
            ax.annotate(f"D{di + 1}", (dp.x, dp.y), fontsize=8,
                        fontweight="bold", ha="center", va="bottom",
                        xytext=(0, 2.4), textcoords="offset points",
                        color="#111111", zorder=6,
                        bbox=dict(fc="white", ec="none", pad=0.12))
        gap = (d["best_cost"] - BKS[inst]) / BKS[inst] * 100.0
        lab = ("(a)", "(b)")[idx]
        ax.set_title(f"{lab}  {inst}  (种子 {d['seed']}, {d['vehicles']} 车)",
                     fontsize=8.5, pad=4)
        ax.text(0.985, 0.025, f"成本 {d['best_cost']:.1f}  "
                f"(对 BKS {gap:+.2f}%)", transform=ax.transAxes,
                fontsize=7.5, ha="right", va="bottom",
                bbox=dict(fc="white", ec="0.65", lw=0.6, pad=1.2), zorder=6)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7, width=0.7, length=2.2)
        ax.set_xlabel("x 坐标", fontsize=7.5)
        if idx == 0:
            ax.set_ylabel("y 坐标", fontsize=7.5)
    # 共享图例置于两面板上方 (避免遮挡路线)
    fig.legend(handles=proxies, loc="upper center", ncol=4, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.998),
               handlelength=2.4, columnspacing=1.8, handletextpad=0.5)
    _alignment_gate(fig, "fig3")
    _save(fig, "fig3_final_routes", W_FIG3, H_FIG3)
    print(f"[fig3] dumps: " + ", ".join(f"{i}={best[i].name}" for i in INSTS))
    return OUT / "fig3_final_routes.png"


# ─────── 图 4 双口径对照: (a) 双口径对 BKS 偏差 + (b-c) 固定迭代收敛轨迹 ───────

def fig4_dual_budget():
    """数据源 (v3.2; RES 根 = results/durfix):
    (a) 表 8/9 同批 10 种子均值 JSON:
        MA@6000iter = population_m5/{inst}_s{1..10}_popcxge.json (根目录)
        MA@600s     = population_m5/600s/{inst}_s{1..10}_popcxge.json
        pyvrp@600s  = pyvrp_wide_corrected/{inst}_s{1..10}.json
                      (std_cost_polished = 标准转换值, 修复+打磨确定性)
    (b-c) convergence_trace_v3/ (run_convergence_trace.py v3,
        6000iter × s42-44 × base/popcxge; 端点与 durfix 批 JSON 断言一致)。
    v3.3 版面: 画布 14.0x11.0 cm 1:1; (a) 加 10 种子标准差误差线;
    (b)(c) 统一纵轴范围。
    """
    # ---- 数据 (a): 双口径对 BKS 偏差 + 10 种子标准差 ----
    def seed_costs(pat):
        v = []
        for s in range(1, 11):
            p = RES / pat.format(s=s)
            if not p.exists():
                raise SystemExit(f"[fig4] 缺 {p}")
            v.append(json.loads(p.read_text(encoding="utf-8"))
                     ["best_feasible_cost"])
        return v

    gaps, sds = {}, {}
    gaps["MA@6000iter"] = [(np.mean(seed_costs(f"population_m5/{i}_s{{s}}_popcxge.json")) - BKS[i]) / BKS[i] * 100 for i in INSTS]
    sds["MA@6000iter"] = [(np.std(seed_costs(f"population_m5/{i}_s{{s}}_popcxge.json"), ddof=1) / BKS[i] * 100) for i in INSTS]
    gaps["MA@600s"] = [(np.mean(seed_costs(f"population_m5/600s/{i}_s{{s}}_popcxge.json")) - BKS[i]) / BKS[i] * 100 for i in INSTS]
    sds["MA@600s"] = [(np.std(seed_costs(f"population_m5/600s/{i}_s{{s}}_popcxge.json"), ddof=1) / BKS[i] * 100) for i in INSTS]
    gaps["pyvrp@600s"], sds["pyvrp@600s"] = [], []
    for i in INSTS:
        v = []
        for s in range(1, 11):
            d = json.loads((RES / "pyvrp_wide_corrected" / f"{i}_s{s}.json")
                           .read_text(encoding="utf-8"))
            v.append(d["std_cost_polished"])
        gaps["pyvrp@600s"].append((np.mean(v) - BKS[i]) / BKS[i] * 100)
        sds["pyvrp@600s"].append(np.std(v, ddof=1) / BKS[i] * 100)
    for k, exp in [("MA@6000iter", [4.54, 3.68]),
                   ("MA@600s", [6.44, 6.93]),
                   ("pyvrp@600s", [5.92, 4.40])]:
        for i, (g, e) in enumerate(zip(gaps[k], exp)):
            assert abs(g - e) < 0.06, f"{k} {INSTS[i]}: {g:.3f} != {e}"
    for k, exp in [("MA@6000iter", [1.565, 0.748]), ("MA@600s", [1.706, 1.187]),
                   ("pyvrp@600s", [1.549, 1.967])]:
        for i, (g, e) in enumerate(zip(sds[k], exp)):
            assert abs(g - e) < 0.02, f"{k} sd {INSTS[i]}: {g:.3f} != {e}"

    # ---- 数据 (b-d): 收敛轨迹 ----
    traces = {}
    for inst in INSTS:
        for arm in ("base", "popcxge"):
            for sd in (42, 43, 44):
                p = RES / "convergence_trace_v3" \
                    / f"{inst}_s{sd}_{arm}_trace.json"
                if not p.exists():
                    raise SystemExit(f"[fig4] 缺 {p} (等轨迹批完成)")
                d = json.loads(p.read_text(encoding="utf-8"))
                traces[(inst, arm, sd)] = \
                    {it: c for it, c in d["samples"] if c is not None}
    exp_end = {"pr15": (2664.4, 2607.8), "pr16": (3128.0, 2926.9)}
    for inst in INSTS:
        for arm, e in zip(("base", "popcxge"), exp_end[inst]):
            m = np.mean([traces[(inst, arm, sd)][6000]
                         for sd in (42, 43, 44)])
            assert abs(m - e) < 0.5, f"{inst} {arm} 端点 {m:.1f} != {e}"
    iters = sorted({it for t in traces.values() for it in t})

    # ---- 绘制: (a) 顶部通栏 + (b)(c) 底部两格 ----
    fig = plt.figure(figsize=(W_FIG4 / CM, H_FIG4 / CM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.22], hspace=0.44,
                          wspace=0.28, left=0.088, right=0.970, top=0.938,
                          bottom=0.082)
    ax = fig.add_subplot(gs[0, :])

    # (a) 分组柱状 + 10 种子标准差误差线
    x = np.arange(2)
    w = 0.26
    series_list = ["MA@6000iter", "MA@600s", "pyvrp@600s"]
    # 灰度安全: 黑 / 中灰 / 白(黑边) 三档填充 (印后区分明确, 不使用点纹)
    draw = [dict(color="#1a1a1a"),
            dict(color="#b8b8b8", edgecolor="#333333"),
            dict(color="white", edgecolor="#333333")]
    for k, (key, st) in enumerate(zip(series_list, draw)):
        for xi, gi in enumerate(gaps[key]):
            if gi is None:
                ax.text(x[xi] + (k - 1) * w, 0.12, "—", ha="center",
                        va="bottom", fontsize=8, color="#555555")
                continue
            bx = x[xi] + (k - 1) * w
            ax.bar(bx, gi, w, yerr=[[0], [sds[key][xi]]], capsize=1.8,
                   error_kw=dict(elinewidth=0.8, capthick=0.8,
                                 ecolor="#222222"), zorder=3, **st)
            ax.text(bx, gi + sds[key][xi] + 0.35, f"{gi:+.2f}", ha="center",
                    va="bottom", fontsize=7.2)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["pr15", "pr16"], fontsize=8)
    ax.set_ylabel("对 BKS 的偏差 (%)", fontsize=7.5)
    ax.set_title("(a)  双口径对 BKS 偏差", fontsize=8.5, pad=4)
    ax.set_ylim(0, 11.2)
    ax.set_yticks([0, 2, 4, 6, 8])
    handles = [plt.Rectangle((0, 0), 1, 1, **s) for s in
               [dict(fc="#1a1a1a", ec="#1a1a1a"),
                dict(fc="#b8b8b8", ec="#333333"),
                dict(fc="white", ec="#333333")]]
    ax.legend(handles, ["MA-ALNS, 6000 迭代", "MA-ALNS, 600 s",
                        "pyvrp, 600 s"],
              fontsize=7, frameon=False, loc="upper center", ncol=3,
              handlelength=1.2, borderaxespad=0.25, labelspacing=0.32,
              columnspacing=1.4)
    ax.tick_params(labelsize=7.5, width=0.7, length=2.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # (b-c) 收敛轨迹 (b=pr15, c=pr16); 两面板统一纵轴范围便于对照
    styles = {"base": ("-", "black"), "popcxge": ("--", "#555555")}
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ymax = 0.0
    for inst in INSTS:
        for arm in ("base", "popcxge"):
            Y = np.array([[traces[(inst, arm, sd)].get(it, np.nan)
                           for sd in (42, 43, 44)] for it in iters])
            mu = np.nanmean(Y, axis=1)
            sd_ = np.nanstd(Y, axis=1)
            ymax = max(ymax, np.nanmax((mu + sd_ - BKS[inst]) / BKS[inst] * 100))
    ylim_top = float(np.ceil(ymax / 5.0) * 5)
    print(f"[fig4] (b)(c) 统一纵轴上限 = {ylim_top:.0f}%")
    for idx, (inst, ax2) in enumerate(zip(INSTS, (ax_b, ax_c))):
        xs = iters
        for arm, (ls, col) in styles.items():
            Y = np.array([[traces[(inst, arm, sd)].get(it, np.nan)
                           for sd in (42, 43, 44)] for it in xs])
            mu = np.nanmean(Y, axis=1)
            sd_ = np.nanstd(Y, axis=1)
            ax2.plot(xs, (mu - BKS[inst]) / BKS[inst] * 100, ls, color=col,
                     lw=1.2, label="BASE" if arm == "base" else "MA-ALNS")
            ax2.fill_between(xs, (mu - sd_ - BKS[inst]) / BKS[inst] * 100,
                             (mu + sd_ - BKS[inst]) / BKS[inst] * 100,
                             color=col, alpha=0.10 if arm == "base" else 0.15,
                             lw=0)
        base_end = np.mean([traces[(inst, "base", sd)][6000]
                            for sd in (42, 43, 44)])
        base_gap = (base_end - BKS[inst]) / BKS[inst] * 100
        ax2.axhline(base_gap, color="black", lw=0.7, ls=":")
        crossing = None
        for it in xs:
            m_ma = np.mean([traces[(inst, "popcxge", sd)].get(it, np.nan)
                            for sd in (42, 43, 44)])
            if np.isfinite(m_ma) and m_ma <= base_end:
                crossing = it
                break
        if crossing is not None:
            ax2.axvline(crossing, color="#7d7d7d", lw=0.8, ls="-.")
            ax2.annotate(f"首穿 ~{crossing}", (crossing, ylim_top),
                         fontsize=7, ha="left", va="top",
                         xytext=(3, -2), textcoords="offset points")
        letter = ("(b)", "(c)")[idx]
        ax2.set_title(f"{letter}  {inst}", fontsize=8.5, pad=3.5)
        ax2.set_xlim(0, 6000)
        ax2.set_ylim(0, ylim_top)
        ax2.set_yticks(range(0, int(ylim_top) + 1, 10))
        ax2.set_xticks([0, 2000, 4000, 6000])
        ax2.tick_params(labelsize=7.5, width=0.7, length=2.2)
        ax2.set_xlabel("迭代次数", fontsize=7.5)
        if idx == 0:
            ax2.legend(fontsize=7, frameon=False, loc="upper right",
                       borderaxespad=0.2, handlelength=1.9, labelspacing=0.3,
                       title="阴影带：±样本标准差", title_fontsize=6.5)
            ax2.set_ylabel("三种子均值对 BKS 的偏差 (%)", fontsize=7.5)
        for s in ("top", "right"):
            ax2.spines[s].set_visible(False)

    _alignment_gate(fig, "fig4")
    _save(fig, "fig4_dual_budget_gap", W_FIG4, H_FIG4)
    return OUT / "fig4_dual_budget_gap.png"


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("fig1", "all"):
        fig1_flow()
    if which in ("fig3", "all"):
        fig3_routes()
    if which in ("fig2", "all"):
        fig2_chain()
    if which in ("fig4", "all"):
        fig4_dual_budget()
