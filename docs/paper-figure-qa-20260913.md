# 论文图 1-4 投稿级改造与渲染 QA (2026-09-13)

**范围**: 按 nature-figure 技能 (Hermes `academic-writing` 分类) 对论文图 1-4 做投稿级改造 —
印宽 1:1 设计、印后字号规范化、信息层增强 (图 2 逐种子层 / 图 4 误差线)、技能 QA 链全过、
Word 渲染逐页验收。生成器 `experiments/make_ma_figs.py` (v3.3; 数据根
`MDVRPTW_FIG_RESULTS=results/durfix`); 文档构建 `scripts/make_paper_docx.py`
(fig_w 与画布 1:1: fig1 15.0 / fig2 10.5 / fig3 15.0 / fig4 14.0 cm)。

## 1. 设计口径

- **画布 = 刊印插入宽 1:1**; 图内字号即印后字号 (7–8.5 pt)。旧版实况: 大画布被缩 0.33–0.55 倍,
  印后字仅约 2–4.5 pt (图 1 最甚) — 本次为该项遗留的正式修复。
- **灰度安全**: 图 3 路线"颜色+线型/线宽+标记"三重编码; 图 4(a) 三档**纯灰度填充** (黑/中灰/白,
  替代 v3.2 的点纹/斜纹, 印后更干净且更耐缩放); 图 1/2 线型+标记编码。
- **信息层增强 (均为既有数据的可视化)**: 图 2 加各种子累计值小点 (3 种子, 按实例左右分开;
  脚本内断言"种子累计均值 = 均值累计"与逐级闭合); 图 4(a) 加 ±10 种子标准差误差线;
  图 4(b)(c) 统一纵轴 (0–60%) 便于两算例对照。
- **题注同步** (中英双语, paper-mdvrptw-v2 / submission-main / submission-main-wlkj 三份):
  图 2 加"折线为 3 种子均值、小点为各种子累计值"; 图 4(a) 加"误差线为 10 种子标准差"。

## 2. QA 结果 (nature-figure 技能链 + 渲染验收)

| 项 | 结果 |
|---|---|
| 源预检 `validate_figure.py` | 17 pass / 4 warn / 0 fail (warn 均为记录性: TIFF 由 docx 构建器另出、300 dpi 标准档、静态宽度检测限制、流程图白底标签) |
| 多面板对齐门 (fig3/fig4 渲染时, 1.5 pt) | **PASS** (`figs_qa/*.alignment.json`) |
| PDF 字号审计 (`audit_pdf_text --min-pt 5`) | 四图全 **PASS**; 实测最小 fig1 6.8 / fig2 7.0 / fig3 7.0 / fig4 6.5 pt |
| 碰撞审计 (`audit_figure_collisions`) | fig1 **PASS** (0/0); fig2 **PASS** (0/0); fig3 10 fail/10 warn (见 §3); fig4 **0 fail**/1 warn |
| `check_paper_consistency.py` | ALL PASS (表 15 / 图 4 / 文献 11 / 节 6+附录) |
| 渲染验收 (投稿前遗留项) | 物流科技版 docx → Word COM → PDF **13 页**; 图 1-4 位于第 **5/8/9/11** 页; 逐页 vision 复核: 无越界/遮挡/压字, 图内字可读 |

**修复链 (审计驱动)**:
1. fig2 纵轴题字左缘越界 (PDF 页外 0.25 cm) → 轴框左移 (left 0.135→0.168)。
2. fig3 底部"x 坐标"题字越界 → 画布 7.2→7.5 cm (面板尺寸不变, 底边距 0.83→1.09 cm)。
3. fig4 "+6.44/+6.93" 与图例文字实测重叠 0.7 pt → 标签上移 (+0.35) + 纵轴上限 10.4→11.2。
4. fig4 **全部 17 个 fail 溯源 = matplotlib hatch 点纹的 PDF 笔画几何** (2.5 秒样本判读为
   hatch 路径) → 换纯灰度三档填充后 **fail 归零**; 该改动同时提升了印后观感 (审计工具对
   hatch 的误报类问题记录在案)。
5. fig1 "下一迭代"竖排标签与右侧框缘几何相切 → 标签移至回环上段 (14.22, 2.0), 回环通道 14.55。

## 3. 图 3 残余审计项的判定 (有意保留)

10 个 fail/10 个 warn 全部是同一类: **D1–D4 车场标签与两处成本标注框被路线笔画几何穿过**
(车场是全部路线的汇合点, 叠印不可避免)。三点复核后保留: ① 标签/成本框均有不透明白底光垫,
视觉可读 (3 倍放大裁剪逐项确认, `figs_qa/crops/`); ② 灰度三重编码仍完整; ③ 把标签移离车场
反而失去"该方块是哪个车场"的锚定。此类叠印为路线图惯例, 判为有意设计。

## 4. 复现命令

```bash
# 图件重生成 (含对齐门; 需 nature-figure 技能 scripts 目录)
MDVRPTW_FIG_RESULTS=results/durfix \
  PYTHONPATH=~/ai-skills/nature-skills/skills/nature-figure/scripts \
  python experiments/make_ma_figs.py all
# 文档重建 (docx 4 目标 + figs_tif 备份自动重生成)
python scripts/make_paper_docx.py all
# 一致性 + 渲染后审计 (审计需 pymupdf)
python scripts/check_paper_consistency.py
python <nature-figure>/scripts/audit_pdf_text.py docs/paper/figs/figN.pdf --min-pt 5
python <nature-figure>/scripts/audit_figure_collisions.py docs/paper/figs/figN.pdf \
  --json-out docs/paper/figs_qa/figN.collision.json
```

QA 产物 (对齐 JSON/SVG、碰撞 JSON、放大裁剪、渲染页 PNG) 均在 `docs/paper/figs_qa/` (gitignore);
图件 PDF (矢量原件) 在 `docs/paper/figs/` (gitignore); 投稿图件为 `figs/*.png` (300 dpi, 入库)
与 `figs_tif/` (本地备份)。

## 5. 遗留

- 表 7 pr15 行计数记法 "（1/3 组为负）" 与表 6 / 表 7 pr16 行的 "(n/3)" 不统一 (渲染抽检发现;
  未改, 表内容待定)。
- `figs_tif/fig5_convergence.tif` 为 v3.2 之前的旧版本残留备份 (gitignore, 不影响投递包)。
