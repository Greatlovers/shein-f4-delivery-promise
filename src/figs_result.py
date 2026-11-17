"""报告用的结果图：迭代瀑布、漂移σ敏感性、测试集承诺解读、决策层示意。输出到 docs/fig/。"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from common import ROOT, WORK

FIG = ROOT / "docs" / "fig"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160})
INK, MUTED, ACC, ACC2, WARN = "#1f2937", "#9ca3af", "#2563eb", "#0d9488", "#dc2626"
R = json.loads((ROOT / "docs" / "results.json").read_text(encoding="utf-8"))

# 图 4：迭代过程（三折平均的严格版得分，对数纵轴）
steps = R["iteration"]
fig, ax = plt.subplots(figsize=(6.4, 2.5))
names = [s["name"] for s in steps]
vals = [s["A"] for s in steps]
cols = [MUTED if s.get("kind") == "base" else (ACC if s.get("kind") == "model" else ACC2) for s in steps]
x = np.arange(len(steps))
ax.bar(x, vals, color=cols, width=0.62)
for i, v in enumerate(vals):
    ax.text(i, v * 1.04, f"{v:.1f}", ha="center", fontsize=7.5, color=INK)
ax.set_yscale("log")
ax.set_xticks(x, names, fontsize=7, rotation=0)
ax.set_ylabel("严格版得分 A（越低越好，对数轴）")
ax.set_ylim(40, 700)
fig.tight_layout()
fig.savefig(FIG / "fig_iter.png")

# 图 5：漂移余量 σ 的迁移验证
tr = R["transfer"]  # {目标窗口: {sigma: A}}
fig, ax = plt.subplots(figsize=(3.2, 2.4))
full = [k for k in tr if k not in ("9/15", "9/18")]
for k in tr:
    d = tr[k]
    s = sorted(float(v) for v in d)
    ax.plot(s, [d[f"{v:g}"] for v in s], color=MUTED, lw=0.8, ls="-" if k in full else ":", alpha=0.8,
            label="各目标窗口" if k == full[0] else None)
sg = sorted({float(v) for k in full for v in tr[k]})
mean = [np.mean([tr[k][f"{v:g}"] for k in full]) for v in sg]
ax.plot(sg, mean, color=ACC, lw=2, marker="o", ms=3.5, label="5 对完整窗口平均")
ax.axvline(R["final"]["sigma"], color=WARN, ls="--", lw=1)
ax.set_xlabel("漂移情景 σ（对数尺度）")
ax.set_ylabel("下一窗口真实得分 A")
ax.set_ylim(65, 125)
ax.set_xlim(-0.003, max(sg) + 0.005)
ax.legend(frameon=False, fontsize=6.5, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "fig_sigma.png")

# 图 6：测试集承诺 —— 承诺/模型P90 的比值与预测离散度的关系（决策层“按难度分层收保险费”）
d = pl.read_parquet(ROOT / "outputs" / "test_detail.parquet").filter(~pl.col("rare"))
d = d.with_columns(ratio=(pl.col("pred_hours") - pl.col("lag_h")) / pl.col("q900"))
smp = d.sample(30000, seed=1)
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
axs[0].scatter(smp["q900"], smp["pred_hours"] - smp["lag_h"], s=1, alpha=0.15, color=ACC, rasterized=True)
mx = float(smp["q900"].quantile(0.995))
axs[0].plot([0, mx], [0, mx], color=MUTED, lw=1, ls="--")
axs[0].plot([0, mx], [24, mx + 24], color=WARN, lw=0.8, ls=":")
axs[0].set_xlim(0, mx)
axs[0].set_ylim(0, mx * 1.6)
axs[0].set_xlabel("模型 P90（流入后剩余时长，h）")
axs[0].set_ylabel("最终承诺 − lag（h）")
axs[0].set_title("短时效件多留“24h缓冲”比例，长时效件按比例加保险", fontsize=8)
per = (pl.read_parquet(ROOT / "outputs" / "test_detail.parquet")      # 含稀有/新渠道，与第四节参数表一致
       .group_by("lgs_company_nm").agg(meanT=pl.col("pred_hours").mean(), n=pl.len()).sort("n", descending=True))
axs[1].barh([s.replace("LSP_", "") for s in per["lgs_company_nm"]][::-1], per["meanT"].to_list()[::-1], color=ACC2)
for i, v in enumerate(per["meanT"].to_list()[::-1]):
    axs[1].text(v + 3, i, f"{v:.0f}", va="center", fontsize=7)
axs[1].set_xlabel("测试集平均承诺（h）")
axs[1].set_title("各服务商平均承诺", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig_test.png")
# 图 3b：特征重要性（来源 work/e2.log：9/12 折、q0.9、基础+近况特征，增益占比 %）
imp = [("渠道", 41.7), ("目的省份", 10.7), ("近况：渠道 28 天基线", 8.7), ("流入周内小时", 5.8),
       ("近况：服务商 28 天基线", 3.5), ("始发机场", 3.4), ("合包类型", 2.1), ("重量", 1.9),
       ("近况：仓×机场 28 天基线", 1.7), ("近况：渠道近 7 天件量", 1.6), ("lag（支付→流入）", 1.3), ("流入时刻", 1.3)]
fig, ax = plt.subplots(figsize=(6.2, 2.5))
names = [a for a, _ in imp][::-1]
vals = [b for _, b in imp][::-1]
cols = [ACC2 if n.startswith("近况") else ACC for n in names]
ax.barh(names, vals, color=cols, height=0.65)
for i, v in enumerate(vals):
    ax.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=6.8)
ax.tick_params(axis="y", labelsize=7.5)
ax.set_xlim(0, 47)
ax.set_xlabel("增益占比 %（绿色为近况特征）", fontsize=7.5)
fig.tight_layout()
fig.savefig(FIG / "fig_imp.png")
print("ok")
