"""报告用的数据洞察图（EDA）。输出到 docs/fig/。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from common import ROOT, WORK

FIG = ROOT / "docs" / "fig"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160})
INK, MUTED, ACC, ACC2, WARN = "#1f2937", "#9ca3af", "#2563eb", "#0d9488", "#dc2626"

df = pl.read_parquet(WORK / "all.parquet", columns=["pkg_no", "lgs_company_nm", "shpp_chnl", "pay_time", "lag_h", "r",
                                                    "y_total_hours", "is_test", "mode_category", "operation_type",
                                                    "stk_cate_tp_nm"])
tr, te = df.filter(~pl.col("is_test")), df.filter(pl.col("is_test"))

# 图 1：服务商在测试集中的包裹占比 vs 罚分权重（等权）
sh = te.group_by("lgs_company_nm").agg(n=pl.len()).with_columns(share=pl.col("n") / pl.col("n").sum() * 100).sort("n", descending=True)
fig, ax = plt.subplots(figsize=(6.2, 2.4))
x = np.arange(sh.height)
ax.bar(x, sh["share"], color=ACC, width=0.6, label="测试集包裹占比（决定 ME 中的权重）")
ax.axhline(100 / 8, color=WARN, ls="--", lw=1, label="罚分权重：每家相同（1/8）")
for i, v in enumerate(sh["share"]):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8, color=INK)
ax.set_xticks(x, [s.replace("LSP_", "") for s in sh["lgs_company_nm"]])
ax.set_ylabel("%")
ax.set_xlabel("物流服务商（LSP_）")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_ylim(0, 58)
fig.tight_layout()
fig.savefig(FIG / "fig_share.png")

# 图 2：按支付日的时效趋势（全体均值、P90）与主要服务商 P90
d = tr.with_columns(d=pl.col("pay_time").dt.date())
a = d.group_by("d").agg(mean=pl.col("y_total_hours").mean(), p90=pl.col("y_total_hours").quantile(0.9)).sort("d")
fig, ax = plt.subplots(figsize=(6.2, 2.4))
ax.plot(a["d"], a["mean"], color=INK, lw=1.4, label="全体均值")
ax.plot(a["d"], a["p90"], color=ACC, lw=1.4, label="全体 P90")
for k, c in [("LSP_0007", ACC2), ("LSP_0004", WARN)]:
    b = d.filter(pl.col("lgs_company_nm") == k).group_by("d").agg(p90=pl.col("y_total_hours").quantile(0.9)).sort("d")
    ax.plot(b["d"], b["p90"], color=c, lw=0.9, alpha=0.8, label=f"{k} P90")
ax.axvspan(np.datetime64("2026-09-21"), np.datetime64("2026-09-30"), color="#fde68a", alpha=0.6, lw=0)
ax.text(np.datetime64("2026-09-21T12"), 395, "测试期\n(9/21–29)", fontsize=8, color=INK)
ax.set_xlim(np.datetime64("2026-08-01"), np.datetime64("2026-09-30"))
ax.set_ylabel("小时")
ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper left")
ax.set_ylim(120, 440)
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
fig.tight_layout()
fig.savefig(FIG / "fig_trend.png")

# 图 3：LSP_0001 新线路：测试期每日件数（老渠道 vs 新渠道）+ 剩余时长对比
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.3), gridspec_kw={"width_ratios": [1.25, 1]})
t1 = te.filter(pl.col("lgs_company_nm") == "LSP_0001").with_columns(
    line=pl.when(pl.col("shpp_chnl").is_in(["CH_0051", "CH_0053"])).then(pl.lit("新线路")).otherwise(pl.lit("老渠道")),
    d=pl.col("pay_time").dt.date())
p = t1.group_by("d", "line").agg(n=pl.len()).pivot(on="line", index="d", values="n").sort("d").fill_null(0)
xd = np.arange(p.height)
axs[0].bar(xd, p["老渠道"], color=MUTED, width=0.65, label="老渠道 CH_0228")
axs[0].bar(xd, p["新线路"], bottom=p["老渠道"], color=WARN, width=0.65, label="新线路 CH_0051/0053")
axs[0].set_xticks(xd, [str(v)[5:] for v in p["d"]], rotation=45, fontsize=7)
axs[0].set_title("测试期 LSP_0001 每日件数", fontsize=9)
axs[0].legend(frameon=False, fontsize=7)
new19 = tr.filter(pl.col("shpp_chnl").is_in(["CH_0051", "CH_0053"]))
peer = tr.filter((pl.col("mode_category") == "MC_0003") & (pl.col("operation_type") == "OT_0002") &
                 (pl.col("stk_cate_tp_nm") == "WH_0005") & (pl.col("pay_time") >= pl.datetime(2026, 9, 6)) &
                 (pl.col("lgs_company_nm") != "LSP_0001"))
old = tr.filter((pl.col("shpp_chnl") == "CH_0228") & (pl.col("pay_time") >= pl.datetime(2026, 9, 6)))
data = [old["y_total_hours"].to_numpy(), new19["y_total_hours"].to_numpy(), new19["r"].to_numpy(), peer["r"].sample(20000, seed=1).to_numpy()]
bp = axs[1].boxplot(data, widths=0.55, showfliers=False, patch_artist=True)
for patch, c in zip(bp["boxes"], [MUTED, WARN, WARN, ACC]):
    patch.set_facecolor(c); patch.set_alpha(0.55)
axs[1].set_xticks([1, 2, 3, 4], ["老渠道\n总时效", "新线路19件\n总时效", "新线路19件\n流入后", "同仓同模式\n同行·流入后"], fontsize=6.8)
axs[1].set_ylabel("小时")
axs[1].set_title("新线路慢在“流入前积压”", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig_newline.png")
print("ok")
