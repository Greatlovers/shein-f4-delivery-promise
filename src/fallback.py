"""新渠道/稀有渠道的“同行线路”兜底模型。

测试集中 LSP_0001 的新线路（CH_0051/CH_0053，MC_0003/OT_0002，WH_0005）训练集里几乎没有样本，
LSP_0006 的 CH_0218 也只有 2 条。对这类渠道，不用渠道/服务商身份，而用“仓 × 模式 × 运营类型 × 机场 × 路由 ×
目的省份 × 时段”等可迁移属性学习同行线路的条件分位数。

loco_check()：留渠道验证——把一批真实渠道从训练中整体拿掉，检验兜底模型对“没见过的渠道”的分位数覆盖率，
据此确定新渠道需要加多少保险余量。
"""
from __future__ import annotations

import sys
from datetime import datetime

import numpy as np
import polars as pl

from common import WORK
from exp import get_fold
from feats import BASE_FEATS
from model import TEST_CUT, fit_quantile, predict
from recency import RC_FEATS

FB_FEATS = [f for f in BASE_FEATS if f not in ("shpp_chnl_id", "lgs_company_nm_id")] + \
           ["gap_d"] + [f for f in RC_FEATS if f.startswith("rc_wa_")]
FB_TAUS = [0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995]
RARE_MIN = 300  # 训练窗口内样本数低于此值的渠道视为“稀有/新渠道”


def fit_fallback(tr: pl.DataFrame, rows: pl.DataFrame, taus=FB_TAUS, frac=0.5, rounds=300):
    sub = tr.sample(fraction=frac, seed=1) if frac < 1 else tr
    out = {}
    for tau in taus:
        m = fit_quantile(sub, tau, feats=FB_FEATS, rounds=rounds)
        out[f"q{int(round(tau * 1000)):03d}"] = predict(m, rows, FB_FEATS)
        print(f"  fallback tau={tau} {m._fit_secs:.0f}s", flush=True)
    return pl.DataFrame(out)


def rare_mask(tr: pl.DataFrame, rows: pl.DataFrame) -> np.ndarray:
    cnt = tr.group_by("shpp_chnl").agg(n=pl.len())
    j = rows.select("shpp_chnl").join(cnt, on="shpp_chnl", how="left", maintain_order="left")
    return (j["n"].fill_null(0) < RARE_MIN).to_numpy()


def loco_check(t0=datetime(2026, 9, 12), n_hold=12):
    """留渠道验证：随机拿掉 n_hold 个中等规模、MC_0003 模式的渠道。"""
    tr, va = get_fold(t0, tag="_rc")
    cand = (va.group_by("shpp_chnl", "mode_category").agg(n=pl.len())
              .filter((pl.col("mode_category") == "MC_0003") & (pl.col("n").is_between(1000, 15000))))
    hold = cand.sample(n=min(n_hold, cand.height), seed=3)["shpp_chnl"].to_list()
    tr2 = tr.filter(~pl.col("shpp_chnl").is_in(hold))
    vh = va.filter(pl.col("shpp_chnl").is_in(hold))
    q = fit_fallback(tr2, vh, taus=[0.5, 0.9, 0.97, 0.99, 0.995])
    r = (vh["y_total_hours"] - vh["lag_h"]).to_numpy()
    print("held-out channels:", hold, "rows:", vh.height)
    res = vh.select("shpp_chnl", "lgs_company_nm").with_columns(r=r, **{c: q[c] for c in q.columns})
    for c in q.columns:
        print(f"  {c}: coverage={np.mean(r <= q[c].to_numpy()):.4f}  (r-24<=q: {np.mean(r - 24 <= q[c].to_numpy()):.4f})")
    per = res.group_by("shpp_chnl", "lgs_company_nm").agg(
        n=pl.len(), cov90=(pl.col("r") <= pl.col("q900")).mean(), cov99=(pl.col("r") <= pl.col("q990")).mean(),
        sev_at_q97=((pl.col("r") - pl.col("q970")) > 24).mean(), ratio_med=(pl.col("r") / pl.col("q500")).median())
    print(per.sort("n", descending=True))
    res.write_parquet(WORK / "loco_check.parquet")


if __name__ == "__main__":
    if sys.argv[1] == "loco":
        loco_check()
