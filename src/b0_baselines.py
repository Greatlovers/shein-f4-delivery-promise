"""基线：常数/分组分位数承诺，在 3 个时间切分折上评分。"""
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from common import FOLLOW_UP, fmt_score, load_train, score

CUTS = [datetime(2026, 8, 24), datetime(2026, 9, 2), datetime(2026, 9, 12)]
W = timedelta(days=9)

tr = load_train(stage=False)


def fold(t0):
    known = tr.filter((pl.col("pay_time") < t0) & (pl.col("sign_time") < t0 + FOLLOW_UP))
    val = tr.filter((pl.col("pay_time") >= t0) & (pl.col("pay_time") < t0 + W))
    return known, val


def grp_quant(known, val, keys, q, recent_days=None, t0=None):
    k = known if recent_days is None else known.filter(pl.col("pay_time") >= t0 - timedelta(days=recent_days))
    tab = k.group_by(keys).agg(T=pl.col("y_total_hours").quantile(q))
    glob = k["y_total_hours"].quantile(q)
    return val.join(tab, on=keys, how="left").with_columns(pl.col("T").fill_null(glob))["T"].to_numpy()


for t0 in CUTS:
    known, val = fold(t0)
    g, y = val["lgs_company_nm"].to_numpy(), val["y_total_hours"].to_numpy()
    print(f"\n=== cut {t0.date()}  known={known.height:,} val={val.height:,}")
    T = np.full(len(y), known["y_total_hours"].quantile(0.9))
    print("B0 全局P90常数          ", fmt_score(score(g, y, T)))
    T = grp_quant(known, val, ["lgs_company_nm"], 0.9)
    print("B1 服务商P90            ", fmt_score(score(g, y, T)))
    T = grp_quant(known, val, ["lgs_company_nm"], 0.9, 21, t0)
    print("B1r 服务商P90(近21天)   ", fmt_score(score(g, y, T)))
    T = grp_quant(known, val, ["shpp_chnl"], 0.9, 21, t0)
    print("B2 渠道P90(近21天)      ", fmt_score(score(g, y, T)))
    T = grp_quant(known, val, ["shpp_chnl", "shpp_prvn_nm"], 0.9, 21, t0)
    print("B3 渠道x省P90(近21天)   ", fmt_score(score(g, y, T)))
    # 理论下界参考：真值里每个服务商的“恰好 90%/3%”常数（不可实现，只看量级）
    T = np.zeros(len(y))
    for k in np.unique(g):
        m = g == k
        T[m] = max(np.quantile(y[m], 0.9), np.quantile(y[m], 0.97) - 24)
    print("Oracle 服务商最优常数   ", fmt_score(score(g, y, T)))
