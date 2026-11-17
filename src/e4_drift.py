"""实验 4：漂移估计。

每 3 天一个切分点 t0，训练 q0.9 与 q0.97 两个模型，预测 [t0, t0+9d) 窗口，
求每个服务商“恰好满足约束所需的剩余时长缩放” s_g(t0)。
相隔 9 天的两个切分点之间 log(s_g(t0+9)/s_g(t0)) 就是“用上一窗口校准、用于下一窗口”时的漂移 δ。
输出 work/drift.csv
"""
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from common import WORK
from exp import get_fold
from feats import BASE_FEATS
from model import fit_quantile, predict
from recency import RC_FEATS

FEATS = BASE_FEATS + RC_FEATS
cuts = [datetime(2026, 8, 19) + timedelta(days=3 * i) for i in range(9)]  # 8/19 … 9/12
rows = []
for t0 in cuts:
    tr, va = get_fold(t0, tag="_rc")
    y, lag, g = va["y_total_hours"].to_numpy(), va["lag_h"].to_numpy(), va["lgs_company_nm"].to_numpy()
    preds = {}
    for tau in (0.9, 0.97):
        m = fit_quantile(tr, tau, feats=FEATS, rounds=300)
        preds[tau] = predict(m, va, FEATS)
    r = y - lag
    for k in np.unique(g):
        msk = g == k
        s90 = np.quantile(r[msk] / preds[0.9][msk], 0.90)               # 准时率约束所需缩放
        s97 = np.quantile((r[msk] - 24) / preds[0.97][msk], 0.97)       # 严重超时约束所需缩放
        rows.append(dict(cut=t0, carrier=k, n=int(msk.sum()), s90=float(s90), s97=float(s97),
                         cov90=float((r[msk] <= preds[0.9][msk]).mean()), cov97=float((r[msk] - 24 <= preds[0.97][msk]).mean())))
    print(t0.date(), "done", flush=True)
    pl.DataFrame(rows).write_csv(WORK / "drift.csv")
