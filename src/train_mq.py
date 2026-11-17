"""多分位数模型：对每个折训练一组分位数模型，保存验证/测试集的条件分位数预测。

用法：python train_mq.py <cut YYYY-MM-DD | test> [tag]
输出：work/pred_<cut>.parquet  （pkg_no, lgs_company_nm, lag_h, y_total_hours, q050..q995，单位：小时，为“流入→签收”剩余时长）
"""
import sys
import time
from datetime import datetime

import numpy as np
import polars as pl

from common import WORK
from exp import get_fold
from feats import BASE_FEATS
from model import TEST_CUT, fit_quantile, predict
from recency import RC_FEATS

TAUS = [0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995]
FEATS = BASE_FEATS + RC_FEATS


def run(t0: datetime, tag: str = "", rounds: int = 400, train_days: int = 35, save_models: bool = False):
    tr, va = get_fold(t0, train_days=train_days, tag="_rc")
    cols = {"pkg_no": va["pkg_no"], "lgs_company_nm": va["lgs_company_nm"], "lag_h": va["lag_h"],
            "y_total_hours": va["y_total_hours"] if "y_total_hours" in va.columns else None}
    out = pl.DataFrame({k: v for k, v in cols.items() if v is not None})
    for tau in TAUS:
        t = time.time()
        m = fit_quantile(tr, tau, feats=FEATS, rounds=rounds)
        out = out.with_columns(pl.Series(f"q{int(round(tau * 1000)):03d}", predict(m, va, FEATS)))
        print(f"  {t0:%m-%d} tau={tau} fit {time.time() - t:.0f}s", flush=True)
        if save_models:
            m.save_model(str(WORK / f"lgb_{t0:%m%d}{tag}_q{int(round(tau * 1000)):03d}.txt"))
    out.write_parquet(WORK / f"pred_{t0:%m%d}{tag}.parquet")
    return out


if __name__ == "__main__":
    arg = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    t0 = TEST_CUT if arg == "test" else datetime.strptime(arg, "%Y-%m-%d")
    run(t0, tag)
