"""实验 8：在 q0.9 / q0.97 模型里加入日历时间近况特征（nowcast_cal.py），各切分点重训。

输出 work/pred_<mmdd>_cal.parquet（格式同 e6_cuts），供 e7_transfer.py 用 suf="_cal" 评估。
用法：python e8_cal.py 2026-08-22 ...    （test 表示测试切分 9/21，会训练全部 11 个分位点）
"""
import sys
import time
from datetime import datetime

import polars as pl

from common import FOLLOW_UP, WORK
from exp import get_fold
from feats import BASE_FEATS
from model import TEST_CUT, fit_quantile, predict
from nowcast_cal import CAL_FEATS, add_calendar
from recency import RC_FEATS

FEATS = BASE_FEATS + RC_FEATS + CAL_FEATS
KCOLS = ["pay_time", "issue_time", "sign_time", "st_wh", "st_mid", "st_lm", "shpp_chnl_id", "lgs_company_nm_id", "is_test"]


def with_cal(t0: datetime):
    tr, va = get_fold(t0, tag="_rc")
    df = pl.read_parquet(WORK / "all.parquet", columns=KCOLS)
    known = df.filter(~pl.col("is_test") & (pl.col("pay_time") < t0) & (pl.col("sign_time") < t0 + FOLLOW_UP))
    return add_calendar(tr, known, "train"), add_calendar(va, known, "fixed", t0)


def run(t0: datetime, taus=(0.9, 0.97), rounds: int = 400, suf: str = "_cal"):
    out_path = WORK / f"pred_{t0:%m%d}{suf}.parquet"
    if out_path.exists():
        print(f"{t0:%m-%d} 已存在，跳过", flush=True)
        return
    t = time.time()
    tr, va = with_cal(t0)
    print(f"{t0:%m-%d} 特征 {time.time() - t:.0f}s", flush=True)
    keep = [c for c in ["pkg_no", "lgs_company_nm", "lag_h", "y_total_hours"] if c in va.columns]
    out = va.select(keep)
    for tau in taus:
        t = time.time()
        m = fit_quantile(tr, tau, feats=FEATS, rounds=rounds)
        out = out.with_columns(pl.Series(f"q{int(round(tau * 1000)):03d}", predict(m, va, FEATS)))
        imp = sorted(zip(FEATS, m.feature_importance("gain")), key=lambda x: -x[1])
        tot = sum(v for _, v in imp)
        cal_imp = {f: round(100 * v / tot, 1) for f, v in imp if f in CAL_FEATS}
        print(f"  {t0:%m-%d} tau={tau} fit {time.time() - t:.0f}s  日历特征增益% {cal_imp}", flush=True)
    out.write_parquet(out_path)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        if a == "test":
            from train_mq import TAUS
            run(TEST_CUT, taus=TAUS)
        else:
            run(datetime.strptime(a, "%Y-%m-%d"))
