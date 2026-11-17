"""实验 6：增加切分点，扩充"跨窗口迁移"验证对。

决策层只用到模型的 P90（尺度）与 P97/P90（离散度），所以新增切分点只训练 q0.9、q0.97 两个模型。
每 3 天一个切分点；切分点 t 的验证窗口 [t, t+9d) 用来校准，评估在 [t+9d, t+18d)（即切分点 t+9 的窗口）上。
用法：python e6_cuts.py 2026-08-22 2026-08-28 ...
输出：work/pred_<mmdd>.parquet（与 train_mq 同格式，只有 q900/q970 两列分位数）
"""
import sys
import time
from datetime import datetime

import polars as pl

from common import WORK
from exp import get_fold
from feats import BASE_FEATS
from model import fit_quantile, predict
from recency import RC_FEATS

FEATS = BASE_FEATS + RC_FEATS


def run(t0: datetime, rounds: int = 400):
    out_path = WORK / f"pred_{t0:%m%d}.parquet"
    if out_path.exists():
        print(f"{t0:%m-%d} 已存在，跳过", flush=True)
        return
    t = time.time()
    tr, va = get_fold(t0, tag="_rc")
    print(f"{t0:%m-%d} 折构建 {time.time() - t:.0f}s  训练 {tr.height:,} 验证 {va.height:,}", flush=True)
    out = va.select(["pkg_no", "lgs_company_nm", "lag_h", "y_total_hours"])
    for tau in (0.9, 0.97):
        t = time.time()
        m = fit_quantile(tr, tau, feats=FEATS, rounds=rounds)
        out = out.with_columns(pl.Series(f"q{int(round(tau * 1000)):03d}", predict(m, va, FEATS)))
        print(f"  {t0:%m-%d} tau={tau} fit {time.time() - t:.0f}s", flush=True)
    out.write_parquet(out_path)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        run(datetime.strptime(a, "%Y-%m-%d"))
