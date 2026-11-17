"""决策层用到的 q0.9 / q0.97 做 3 个随机种子平均（对数尺度几何平均），降低单模型方差。

原 pred_<cut>.parquet 里的 q900/q970 来自 seed=42；这里再训 seed=7、99 两个，三者几何平均，
写出 pred_<cut>_ens.parquet（其余列不变）。只需要校准窗口（9/12）与测试（9/21）两个切分点。
实验依据：9/12 折同窗 P90 缩放得分 67.02 → 66.67（work/e9_0912.log，ens3）。
用法：python ens_pred.py 2026-09-12 test
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

FEATS = BASE_FEATS + RC_FEATS
SEEDS = (7, 99)


def run(t0: datetime, rounds: int = 400):
    src = WORK / f"pred_{t0:%m%d}.parquet"
    dst = WORK / f"pred_{t0:%m%d}_ens.parquet"
    base = pl.read_parquet(src)
    tr, va = get_fold(t0, tag="_rc")
    assert base["pkg_no"].to_list() == va["pkg_no"].to_list()
    for col, tau in (("q900", 0.9), ("q970", 0.97)):
        logs = [np.log(base[col].to_numpy())]
        for s in SEEDS:
            t = time.time()
            m = fit_quantile(tr, tau, feats=FEATS, rounds=rounds, params=dict(seed=s))
            logs.append(np.log(predict(m, va, FEATS)))
            print(f"  {t0:%m-%d} {col} seed={s} fit {time.time() - t:.0f}s", flush=True)
        base = base.with_columns(pl.Series(col, np.exp(np.mean(logs, axis=0))))
    base.write_parquet(dst)
    print(f"写出 {dst.name}", flush=True)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        run(TEST_CUT if a == "test" else datetime.strptime(a, "%Y-%m-%d"))
