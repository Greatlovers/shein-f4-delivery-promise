"""实验 10 融合筛选：9/12 折，LightGBM（单种子 / 三种子）与 CatBoost、XGBoost 的 q0.9 在对数尺度加权平均，
按"同窗事后 P90 缩放"得分比较（与 e9 同口径，基线：LightGBM 单种子 67.02、三种子 66.67）。
用法：python e10_blend.py   （需先跑 e10_alt.py screen）
"""
from datetime import datetime

import numpy as np
import polars as pl

from common import WORK
from e10_alt import report
from exp import get_fold

t0 = datetime(2026, 9, 12)
_, va = get_fold(t0, tag="_rc")
P = {"lgb1": pl.read_parquet(WORK / "pred_0912.parquet"), "lgb3": pl.read_parquet(WORK / "pred_0912_ens.parquet")}
for k, p in P.items():
    assert p["pkg_no"].to_list() == va["pkg_no"].to_list()
L = {k: np.log(p["q900"].to_numpy()) for k, p in P.items()}
for name in ("cb", "xgb"):
    f = WORK / f"alt_0912_{name}_q900.npy"
    if f.exists():
        L[name] = np.log(np.load(f))

BLENDS = [{"lgb1": 1}, {"lgb3": 1}, {"cb": 1}, {"xgb": 1},
          {"lgb1": 1, "cb": 1}, {"lgb1": 1, "xgb": 1},
          {"lgb3": 2, "cb": 1}, {"lgb3": 1, "cb": 1}, {"lgb3": 2, "xgb": 1}, {"lgb3": 1, "xgb": 1},
          {"lgb3": 1, "cb": 1, "xgb": 1}, {"lgb3": 2, "cb": 1, "xgb": 1}]
for b in BLENDS:
    if not all(k in L for k in b):
        continue
    w = sum(b.values())
    q = np.exp(sum(v * L[k] for k, v in b.items()) / w)
    report(va, q, "+".join(f"{k}{'' if v == 1 else '×' + str(v)}" for k, v in b.items()), 0)
