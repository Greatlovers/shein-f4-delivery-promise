"""实验工具：折缓存、单分位模型评估、“oracle 校准后的平均承诺”（衡量模型锐度，排除漂移）。"""
from __future__ import annotations

import sys
from datetime import datetime

import numpy as np
import polars as pl

from common import WORK, score
from feats import prepare_all
from model import make_fold

CUTS = [datetime(2026, 8, 25), datetime(2026, 9, 3), datetime(2026, 9, 12)]


def get_fold(t0, train_days=35, tag=""):
    p_tr = WORK / f"fold_{t0:%m%d}_{train_days}{tag}_tr.parquet"
    p_va = WORK / f"fold_{t0:%m%d}_{train_days}{tag}_va.parquet"
    if p_tr.exists():
        return pl.read_parquet(p_tr), pl.read_parquet(p_va)
    df = prepare_all()
    tr, va = make_fold(df, t0, train_days=train_days)
    tr.write_parquet(p_tr)
    va.write_parquet(p_va)
    return tr, va


def pinball(y, q, tau):
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def min_scale(g, y, lag, Q, on=0.90, sev=0.03):
    """每个服务商找最小乘子 s，使 T=lag+s*Q 同时满足准时率与严重超时约束（在给定数据上，事后最优）。"""
    s_out = {}
    for k in np.unique(g):
        m = g == k
        # 需要 (y-lag)/Q <= s 的比例 >= on，且 (y-24-lag)/Q <= s 的比例 >= 1-sev
        a = np.quantile((y[m] - lag[m]) / Q[m], on)
        b = np.quantile((y[m] - 24 - lag[m]) / Q[m], 1 - sev)
        s_out[k] = max(a, b)
    return s_out


def oracle_eval(va: pl.DataFrame, Q: np.ndarray):
    g = va["lgs_company_nm"].to_numpy()
    y = va["y_total_hours"].to_numpy()
    lag = va["lag_h"].to_numpy()
    s = min_scale(g, y, lag, Q)
    T = lag + Q * np.vectorize(s.get)(g)
    return score(g, y, T), s
