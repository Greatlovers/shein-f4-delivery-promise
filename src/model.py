"""折构建 + LightGBM 分位数模型。"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import lightgbm as lgb
import numpy as np
import polars as pl

from common import FOLLOW_UP
from feats import BASE_FEATS, CAT_FEATS, HIST_COLS, hist_features
from recency import RC_FEATS, add_recency

VAL_DAYS = 9
TEST_CUT = datetime(2026, 9, 21)


def make_fold(df: pl.DataFrame, t0: datetime, train_days: int = 35, follow_up=FOLLOW_UP, hist_window: int = 28,
              use_hist: bool = False, use_recency: bool = True):
    """返回 (训练集, 验证/测试集)。

    训练侧只用切分时刻 t0 前支付、且在 t0+follow_up 前签收的标签——与测试时刻“标签观察到 11/5”的信息量一致。
    验证集用完整标签（含之后才签收的慢件）。t0 == TEST_CUT 时验证集即测试集。
    历史统计：验证行用 t0 前 hist_window 天；训练行用“留一周”（排除本包裹所在支付周），避免同批次邻居信息泄漏。
    """
    known = df.filter(~pl.col("is_test") & (pl.col("pay_time") < t0) & (pl.col("sign_time") < t0 + follow_up))
    if t0 >= TEST_CUT:
        val = df.filter(pl.col("is_test"))
    else:
        val = df.filter(~pl.col("is_test") & (pl.col("pay_time") >= t0) & (pl.col("pay_time") < t0 + timedelta(days=VAL_DAYS)))
    wk = ((pl.col("pay_time") - pl.lit(t0)).dt.total_seconds() / (7 * 86400)).floor().cast(pl.Int32).alias("_wk")
    known = known.with_columns(wk)
    kn = known.filter(pl.col("pay_time") >= t0 - timedelta(days=hist_window))
    tr = known.filter(pl.col("pay_time") >= t0 - timedelta(days=train_days))
    if use_hist:
        parts = [hist_features(tr.filter(pl.col("_wk") == w), kn.filter(pl.col("_wk") != w))
                 for w in tr["_wk"].unique().to_list()]
        tr = pl.concat(parts)
        val = hist_features(val, kn)
    tr = tr.drop("_wk").sort("pay_time")
    if use_recency:
        tr = add_recency(tr, known, "train")
        val = add_recency(val, known, "fixed", t0)
    return tr, val


FEATS = BASE_FEATS + RC_FEATS

PARAMS = dict(objective="quantile", learning_rate=0.08, num_leaves=255, min_data_in_leaf=400, feature_fraction=0.8,
              bagging_fraction=0.7, bagging_freq=1, max_bin=255, cat_smooth=20, cat_l2=10, max_cat_threshold=64,
              num_threads=12, verbose=-1, seed=42,
              deterministic=True, force_col_wise=True)   # 复现：多线程下结果逐位一致


def to_np(d: pl.DataFrame, feats=FEATS):
    return d.select(feats).to_numpy().astype(np.float32)


def fit_quantile(tr: pl.DataFrame, q: float, feats=FEATS, rounds: int = 500, weight=None, target="r", log=True,
                 params=None):
    p = dict(PARAMS, alpha=q, **(params or {}))
    y = tr[target].to_numpy()
    if log:
        y = np.log(y)
    ds = lgb.Dataset(to_np(tr, feats), y, weight=weight, feature_name=feats,
                     categorical_feature=[f for f in CAT_FEATS if f in feats], free_raw_data=True)
    t = time.time()
    m = lgb.train(p, ds, num_boost_round=rounds)
    m._fit_secs = time.time() - t
    m._log = log
    return m


def predict(m, d: pl.DataFrame, feats=FEATS):
    p = m.predict(to_np(d, feats), num_threads=12)
    return np.exp(p) if m._log else p
