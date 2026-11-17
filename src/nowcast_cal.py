"""日历时间近况特征：用"已知包裹在某个日历日开始尾程（揽收）后的耗时"去估计本包裹将来走到同一环节时的拥堵程度。

动机：recency.py 按支付日看"最近几批包裹的总时长"，而冲击往往发生在某个日历日的某个环节（如 9/16 起
LSP_0007 尾程 P90 从 ~85h 升到 ~110h）。较早支付、走得慢的已知包裹会先"路过"这些日历日，
它们的环节耗时就是信息。

信息边界与 recency.py 完全一致：只用支付日 ≤ c_day 的已知包裹（c 为信息时点：训练行 c = pay − gap，
验证/测试行 c = 切分时刻）。环节时间戳只用于这类统计，不作为单个包裹的输入。

特征（lvl ∈ ch 渠道 / car 服务商）：
  cl_{lvl}_ahead : 本包裹"预计揽收日" D* 前后 3 天内揽收的已知包裹，log 尾程耗时均值 − 28 天基线
  cl_{lvl}_now   : 最近 3 个日历日（c_day-2…c_day）揽收的已知包裹，同上
  cl_{lvl}_nahead: 上面 ahead 的样本量；cl_gap: D* − c_day（天）
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from recency import DAY0

LEVELS = {"ch": "shpp_chnl_id", "car": "lgs_company_nm_id"}
ND = 140          # 日网格长度（DAY0=7/1 起，覆盖到 11/18）
BASE = 28
MIN_N = 20


def _grid(known: pl.DataFrame, key: str, nk: int):
    k = known.select(
        pl.col(key).alias("k"),
        ((pl.col("pay_time") - pl.lit(DAY0)).dt.total_days()).alias("p"),
        ((pl.col("issue_time") + pl.duration(seconds=((pl.col("st_wh") + pl.col("st_mid")) * 3600).cast(pl.Int64))
          - pl.lit(DAY0)).dt.total_days()).alias("a"),
        pl.col("st_lm").clip(0.5, None).log().alias("l"),
    ).filter((pl.col("k") >= 0) & pl.col("a").is_between(0, ND - 1) & pl.col("p").is_between(0, ND - 1)
             & pl.col("l").is_finite())
    N = np.zeros((nk, ND, ND))
    S = np.zeros((nk, ND, ND))
    kk, pp, aa, ll = (k[c].to_numpy() for c in ("k", "p", "a", "l"))
    np.add.at(N, (kk, pp, aa), 1.0)
    np.add.at(S, (kk, pp, aa), ll)
    # 二维前缀和：沿支付日、揽收日累加（前面补 0 方便区间查询）
    N = np.pad(N.cumsum(1).cumsum(2), ((0, 0), (1, 0), (1, 0)))
    S = np.pad(S.cumsum(1).cumsum(2), ((0, 0), (1, 0), (1, 0)))
    return N, S


def _rect(C, k, p_lo, p_hi, a_lo, a_hi):
    """支付日 ∈ [p_lo, p_hi]、揽收日 ∈ [a_lo, a_hi] 的和（闭区间，已裁剪到网格）。"""
    p_lo = np.clip(p_lo, 0, ND - 1); p_hi = np.clip(p_hi, -1, ND - 1)
    a_lo = np.clip(a_lo, 0, ND - 1); a_hi = np.clip(a_hi, -1, ND - 1)
    ok = (p_hi >= p_lo) & (a_hi >= a_lo)
    out = (C[k, p_hi + 1, a_hi + 1] - C[k, p_lo, a_hi + 1] - C[k, p_hi + 1, a_lo] + C[k, p_lo, a_lo])
    return np.where(ok, out, 0.0)


def add_calendar(rows: pl.DataFrame, known: pl.DataFrame, c_mode: str, t0: datetime | None = None) -> pl.DataFrame:
    """rows 需已含 gap_d（add_recency 生成）。"""
    if c_mode == "train":
        c_time = pl.col("pay_time") - pl.duration(seconds=(pl.col("gap_d") * 86400).cast(pl.Int64))
    else:
        c_time = pl.lit(t0)
    # 预计揽收偏移：各渠道（流入→揽收）中位数，来自已知包裹
    off = known.group_by("shpp_chnl_id").agg(
        _off=((pl.col("st_wh") + pl.col("st_mid")).median() / 24.0))
    r = rows.with_columns(_c=((c_time - pl.lit(DAY0)).dt.total_days() - 1).cast(pl.Int32),
                          _iss=((pl.col("issue_time") - pl.lit(DAY0)).dt.total_seconds() / 86400.0))
    r = r.join(off, on="shpp_chnl_id", how="left").with_columns(pl.col("_off").fill_null(pl.col("_off").median()))
    c_day = r["_c"].to_numpy().astype(np.int64)
    dstar = np.floor(r["_iss"].to_numpy() + r["_off"].to_numpy()).astype(np.int64)
    new = {"cl_gap": (dstar - c_day).astype(np.float32)}
    for name, key in LEVELS.items():
        nk = int(max(known[key].max(), rows[key].max())) + 1
        N, S = _grid(known, key, nk)
        k = r[key].to_numpy().astype(np.int64)
        valid = k >= 0
        k = np.where(valid, k, 0)
        nb = _rect(N, k, c_day - BASE + 1, c_day, 0, ND - 1)
        sb = _rect(S, k, c_day - BASE + 1, c_day, 0, ND - 1)
        base = np.where(nb >= MIN_N, sb / np.maximum(nb, 1), np.nan)
        for tag, lo, hi in (("ahead", dstar - 1, dstar + 1), ("now", c_day - 2, c_day)):
            n = _rect(N, k, 0, c_day, lo, hi)
            s = _rect(S, k, 0, c_day, lo, hi)
            v = np.where((n >= MIN_N) & valid, s / np.maximum(n, 1) - base, np.nan)
            new[f"cl_{name}_{tag}"] = v.astype(np.float32)
            if tag == "ahead":
                new[f"cl_{name}_nahead"] = np.where(valid, n, np.nan).astype(np.float32)
    out = rows.with_columns([pl.Series(k2, v2, nan_to_null=True) for k2, v2 in new.items()])
    assert out.height == rows.height
    return out


CAL_FEATS = ["cl_gap"] + [f"cl_{n}_{t}" for n in LEVELS for t in ("ahead", "now", "nahead")]
