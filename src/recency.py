"""近况（nowcast）特征：包裹决策时可见的“最近 3/7 天已知标签”相对“28 天基线”的偏离。

信息时点 c：
  - 测试包裹：c = 9/21 00:00（训练标签只到 9/20 支付），信息滞后 gap = pay_time - c ∈ [0, 9) 天
  - 验证包裹：c = 切分时刻 t0
  - 训练包裹：c = pay_time - gap，gap 由 pkg_no 哈希得到 U[0, 9) 天 —— 让模型学到“近况信息随滞后衰减”的规律，
    与测试时的信息结构一致
所有统计只来自 c 之前支付、且属于“已知”集合的包裹（在 make_fold 的 known 内），故无未来信息。
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

LEVELS = {
    "ch": ["shpp_chnl"],
    "car": ["lgs_company_nm"],
    "wa": ["stk_cate_tp_nm", "start_airport_cd", "mode_category"],
}
WINDOWS = (3, 7)
BASE_WINDOW = 28
MAX_GAP_DAYS = 9.0
DAY0 = datetime(2026, 7, 1)


def _daily(known: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """按 (keys, 支付日) 汇总可累加的统计量。

    求和用 numpy.bincount 按行顺序累加（确定性）：polars 的并行分组求和在大分组上会以不同顺序合并部分和，
    产生 1e-14 级的浮点差异，足以让 LightGBM 的分箱/分裂改变，导致两次运行结果不一致。"""
    d = known.select(*keys, day=((pl.col("pay_time") - pl.lit(DAY0)).dt.total_days()).cast(pl.Int32),
                     lr=pl.col("r").log(), lmid=(pl.col("st_mid").clip(0.5, None)).log(),
                     lwh=(pl.col("st_wh").clip(0.5, None)).log())
    d = d.with_columns(_g=(pl.struct(keys + ["day"]).rank("dense") - 1).cast(pl.Int64))
    g = d["_g"].to_numpy()
    ng = int(g.max()) + 1
    out = d.group_by("_g").agg([pl.col(k).first() for k in keys + ["day"]]).sort("_g")
    w = lambda c: d[c].fill_null(0.0).to_numpy().astype(np.float64)
    return out.with_columns(
        n=pl.Series(np.bincount(g, minlength=ng).astype(np.uint32)),
        s=pl.Series(np.bincount(g, weights=w("lr"), minlength=ng)),
        smid=pl.Series(np.bincount(g, weights=w("lmid"), minlength=ng)),
        swh=pl.Series(np.bincount(g, weights=w("lwh"), minlength=ng)),
    ).drop("_g")


def _cum(daily: pl.DataFrame, keys: list[str], last_day: int) -> pl.DataFrame:
    """补全为稠密 (keys × day) 网格并做累加，便于 O(1) 查询任意窗口和。"""
    grp = daily.select(keys).unique()
    days = pl.DataFrame({"day": np.arange(0, last_day + 1, dtype=np.int32)})
    grid = grp.join(days, how="cross")
    g = grid.join(daily, on=keys + ["day"], how="left", nulls_equal=True).fill_null(0)
    g = g.sort(keys + ["day"])
    return g.with_columns([pl.col(c).cum_sum().over(keys).alias(f"c_{c}") for c in ("n", "s", "smid", "swh")]).select(
        keys + ["day", "c_n", "c_s", "c_smid", "c_swh"])


def add_recency(rows: pl.DataFrame, known: pl.DataFrame, c_mode: str, t0: datetime | None = None,
                seed: int = 11) -> pl.DataFrame:
    """c_mode='train'：c = pay - 哈希 gap；c_mode='fixed'：c = t0。"""
    if c_mode == "train":
        u = (pl.col("pkg_no").hash(seed) % 1_000_003).cast(pl.Float64) / 1_000_003.0
        rows = rows.with_columns(gap_d=u * MAX_GAP_DAYS)
        rows = rows.with_columns(c_time=pl.col("pay_time") - pl.duration(seconds=(pl.col("gap_d") * 86400).cast(pl.Int64)))
    else:
        rows = rows.with_columns(c_time=pl.lit(t0), gap_d=(pl.col("pay_time") - pl.lit(t0)).dt.total_seconds() / 86400.0)
    # c 之前的最后一个完整支付日：c_day = floor(c) - 1（当天未结束的不算，避免同日邻居信息）
    rows = rows.with_columns(c_day=((pl.col("c_time") - pl.lit(DAY0)).dt.total_days() - 1).cast(pl.Int32))
    last_day = int(rows["c_day"].max())
    out = rows
    for name, keys in LEVELS.items():
        cum = _cum(_daily(known, keys), keys, last_day)
        def at(offset, tag):
            return cum.with_columns((pl.col("day") + offset).alias("c_day")).rename(
                {"c_n": f"n{tag}", "c_s": f"s{tag}", "c_smid": f"m{tag}", "c_swh": f"w{tag}"}).drop("day")
        out = out.join(at(0, "_0"), on=keys + ["c_day"], how="left", nulls_equal=True)
        for w in WINDOWS + (BASE_WINDOW,):
            out = out.join(at(w, f"_{w}"), on=keys + ["c_day"], how="left", nulls_equal=True)
        exprs = []
        nb = pl.col("n_0") - pl.col(f"n_{BASE_WINDOW}").fill_null(0)
        base = (pl.col("s_0") - pl.col(f"s_{BASE_WINDOW}").fill_null(0)) / nb
        base_m = (pl.col("m_0") - pl.col(f"m_{BASE_WINDOW}").fill_null(0)) / nb
        base_w = (pl.col("w_0") - pl.col(f"w_{BASE_WINDOW}").fill_null(0)) / nb
        for w in WINDOWS:
            nw = pl.col("n_0") - pl.col(f"n_{w}").fill_null(0)
            ok = nw >= 30
            exprs.append(pl.when(ok).then((pl.col("s_0") - pl.col(f"s_{w}").fill_null(0)) / nw - base).alias(f"rc_{name}_d{w}"))
            exprs.append(pl.when(ok).then((pl.col("m_0") - pl.col(f"m_{w}").fill_null(0)) / nw - base_m).alias(f"rc_{name}_mid{w}"))
            exprs.append(pl.when(ok).then((pl.col("w_0") - pl.col(f"w_{w}").fill_null(0)) / nw - base_w).alias(f"rc_{name}_wh{w}"))
            exprs.append(nw.cast(pl.Float64).alias(f"rc_{name}_n{w}"))
        exprs.append(pl.when(nb >= 30).then(base).alias(f"rc_{name}_base"))
        out = out.with_columns(exprs).drop([c for c in out.columns if c[:2] in ("n_", "s_", "m_", "w_") and c[2:].isdigit()])
    assert out.height == rows.height
    return out.drop(["c_time", "c_day"])


RC_FEATS = (["gap_d"] + [f"rc_{n}_{k}{w}" for n in LEVELS for w in WINDOWS for k in ("d", "mid", "wh")]
            + [f"rc_{n}_n7" for n in LEVELS] + [f"rc_{n}_base" for n in LEVELS])
