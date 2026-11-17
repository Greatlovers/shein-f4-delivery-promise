"""公共工具：读数、决策时点特征、历史统计特征、评分。

全部特征只使用包裹在 issue_time（订单流入）时点可得的信息：
  - 包裹属性（渠道、服务商、仓、区域、重量……）
  - pay_time / issue_time 及二者之差（已发生的“支付→流入”耗时）
  - 训练期（切分时刻之前）的历史统计：各渠道/服务商/区域的历史时效与各环节耗时分位数
环节时间戳只用于第三类“历史统计”，从不作为单个包裹的输入。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
DATA_END = datetime(2026, 11, 5, 0, 12, 1)  # 训练集最晚签收时间（数据整理时刻）
TEST_START = datetime(2026, 9, 21)
FOLLOW_UP = DATA_END - TEST_START  # 测试时可用标签相对切分时刻的“观察期”≈45 天

CAT_COLS = [
    "lgs_company_nm", "shpp_chnl", "stk_cate_tp_nm", "mode_category", "operation_type", "business_tp",
    "route_cat", "start_airport_cd", "end_airport_cd", "combine_flag", "combine_type", "sale_area",
    "send_country_nm", "shpp_prvn_nm",
]
BASE_COLS = [
    "pkg_no", "pkg_goods_cnt", "weight", "shpp_prvn_nm", "shpp_city_nm", "stk_cate_tp_nm", "shpp_chnl",
    "lgs_company_nm", "combine_flag", "combine_type", "start_airport_cd", "end_airport_cd", "mode_category",
    "business_tp", "operation_type", "route_cat", "sale_area", "send_country_nm", "pay_time", "issue_time",
]
STAGE_COLS = ["goods_send_time", "takeoff_time", "customs_clear_time", "accept_time", "outsend_time", "sign_time"]


def hours(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return (a - b).dt.total_seconds() / 3600.0


def load_train(stage: bool = True) -> pl.DataFrame:
    cols = BASE_COLS + (STAGE_COLS if stage else ["sign_time"]) + ["y_total_hours"]
    df = pl.read_parquet(WORK / "train.parquet", columns=list(dict.fromkeys(cols)))
    return _cast(df)


def load_test() -> pl.DataFrame:
    return _cast(pl.read_parquet(WORK / "test.parquet", columns=BASE_COLS))


def _cast(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([pl.col(c).cast(pl.Utf8) for c in CAT_COLS + ["shpp_city_nm"] if c in df.columns])


def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """决策时点即可知道的时间特征。"""
    return df.with_columns(
        lag_h=hours(pl.col("issue_time"), pl.col("pay_time")),
        pay_dow=pl.col("pay_time").dt.weekday().cast(pl.Int8),
        pay_hour=pl.col("pay_time").dt.hour().cast(pl.Int8),
        issue_dow=pl.col("issue_time").dt.weekday().cast(pl.Int8),
        issue_hod=(pl.col("issue_time").dt.hour() + pl.col("issue_time").dt.minute() / 60.0).cast(pl.Float32),
        issue_how=((pl.col("issue_time").dt.weekday().cast(pl.Int32) - 1) * 24 + pl.col("issue_time").dt.hour()).cast(pl.Int16),
        log_w=pl.col("weight").clip(1, None).log(),
        no_send_country=pl.col("send_country_nm").is_null().cast(pl.Int8),
    )


def score(g, y, T) -> dict:
    """与 tools/score.py 完全一致的计分公式（g：服务商，y：真值，T：预估）。"""
    g = np.asarray(g)
    y = np.asarray(y, float)
    T = np.asarray(T, float)
    me = abs(T.mean() - y.mean())
    pen_on = pen_sev = 0.0
    per = {}
    for k in np.unique(g):
        m = g == k
        r = float((y[m] <= T[m]).mean())
        s = float(((y[m] - T[m]) > 24).mean())
        pen_on += 5 * max(0.0, (0.90 - r) * 100)
        pen_sev += 5 * max(0.0, (s - 0.03) * 100)
        per[k] = (r, s, int(m.sum()))
    return dict(ME=me, meanT=T.mean(), meanY=y.mean(), pen_on=pen_on, pen_sev=pen_sev, B=me + pen_on,
                A=me + pen_on + pen_sev, per=per)


def fmt_score(res: dict) -> str:
    s = (f"A={res['A']:.2f} B={res['B']:.2f} ME={res['ME']:.2f} (meanT={res['meanT']:.1f} meanY={res['meanY']:.1f}) "
         f"pen_on={res['pen_on']:.2f} pen_sev={res['pen_sev']:.2f}\n")
    s += "   " + " | ".join(f"{k[-2:]}:{v[0]*100:.1f}/{v[1]*100:.1f}" for k, v in sorted(res["per"].items()))
    return s
