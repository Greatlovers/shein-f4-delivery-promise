"""把原始 CSV 转成 parquet（类别列转为整数编码前保留原字符串，时间戳解析为 datetime）。

用法：python -I src/00_convert.py <资料包目录> <输出目录>
"""
import sys
import time
from pathlib import Path

import polars as pl

RAW = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

BASE_STR = [
    "pkg_no", "shpp_country_nm", "shpp_prvn_nm", "shpp_city_nm", "stk_cate_tp_nm", "shpp_chnl",
    "lgs_company_nm", "combine_flag", "combine_type", "start_airport_cd", "end_airport_cd",
    "mode_category", "pay_mthd", "business_tp", "operation_type", "route_cat", "sale_area",
    "send_country_nm", "shpp_post_no",
]
TS_FMT = "%Y-%m-%d %H:%M:%S"


def convert(name: str, out_name: str):
    t = time.time()
    path = RAW / "data" / name
    header = path.open(encoding="utf-8").readline().strip().split(",")
    schema = {}
    for c in header:
        if c == "pkg_goods_cnt":
            schema[c] = pl.Int32
        elif c in ("weight", "y_total_hours"):
            schema[c] = pl.Float64
        else:
            schema[c] = pl.Utf8
    lf = pl.scan_csv(path, schema=schema)
    ts_cols = [c for c in header if c not in BASE_STR and c not in ("pkg_goods_cnt", "weight", "y_total_hours")]
    lf = lf.with_columns([pl.col(c).str.strptime(pl.Datetime("us"), TS_FMT, strict=False) for c in ts_cols])
    lf = lf.with_columns([pl.col(c).cast(pl.Categorical) for c in BASE_STR if c != "pkg_no"])
    df = lf.collect()
    df.write_parquet(OUT / out_name, compression="zstd")
    print(name, df.shape, f"{time.time() - t:.1f}s")
    # 检查解析失败（非空字符串却解析成 null）
    return df


convert("test_202609.csv", "test.parquet")
convert("train_202608_0920.csv", "train.parquet")
