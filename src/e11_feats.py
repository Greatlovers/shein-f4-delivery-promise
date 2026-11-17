"""实验 11：两类新特征的快速筛选（9/12 折只训 q0.9，"同窗事后 P90 缩放"得分，与 e9/e10 同口径）。

1) 负荷特征 ld_*：同一渠道 / 仓×机场×模式 / 起飞机场 / 服务商，在本包裹流入日**之前**的完整日历日里
   流入了多少包裹（含未签收、含测试集包裹），以及相对前 14 天日均的偏离。
   - 只用流入日之前已结束的日子：上线时在仓库侧可实时得到，不看未来；也避开测试集 9/30 截断
     （9/30 当天支付的包裹不在数据里，若用"截至本包裹流入时刻的 24 小时"会在 9/30 少算）。
   - 数据从 8/1 支付开始，8/1–8/2 流入量不完整（7 月支付的包裹缺失）→ 基线只用 8/3 起的日子，不足 5 天置空。
2) 目的地近况 rc_cp_*：服务商 × 目的省份的近 3/7 天相对 28 天基线的偏离，做法与 recency.py 完全相同
   （同样的 0–9 天信息滞后模拟、只用已知标签）。

用法：python e11_feats.py base load dest both
"""
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import polars as pl

import recency
from common import FOLLOW_UP, WORK
from e10_alt import report
from exp import get_fold
from feats import BASE_FEATS, prepare_all
from model import fit_quantile, predict
from recency import RC_FEATS

FEATS = BASE_FEATS + RC_FEATS
T0 = datetime(2026, 9, 12)
LD_LEVELS = {"ch": ["shpp_chnl"], "wa": ["stk_cate_tp_nm", "start_airport_cd", "mode_category"],
             "air": ["start_airport_cd"], "car": ["lgs_company_nm"]}
LD_FEATS = [f"ld_{n}_{k}" for n in LD_LEVELS for k in ("n1", "rel1", "rel3")]
CP_LEVEL = {"cp": ["lgs_company_nm", "shpp_prvn_nm"]}
CP_FEATS = [f"rc_cp_{k}{w}" for w in recency.WINDOWS for k in ("d", "mid", "wh")] + ["rc_cp_n7", "rc_cp_base"]
DAY0 = datetime(2026, 8, 1)
FIRST_FULL = 2          # 8/3 起的流入日计数视为完整


def load_feats(df: pl.DataFrame) -> pl.DataFrame:
    path = WORK / "load_feats.parquet"
    if path.exists():
        return pl.read_parquet(path)
    d = df.select("pkg_no", *{k for v in LD_LEVELS.values() for k in v},
                  iday=((pl.col("issue_time") - pl.lit(DAY0)).dt.total_days()).cast(pl.Int32))
    nday = int(d["iday"].max()) + 1
    out = d.select("pkg_no", "iday")
    for name, keys in LD_LEVELS.items():
        cnt = d.group_by(keys + ["iday"]).agg(c=pl.len())
        grid = d.select(keys).unique().join(pl.DataFrame({"iday": np.arange(nday, dtype=np.int32)}), how="cross")
        g = grid.join(cnt, on=keys + ["iday"], how="left", nulls_equal=True).with_columns(
            c=pl.when(pl.col("iday") >= FIRST_FULL).then(pl.col("c").fill_null(0)))   # 不完整的日子置空
        g = g.sort(keys + ["iday"]).with_columns(
            p1=pl.col("c").shift(1).over(keys),
            p3=pl.col("c").shift(1).rolling_mean(3, min_samples=3).over(keys),
            b14=pl.col("c").shift(1).rolling_mean(14, min_samples=5).over(keys))
        g = g.select(keys + ["iday",
                             pl.col("p1").cast(pl.Float64).log1p().alias(f"ld_{name}_n1"),
                             ((pl.col("p1") + 1) / (pl.col("b14") + 1)).log().alias(f"ld_{name}_rel1"),
                             ((pl.col("p3") + 1) / (pl.col("b14") + 1)).log().alias(f"ld_{name}_rel3")])
        out = out.join(d.select("pkg_no", *keys, "iday").join(g, on=keys + ["iday"], how="left", nulls_equal=True)
                       .select("pkg_no", *[c for c in g.columns if c.startswith("ld_")]), on="pkg_no", how="left", maintain_order="left")
    assert out.height == d.height
    out = out.drop("iday")
    out.write_parquet(path)
    return out


def dest_feats(df, tr, va, t0):
    known = df.filter(~pl.col("is_test") & (pl.col("pay_time") < t0) & (pl.col("sign_time") < t0 + FOLLOW_UP))
    old = recency.LEVELS
    recency.LEVELS = CP_LEVEL
    try:
        cols = ["pkg_no", "pay_time"] + CP_LEVEL["cp"]
        a = recency.add_recency(tr.select(cols), known, "train").select(["pkg_no"] + CP_FEATS)
        b = recency.add_recency(va.select(cols), known, "fixed", t0).select(["pkg_no"] + CP_FEATS)
    finally:
        recency.LEVELS = old
    return a, b


def main(variants):
    t = time.time()
    tr, va = get_fold(T0, tag="_rc")
    df = prepare_all()
    ld = load_feats(df)
    tr = tr.join(ld, on="pkg_no", how="left", maintain_order="left")
    va = va.join(ld, on="pkg_no", how="left", maintain_order="left")
    if any(v in ("dest", "both") for v in variants):
        a, b = dest_feats(df, tr, va, T0)
        tr = tr.join(a, on="pkg_no", how="left", maintain_order="left")
        va = va.join(b, on="pkg_no", how="left", maintain_order="left")
    del df
    print(f"特征准备 {time.time() - t:.0f}s；验证集负荷特征缺失率 {va['ld_ch_rel3'].null_count() / va.height:.3f}", flush=True)
    feats = {"base": FEATS, "load": FEATS + LD_FEATS, "dest": FEATS + CP_FEATS, "both": FEATS + LD_FEATS + CP_FEATS,
             "loadrel": FEATS + [f for f in LD_FEATS if "_rel" in f]}
    for v in variants:
        m = fit_quantile(tr, 0.9, feats=feats[v], rounds=400)
        q = predict(m, va, feats[v])
        np.save(WORK / f"e11_0912_{v}_q900.npy", q)
        report(va, q, v, m._fit_secs)
        imp = m.feature_importance("gain")
        new = [(f, g) for f, g in zip(feats[v], imp / imp.sum() * 100) if f.startswith(("ld_", "rc_cp_"))]
        if new:
            print("   新特征增益占比 %.1f%%：" % sum(g for _, g in new),
                  "，".join(f"{f} {g:.1f}" for f, g in sorted(new, key=lambda x: -x[1])[:6]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
