"""特征构建。

prepare_all()  —— 合并训练/测试，生成决策时点特征与整数编码，缓存到 work/all.parquet
hist_features(df, known, t0) —— 用“切分时刻前已知”的训练数据计算历史统计特征（只用历史，不用本包裹的环节时间）
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl

from common import CAT_COLS, STAGE_COLS, WORK, add_time_features, hours, load_test, load_train

CODE_COLS = CAT_COLS + ["shpp_city_nm"]


def prepare_all(force: bool = False) -> pl.DataFrame:
    path = WORK / "all.parquet"
    if path.exists() and not force:
        return pl.read_parquet(path)
    tr = load_train(stage=True).with_columns(is_test=pl.lit(False))
    te = load_test().with_columns(is_test=pl.lit(True))
    df = pl.concat([tr, te], how="diagonal_relaxed")
    df = add_time_features(df)
    # 环节耗时（仅供历史统计使用；作为特征时永远不会读取本包裹自己的值）
    df = df.with_columns(
        r=pl.col("y_total_hours") - pl.col("lag_h"),                                   # 流入→签收
        st_wh=hours(pl.col("goods_send_time"), pl.col("issue_time")),                    # 仓内
        st_lm=hours(pl.col("sign_time"), pl.col("accept_time")),                         # 尾程揽收→签收
        st_mid=hours(pl.col("accept_time"), pl.col("goods_send_time")),                  # 出库→尾程揽收（头程+清关）
    )
    # 类别整数编码（训练+测试统一字典；null -> -1）
    for c in CODE_COLS:
        vals = df[c].drop_nulls().unique().sort()
        mapping = pl.DataFrame({c: vals, f"{c}_id": np.arange(len(vals), dtype=np.int32)})
        df = df.join(mapping, on=c, how="left").with_columns(pl.col(f"{c}_id").fill_null(-1))
    df = df.drop(STAGE_COLS[:-1])  # 原始环节时间戳不再需要
    df.write_parquet(path)
    return df


def _smoothed(known: pl.DataFrame, keys: list[str], col: str, prior_keys: list[str] | None, name: str, k: float,
              q: float = 0.5) -> pl.DataFrame:
    """分组分位数，并按样本量向上一级分组收缩（经验贝叶斯式平滑）。"""
    g = known.group_by(keys).agg(v=pl.col(col).quantile(q), n=pl.len())
    if prior_keys is None:
        return g.rename({"v": name}).drop("n")
    p = known.group_by(prior_keys).agg(pv=pl.col(col).quantile(q))
    g = g.join(p, on=prior_keys, how="left")
    g = g.with_columns(((pl.col("v") * pl.col("n") + pl.col("pv") * k) / (pl.col("n") + k)).alias(name))
    return g.select(keys + [name])


def hist_features(df: pl.DataFrame, kn: pl.DataFrame) -> pl.DataFrame:
    """在 df 上拼接历史统计特征。kn 是统计来源：切分时刻前“已签收”的训练样本（已按时间窗/留一周筛好）。"""
    # 尾程（揽收→签收）的地理差异：服务商×城市，向服务商×省份收缩
    lm_prov = _smoothed(kn, ["lgs_company_nm", "shpp_prvn_nm"], "st_lm", ["lgs_company_nm"], "h_lm_prov", 50)
    # 城市级：收缩先验用省份（需要把省份带上）
    c = kn.group_by(["lgs_company_nm", "shpp_prvn_nm", "shpp_city_nm"]).agg(v=pl.col("st_lm").median(), n=pl.len())
    c = c.join(lm_prov, on=["lgs_company_nm", "shpp_prvn_nm"], how="left")
    c = c.with_columns(((pl.col("v") * pl.col("n") + pl.col("h_lm_prov") * 30) / (pl.col("n") + 30)).alias("h_lm_city"))
    lm_city = c.select(["lgs_company_nm", "shpp_prvn_nm", "shpp_city_nm", "h_lm_city"])
    lm_city90 = _smoothed(kn, ["lgs_company_nm", "shpp_prvn_nm", "shpp_city_nm"], "st_lm", ["lgs_company_nm"],
                          "h_lm_city90", 50, 0.9)
    # 城市整体时效残差（相对渠道中位数）：捕捉偏远地区
    kn2 = kn.join(kn.group_by("shpp_chnl").agg(ch_med=pl.col("r").median()), on="shpp_chnl")
    kn2 = kn2.with_columns(res=pl.col("r") - pl.col("ch_med"))
    city_res = _smoothed(kn2, ["shpp_prvn_nm", "shpp_city_nm"], "res", None, "h_city_res", 0)
    cnt = kn2.group_by(["shpp_prvn_nm", "shpp_city_nm"]).agg(n=pl.len())
    city_res = city_res.join(cnt, on=["shpp_prvn_nm", "shpp_city_nm"]).with_columns(
        (pl.col("h_city_res") * pl.col("n") / (pl.col("n") + 30)).alias("h_city_res")).drop("n")
    prov_res = kn2.group_by("shpp_prvn_nm").agg(h_prov_res=pl.col("res").median(), h_prov_res90=pl.col("res").quantile(0.9))
    # 渠道级：仓内/中段/尾程的中位与 P90（对新渠道无值，交给模型的缺失分支）
    ch = kn.group_by("shpp_chnl").agg(
        h_ch_wh50=pl.col("st_wh").median(), h_ch_wh90=pl.col("st_wh").quantile(0.9),
        h_ch_mid50=pl.col("st_mid").median(), h_ch_mid90=pl.col("st_mid").quantile(0.9),
        h_ch_lm50=pl.col("st_lm").median(), h_ch_lm90=pl.col("st_lm").quantile(0.9),
        h_ch_r50=pl.col("r").median(), h_ch_r90=pl.col("r").quantile(0.9), h_ch_r97=pl.col("r").quantile(0.97),
        h_ch_n=pl.len().cast(pl.Float64),
    )
    # 仓×机场×发货时段的“仓内+头程”中位数（跨服务商共享，可迁移到新渠道）
    wa = kn.group_by(["stk_cate_tp_nm", "start_airport_cd", "mode_category", "operation_type"]).agg(
        h_wa_mid50=(pl.col("st_wh") + pl.col("st_mid")).median(),
        h_wa_mid90=(pl.col("st_wh") + pl.col("st_mid")).quantile(0.9))
    out = (df.join(lm_prov, on=["lgs_company_nm", "shpp_prvn_nm"], how="left")
             .join(lm_city, on=["lgs_company_nm", "shpp_prvn_nm", "shpp_city_nm"], how="left")
             .join(lm_city90, on=["lgs_company_nm", "shpp_prvn_nm", "shpp_city_nm"], how="left")
             .join(city_res, on=["shpp_prvn_nm", "shpp_city_nm"], how="left")
             .join(prov_res, on="shpp_prvn_nm", how="left")
             .join(ch, on="shpp_chnl", how="left")
             .join(wa, on=["stk_cate_tp_nm", "start_airport_cd", "mode_category", "operation_type"], how="left", nulls_equal=True))
    assert out.height == df.height, "历史特征拼接出现重复行"
    return out


HIST_COLS = ["h_lm_prov", "h_lm_city", "h_lm_city90", "h_city_res", "h_prov_res", "h_prov_res90", "h_ch_wh50",
             "h_ch_wh90", "h_ch_mid50", "h_ch_mid90", "h_ch_lm50", "h_ch_lm90", "h_ch_r50", "h_ch_r90", "h_ch_r97",
             "h_ch_n", "h_wa_mid50", "h_wa_mid90"]
BASE_FEATS = ([f"{c}_id" for c in CAT_COLS] + ["pkg_goods_cnt", "weight", "lag_h", "pay_dow",
              "pay_hour", "issue_dow", "issue_hod", "issue_how", "no_send_country"])
CAT_FEATS = [f"{c}_id" for c in CAT_COLS if c not in ("send_country_nm",)]
