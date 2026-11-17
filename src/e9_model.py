"""实验 9：预测层改进的快速筛选（只训 q0.9，用"同窗事后 P90 缩放"得分衡量模型锐度）。

变体：base / city（加省×市类别）/ wrecent（近期样本加权，半衰期 14 天）/ long（800 轮、lr 0.05）/ d49（训练窗 49 天）
用法：python e9_model.py 2026-09-12 base city wrecent long d49
"""
import sys
import time
from datetime import datetime

import numpy as np
import polars as pl

from common import WORK, fmt_score
from exp import get_fold, oracle_eval
from feats import BASE_FEATS
from model import fit_quantile, predict
from recency import RC_FEATS

FEATS = BASE_FEATS + RC_FEATS


def add_pc(tr, va):
    """省×市联合编码：按训练集出现次数排序成稠密整数（LightGBM 类别特征按取值大小分配内存）；
    训练集中少于 200 件的组合并为 -1（缺失）。"""
    key = pl.col("shpp_prvn_nm_id").cast(pl.Int64) * 100000 + pl.col("shpp_city_nm_id")
    cnt = tr.group_by(key.alias("_k")).agg(n=pl.len()).filter(pl.col("n") >= 200).sort("n", descending=True)
    cnt = cnt.with_columns(pc_id=pl.int_range(pl.len()).cast(pl.Int32)).select(["_k", "pc_id"])
    f = lambda d: d.with_columns(_k=key).join(cnt, on="_k", how="left").with_columns(
        pl.col("pc_id").fill_null(-1)).drop("_k")
    return f(tr), f(va)


def run(t0, var, threads=8):
    days = 49 if var == "d49" else 35
    tr, va = get_fold(t0, train_days=days, tag="_rc")
    feats, params, rounds, w = FEATS, dict(num_threads=threads), 400, None
    if var == "city":
        tr, va = add_pc(tr, va)
        feats = FEATS + ["pc_id"]
        import model
        model.CAT_FEATS = model.CAT_FEATS + ["pc_id"]
    if var == "wrecent":
        age = tr.select((pl.lit(t0) - pl.col("pay_time")).dt.total_seconds())[:, 0].to_numpy() / 86400.0
        w = np.power(0.5, age / 14.0)
    if var == "long":
        rounds, params = 800, dict(num_threads=threads, learning_rate=0.05)
    if var == "r200":
        rounds = 200
    if var == "mdl2000":
        params = dict(num_threads=threads, min_data_in_leaf=2000)
    if var == "nl63":
        params = dict(num_threads=threads, num_leaves=63)
    t = time.time()
    if var == "ens3":
        lq = [np.log(predict(fit_quantile(tr, 0.9, feats=feats, rounds=rounds, params=dict(num_threads=threads, seed=s)), va, feats))
              for s in (42, 7, 99)]
        q = np.exp(np.mean(lq, axis=0))
    else:
        m = fit_quantile(tr, 0.9, feats=feats, rounds=rounds, weight=w, params=params)
        q = predict(m, va, feats)
    res, _ = oracle_eval(va, q)
    lr = np.log(np.maximum(va["r"].to_numpy(), 0.5))
    e = lr - np.log(q)
    pin = float(np.mean(np.maximum(0.9 * e, -0.1 * e)))
    cov = float((va["r"].to_numpy() <= q).mean())
    print(f"{t0:%m-%d} {var:8s} fit {time.time() - t:4.0f}s  pinball={pin:.5f}  cov={cov:.3f}  oracle {fmt_score(res).splitlines()[0]}", flush=True)


if __name__ == "__main__":
    t0 = datetime.strptime(sys.argv[1], "%Y-%m-%d")
    for v in sys.argv[2:]:
        run(t0, v)
