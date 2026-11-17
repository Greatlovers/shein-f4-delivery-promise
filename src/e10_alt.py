"""实验 10：换模型 / 模型融合（CatBoost、XGBoost 与 LightGBM 融合）。

阶段一（筛选）：9/12 折只训 q0.9，用"同窗事后 P90 缩放"得分（与 e9 同口径）衡量单模型与融合（对数尺度加权平均）。
  python e10_alt.py screen cb xgb
  预测存到 work/alt_0912_<model>_q900.npy，融合在 e10_blend.py 里算，不用重训。
阶段二（迁移验证）：在各切分点训 q0.9、q0.97，写 work/pred_<mmdd>_<model>.parquet（与 e6_cuts 同格式）。
  python e10_alt.py cuts cb 2026-08-22 2026-08-25 ...
"""
import sys
import time
from datetime import datetime

import numpy as np
import polars as pl

from common import WORK, fmt_score
from exp import get_fold, oracle_eval
from feats import BASE_FEATS, CAT_FEATS
from model import TEST_CUT
from recency import RC_FEATS

FEATS = BASE_FEATS + RC_FEATS
THREADS = 12


def _cat_frame(d: pl.DataFrame):
    """CatBoost 用 pandas：类别列为非负整数（-1 → 最大值+1 的单独类别），数值列 float32。"""
    exprs = []
    for f in FEATS:
        if f in CAT_FEATS:
            exprs.append(pl.when(pl.col(f) < 0).then(9999).otherwise(pl.col(f)).cast(pl.Int32).alias(f))
        else:
            exprs.append(pl.col(f).cast(pl.Float32))
    return d.select(exprs).to_pandas()


def fit_predict_cb(tr, va, tau, seed=42, iters=1500, lr=0.12, depth=8):
    from catboost import CatBoostRegressor, Pool
    cats = [f for f in FEATS if f in CAT_FEATS]
    m = CatBoostRegressor(loss_function=f"Quantile:alpha={tau}", iterations=iters, learning_rate=lr, depth=depth,
                          l2_leaf_reg=3, border_count=254, one_hot_max_size=10, max_ctr_complexity=2,
                          bootstrap_type="Bernoulli", subsample=0.7, rsm=1.0, thread_count=THREADS,
                          random_seed=seed, verbose=250, allow_writing_files=False)
    t = time.time()
    m.fit(Pool(_cat_frame(tr), np.log(tr["r"].to_numpy()), cat_features=cats))
    fit_s = time.time() - t
    p = m.predict(Pool(_cat_frame(va), cat_features=cats), thread_count=THREADS)
    return np.exp(p), fit_s


def fit_predict_xgb(tr, va, tau, seed=42, rounds=400):
    import xgboost as xgb
    ftypes = ["c" if f in CAT_FEATS else "q" for f in FEATS]

    def mat(d, label=None):
        X = d.select([pl.when(pl.col(f) < 0).then(None).otherwise(pl.col(f)).alias(f) if f in CAT_FEATS else pl.col(f)
                      for f in FEATS]).to_numpy().astype(np.float32)
        return xgb.DMatrix(X, label=label, feature_names=FEATS, feature_types=ftypes, enable_categorical=True,
                           nthread=THREADS)

    p = dict(objective="reg:quantileerror", quantile_alpha=tau, tree_method="hist", grow_policy="lossguide",
             max_leaves=255, max_depth=0, min_child_weight=400, eta=0.08, subsample=0.7, colsample_bytree=0.8,
             max_bin=255, max_cat_to_onehot=1, max_cat_threshold=64, nthread=THREADS, seed=seed, verbosity=0)
    t = time.time()
    b = xgb.train(p, mat(tr, np.log(tr["r"].to_numpy())), num_boost_round=rounds)
    fit_s = time.time() - t
    return np.exp(b.predict(mat(va))), fit_s


FITTERS = {"cb": fit_predict_cb, "xgb": fit_predict_xgb}


def report(va, q, name, fit_s):
    res, _ = oracle_eval(va, q)
    lr = np.log(np.maximum(va["r"].to_numpy(), 0.5))
    e = lr - np.log(q)
    pin = float(np.mean(np.maximum(0.9 * e, -0.1 * e)))
    cov = float((va["r"].to_numpy() <= q).mean())
    print(f"09-12 {name:10s} fit {fit_s:5.0f}s  pinball={pin:.5f}  cov={cov:.3f}  oracle {fmt_score(res).splitlines()[0]}",
          flush=True)


def screen(models):
    t0 = datetime(2026, 9, 12)
    tr, va = get_fold(t0, tag="_rc")
    for name in models:
        q, fit_s = FITTERS[name](tr, va, 0.9)
        np.save(WORK / f"alt_0912_{name}_q900.npy", q)
        report(va, q, name, fit_s)


def cuts(name, dates):
    for t0 in dates:
        out_path = WORK / f"pred_{t0:%m%d}_{name}.parquet"
        if out_path.exists():
            print(f"{t0:%m-%d} 已存在，跳过", flush=True)
            continue
        tr, va = get_fold(t0, tag="_rc")
        out = va.select(["pkg_no", "lgs_company_nm", "lag_h", "y_total_hours"])
        for tau in (0.9, 0.97):
            q, fit_s = FITTERS[name](tr, va, tau)
            out = out.with_columns(pl.Series(f"q{int(round(tau * 1000)):03d}", q))
            print(f"  {t0:%m-%d} {name} tau={tau} fit {fit_s:.0f}s", flush=True)
        out.write_parquet(out_path)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "screen":
        screen(sys.argv[2:])
    elif mode == "cuts":
        cuts(sys.argv[2], [TEST_CUT if a == "test" else datetime.strptime(a, "%Y-%m-%d") for a in sys.argv[3:]])
