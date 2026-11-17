"""实验 3：决策层对比。

(a) 事后最优（同窗校准，无漂移）——衡量策略本身的效率：uniform vs lagr vs lagr+兜底
(b) 跨窗迁移——在上一窗口校准参数，用到下一窗口（与测试时的真实流程一致），比较不同漂移σ假设
"""
import sys

import numpy as np
import polars as pl

from common import WORK, fmt_score, score
from policy import (QCOLS, CondDist, calibrate_lagr, calibrate_uniform, candidates, choose_lagr, choose_uniform,
                    drift_scenarios)

FOLDS = ["0825", "0903", "0912"]


def load(tag):
    d = pl.read_parquet(WORK / f"pred_{tag}.parquet")
    return d


def prep(d, floor):
    lag = d["lag_h"].to_numpy()
    dist = CondDist(d.select(QCOLS).to_numpy())
    fr = None
    if floor == "tiny":
        fr = np.full(len(lag), 1.0)
    elif floor == "q30":
        fr = dist.quantile(np.array([0.3]))[:, 0]
    T, a, b = candidates(dist, lag, fr)
    return dist, lag, T, a, b


def fit_policy(d, kind, floor, sigma, shares):
    dist, lag, T, a, b = prep(d, floor)
    y = d["y_total_hours"].to_numpy()
    g = d["lgs_company_nm"].to_numpy()
    deltas, probs = drift_scenarios(sigma) if sigma > 0 else (np.array([0.0]), np.array([1.0]))
    params = {}
    for k in np.unique(g):
        m = g == k
        if kind == "uniform":
            params[k] = calibrate_uniform(dist.subset(m), y[m], lag[m], shares[k], deltas, probs)[0]
        else:
            params[k] = calibrate_lagr(T[m], a[m], b[m], y[m], lag[m], shares[k], deltas, probs)[:2]
    return params


def apply_policy(d, kind, floor, params):
    dist, lag, T, a, b = prep(d, floor)
    g = d["lgs_company_nm"].to_numpy()
    out = np.empty(len(lag))
    for k in np.unique(g):
        m = g == k
        p = params.get(k)
        if kind == "uniform":
            out[m] = choose_uniform(dist, lag, p)[m]
        else:
            out[m] = choose_lagr(T[m], a[m], b[m], *p)
    return out


def shares_of(d):
    s = d.group_by("lgs_company_nm").agg(pl.len()).with_columns(pl.col("len") / pl.col("len").sum())
    return dict(zip(s["lgs_company_nm"].to_list(), s["len"].to_list()))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "oracle"
    preds = {f: load(f) for f in FOLDS if (WORK / f"pred_{f}.parquet").exists()}
    if mode == "oracle":
        for f, d in preds.items():
            y, g = d["y_total_hours"].to_numpy(), d["lgs_company_nm"].to_numpy()
            sh = shares_of(d)
            print(f"\n=== {f} 事后最优（同窗校准、无漂移）")
            for kind, floor in [("uniform", None), ("lagr", None), ("lagr", "q30"), ("lagr", "tiny")]:
                p = fit_policy(d, kind, floor, 0.0, sh)
                T = apply_policy(d, kind, floor, p)
                print(f"[{kind:7s} floor={floor}]", fmt_score(score(g, y, T)), flush=True)
    else:
        sigmas = [float(s) for s in sys.argv[2:]] or [0.0, 0.03, 0.06]
        keys = list(preds)
        for i in range(1, len(keys)):
            dc, dt = preds[keys[i - 1]], preds[keys[i]]
            y, g = dt["y_total_hours"].to_numpy(), dt["lgs_company_nm"].to_numpy()
            sh = shares_of(dt)
            print(f"\n=== 迁移：{keys[i-1]} 校准 → {keys[i]} 评估")
            for kind, floor in [("uniform", None), ("lagr", None), ("lagr", "tiny")]:
                for s in sigmas:
                    p = fit_policy(dc, kind, floor, s, sh)
                    T = apply_policy(dt, kind, floor, p)
                    print(f"[{kind:7s} floor={floor} σ={s}]", fmt_score(score(g, y, T)), flush=True)
