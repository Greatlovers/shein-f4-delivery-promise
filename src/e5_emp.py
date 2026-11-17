"""实验 5：经验残差 + 拉格朗日（emp） vs P90 缩放（scale）。

用法：
  python e5_emp.py oracle            同窗拟合形状与参数（无漂移），衡量策略效率
  python e5_emp.py transfer [σ...]   用之前所有窗口拟合，评估下一窗口（与测试流程一致）
"""
import sys

import numpy as np
import polars as pl

from common import WORK, fmt_score, score
from policy import drift_scenarios
from policy_emp import DISP, REF, EmpShape, candidates_emp, choose, fit_carrier, fit_carrier_scale

FOLDS = [f for f in ["0825", "0903", "0912"] if (WORK / f"pred_{f}.parquet").exists()]


def arrays(d):
    return dict(g=d["lgs_company_nm"].to_numpy(), lag=d["lag_h"].to_numpy(), y=d["y_total_hours"].to_numpy(),
                ref=d[REF].to_numpy(), disp=np.log(d[DISP[0]].to_numpy() / d[DISP[1]].to_numpy()))


def floor_of(x, kind):
    if kind == "tiny":
        return np.full(len(x["lag"]), 1.0)
    if kind == "half":
        return 0.6 * x["ref"]
    return None


def run(cal_folds, tgt_fold, sigma, floor, kinds=("emp", "scale"), win_w=None):
    cals = [arrays(pl.read_parquet(WORK / f"pred_{f}.parquet")) for f in cal_folds]
    tgt = arrays(pl.read_parquet(WORK / f"pred_{tgt_fold}.parquet"))
    pool = {k: np.concatenate([c[k] for c in cals]) for k in ("g", "ref", "disp", "y", "lag")}
    shape = EmpShape(pool["g"], pool["ref"], pool["disp"], pool["y"] - pool["lag"])
    deltas, probs = drift_scenarios(sigma) if sigma > 0 else (np.array([0.0]), np.array([1.0]))
    shares = {k: float((tgt["g"] == k).mean()) for k in np.unique(tgt["g"])}
    out = {}
    for kind in kinds:
        T = np.empty(len(tgt["lag"]))
        for k in np.unique(tgt["g"]):
            mt = tgt["g"] == k
            if kind == "emp":
                cc, ys, ls = [], [], []
                for c in cals:
                    m = c["g"] == k
                    cc.append(candidates_emp(shape, k, c["lag"][m], c["ref"][m], c["disp"][m], floor_of({"lag": c["lag"][m], "ref": c["ref"][m]}, floor)))
                    ys.append(c["y"][m]); ls.append(c["lag"][m])
                ct = candidates_emp(shape, k, tgt["lag"][mt], tgt["ref"][mt], tgt["disp"][mt], floor_of({"lag": tgt["lag"][mt], "ref": tgt["ref"][mt]}, floor))
                p = fit_carrier(cc, ys, ls, ct, shares[k], deltas, probs, win_w=win_w)
                T[mt] = choose(*ct, p["lam"], p["mu"])
            else:
                cl = [(c["y"][c["g"] == k], c["lag"][c["g"] == k], c["ref"][c["g"] == k]) for c in cals]
                p = fit_carrier_scale(cl, tgt["ref"][mt], tgt["lag"][mt], shares[k], deltas, probs)
                T[mt] = tgt["lag"][mt] + p["s"] * tgt["ref"][mt]
        res = score(tgt["g"], tgt["y"], T)
        out[kind] = res
        print(f"  [{kind:5s} σ={sigma} floor={floor} w={win_w}] {fmt_score(res)}", flush=True)
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "oracle":
        for f in FOLDS:
            print(f"\n=== {f} 事后（同窗）")
            run([f], f, 0.0, None)
            run([f], f, 0.0, "tiny", kinds=("emp",))
            run([f], f, 0.0, "half", kinds=("emp",))
    elif mode == "extra":
        for s in (0.03, 0.07):
            print("\n=== 迁移：0825 → 0903")
            run(["0825"], "0903", s, None, kinds=("emp",))
            print("\n=== 迁移：0903 → 0912")
            run(["0903"], "0912", s, None, kinds=("emp",))
        for s in (0.05, 0.07):
            print("\n=== 迁移：0825+0903 (权重1:2) → 0912")
            run(["0825", "0903"], "0912", s, None, kinds=("emp",), win_w=[1, 2])
    elif mode == "extra2":
        print("\n=== 迁移：0825 → 0903")
        run(["0825"], "0903", 0.06, None, kinds=("emp",))
        print("\n=== 迁移：0903 → 0912")
        run(["0903"], "0912", 0.06, None, kinds=("emp",))
        print("\n=== 迁移：0825+0903 (权重1:4) → 0912")
        run(["0825", "0903"], "0912", 0.06, None, kinds=("emp",), win_w=[1, 4])
    else:
        sigmas = [float(s) for s in sys.argv[2:]] or [0.0, 0.04, 0.08]
        for i in range(1, len(FOLDS)):
            for cal in ([FOLDS[i - 1]], FOLDS[:i]) if i > 1 else ([FOLDS[i - 1]],):
                print(f"\n=== 迁移：{'+'.join(cal)} → {FOLDS[i]}")
                for s in sigmas:
                    run(cal, FOLDS[i], s, None, kinds=("emp",) if s != 0.05 else ("emp", "scale"))
