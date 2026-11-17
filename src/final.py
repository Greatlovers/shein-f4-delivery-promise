"""最终流水线：训练 → 测试集条件分位数 → 稀有渠道兜底 → 决策层（经验残差 + 拉格朗日）→ 提交文件。

用法：python final.py [--skip-train]
完整顺序：train_mq.py 2026-09-12 → train_mq.py test → ens_pred.py 2026-09-12 test → final.py --skip-train
依赖：work/pred_0825|0903|0912.parquet（三个校准窗口，各自由“当时”的模型预测，含真实时效）
配置：config/final_cfg.json（σ 漂移、新线路保险系数等，由 e7 迁移实验与留渠道验证确定）
输出：outputs/submission.csv，outputs/test_detail.parquet，outputs/params.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl

from common import ROOT, WORK
from exp import get_fold
from fallback import fit_fallback, rare_mask
from model import TEST_CUT
from policy import QCOLS, CondDist, drift_scenarios
from policy_emp import DISP, REF, EmpShape, candidates_emp, choose, fit_carrier
from train_mq import run as train_run

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
CFG = ROOT / "config" / "final_cfg.json"
cfg = json.loads(CFG.read_text(encoding="utf-8")) if CFG.exists() else {}
CALIBS = cfg.get("calib_windows", ["0912"])   # 校准窗口：迁移验证表明“只用最近窗口”最好
SIGMA = cfg.get("sigma", 0.06)            # 漂移情景标准差（对数尺度）
BIAS = cfg.get("bias", 0.0)
NEWLINE_TAU = cfg.get("newline_tau", 0.995)   # LSP_0001 新线路：兜底 P99.5 × 1.15（留渠道验证最差渠道仍达标）
NEWLINE_MULT = cfg.get("newline_mult", 1.15)
RARE_TAU = cfg.get("rare_tau", 0.99)           # 其他稀有渠道（占各自服务商 <3%）：兜底 P99
WIN_W = cfg.get("win_w", None)            # 校准窗口权重（越近越大：漂移有持续性）
SUF = cfg.get("pred_suffix", "")          # "_ens"：q900/q970 用 3 个种子几何平均的预测（ens_pred.py 生成）


def arrays(d: pl.DataFrame, Q: np.ndarray | None = None):
    q = d.select(QCOLS).to_numpy() if Q is None else Q
    ref = q[:, QCOLS.index(REF)]
    disp = np.log(q[:, QCOLS.index(DISP[0])] / ref)
    out = dict(g=d["lgs_company_nm"].to_numpy(), lag=d["lag_h"].to_numpy(), ref=ref, disp=disp)
    if "y_total_hours" in d.columns and d["y_total_hours"].null_count() == 0:
        out["y"] = d["y_total_hours"].to_numpy()
    return out


def main(skip_train: bool):
    # 1) 测试集条件分位数（训练：9/21 前支付、近 35 天的全部已签收包裹）
    p_test = WORK / "pred_0921.parquet"
    if not skip_train or not p_test.exists():
        train_run(TEST_CUT)
    if SUF:
        p_test = WORK / f"pred_0921{SUF}.parquet"
        assert p_test.exists(), f"缺少 {p_test.name}：先运行 python ens_pred.py 2026-09-12 test"
    te = pl.read_parquet(p_test)
    tr, va = get_fold(TEST_CUT, tag="_rc")
    assert te["pkg_no"].to_list() == va["pkg_no"].to_list()

    # 2) 稀有/新渠道 → 同行线路兜底模型
    rare = rare_mask(tr, va)
    p_fb = WORK / "pred_0921_fallback.parquet"
    if not skip_train or not p_fb.exists():
        fit_fallback(tr, va.filter(pl.Series(rare))).write_parquet(p_fb)
    fb = pl.read_parquet(p_fb)
    Q = te.select(QCOLS).to_numpy().copy()
    Q[rare] = np.sort(fb.select(QCOLS).to_numpy(), axis=1)
    print(f"稀有/新渠道包裹 {rare.sum():,} 个，改用同行线路兜底模型")

    # 3) 经验残差形状（三个校准窗口合并）
    cals = [arrays(pl.read_parquet(WORK / f"pred_{f}{SUF}.parquet")) for f in CALIBS]
    pool = {k: np.concatenate([c[k] for c in cals]) for k in ("g", "ref", "disp", "y", "lag")}
    shape = EmpShape(pool["g"], pool["ref"], pool["disp"], pool["y"] - pool["lag"])
    tgt = arrays(te, Q)
    g = tgt["g"]
    shares = {k: float((g == k).mean()) for k in np.unique(g)}
    deltas, probs = drift_scenarios(SIGMA, BIAS)

    # 4) 每个服务商拟合 (λ, μ)：最小化 w_g·mean_test(T) + 平均_窗口 E_δ[罚分]
    T = np.empty(len(g))
    params = {}
    for k in np.unique(g):
        mt = (g == k) & ~rare
        cc, ys, ls = [], [], []
        for c in cals:
            m = c["g"] == k
            cc.append(candidates_emp(shape, k, c["lag"][m], c["ref"][m], c["disp"][m]))
            ys.append(c["y"][m])
            ls.append(c["lag"][m])
        ct = candidates_emp(shape, k, tgt["lag"][mt], tgt["ref"][mt], tgt["disp"][mt])
        p = fit_carrier(cc, ys, ls, ct, shares[k], deltas, probs, win_w=WIN_W)
        T[mt] = choose(*ct, p["lam"], p["mu"])
        # 校准窗口上的期望表现（δ=0）
        diag = []
        for (Tc, A, B), y, lag in zip(cc, ys, ls):
            t = choose(Tc, A, B, p["lam"], p["mu"])
            diag.append(((y <= t).mean() * 100, ((y - t) > 24).mean() * 100))
        p["calib_on_sev"] = diag
        params[k] = p
        print(f"  {k}: 占比={shares[k]:.4f} λ={p['lam']:.1f} μ={p['mu']:.1f} 测试平均承诺={T[mt].mean():.1f} "
              f"校准窗(准时/严重)={[f'{a:.1f}/{b:.1f}' for a, b in diag]}", flush=True)

    # 5) 稀有/新渠道：没有可校准的历史，取兜底模型高分位 × 保险系数（属于各自服务商，计入其准时率）
    dist_fb = CondDist(Q)
    newline = rare & (g == "LSP_0001")
    other = rare & ~newline
    T[newline] = tgt["lag"][newline] + dist_fb.subset(newline).quantile(np.array([NEWLINE_TAU]))[:, 0] * NEWLINE_MULT
    T[other] = tgt["lag"][other] + dist_fb.subset(other).quantile(np.array([RARE_TAU]))[:, 0]
    print(f"LSP_0001 新线路 {newline.sum():,} 件，平均承诺 {T[newline].mean():.1f}h；其他稀有渠道 {other.sum():,} 件，平均承诺 {T[other].mean():.1f}h")
    assert np.isfinite(T).all() and (T > 0).all()

    sub = pl.DataFrame({"pkg_no": te["pkg_no"], "pred_hours": np.round(T, 4)})
    assert sub["pkg_no"].n_unique() == sub.height == 911_415
    sub.write_csv(OUT / "submission.csv")
    detail = te.select("pkg_no", "lgs_company_nm", "lag_h").with_columns(
        pred_hours=T, rare=rare, shpp_chnl=va["shpp_chnl"], pay_time=va["pay_time"], q900=Q[:, QCOLS.index("q900")])
    detail.write_parquet(OUT / "test_detail.parquet")
    print(detail.group_by("lgs_company_nm").agg(n=pl.len(), meanT=pl.col("pred_hours").mean(),
                                                  rare=pl.col("rare").sum()).sort("n", descending=True))
    print("整体平均承诺:", T.mean())
    (OUT / "params.json").write_text(json.dumps(
        {"sigma": SIGMA, "bias": BIAS, "newline_tau": NEWLINE_TAU, "newline_mult": NEWLINE_MULT, "rare_tau": RARE_TAU,
         "win_w": WIN_W, "calib_windows": CALIBS, "pred_suffix": SUF,
         "carriers": {k: {kk: (vv if not isinstance(vv, (np.floating,)) else float(vv)) for kk, vv in v.items()}
                      for k, v in params.items()}}, indent=1, ensure_ascii=False, default=float), encoding="utf-8")


if __name__ == "__main__":
    main("--skip-train" in sys.argv)
