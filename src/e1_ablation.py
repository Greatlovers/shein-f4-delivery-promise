"""消融 1：历史统计特征的作用（q0.9 单模型；看 pinball、原始得分与事后校准下的平均承诺）。"""
import sys
import numpy as np
from common import fmt_score, score
from exp import CUTS, get_fold, oracle_eval, pinball
from feats import BASE_FEATS, HIST_COLS
from model import fit_quantile, predict

configs = {"base": BASE_FEATS, "base+hist": BASE_FEATS + HIST_COLS}
for t0 in CUTS:
    tr, va = get_fold(t0)
    y, lag, g = va["y_total_hours"].to_numpy(), va["lag_h"].to_numpy(), va["lgs_company_nm"].to_numpy()
    print(f"\n===== cut {t0:%m-%d} train={tr.height:,} val={va.height:,}", flush=True)
    for name, feats in configs.items():
        m = fit_quantile(tr, 0.9, feats=feats, rounds=300)
        Q = predict(m, va, feats)
        res = score(g, y, lag + Q)
        o, s = oracle_eval(va, Q)
        print(f"[{name}] fit={m._fit_secs:.0f}s pinball90={pinball(y - lag, Q, 0.9):.3f}")
        print("   raw   ", fmt_score(res))
        print("   oracle", fmt_score(o))
        print("   scale ", {k[-2:]: round(v, 3) for k, v in s.items()}, flush=True)
