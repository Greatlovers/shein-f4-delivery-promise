"""消融 2：近况特征（nowcast）的作用。"""
import numpy as np
from common import fmt_score, score
from exp import CUTS, get_fold, oracle_eval, pinball
from feats import BASE_FEATS
from model import fit_quantile, predict
from recency import RC_FEATS

configs = {"base": BASE_FEATS, "base+recency": BASE_FEATS + RC_FEATS}
for t0 in CUTS:
    tr, va = get_fold(t0, tag="_rc")
    y, lag, g = va["y_total_hours"].to_numpy(), va["lag_h"].to_numpy(), va["lgs_company_nm"].to_numpy()
    print(f"\n===== cut {t0:%m-%d} train={tr.height:,} val={va.height:,}", flush=True)
    for name, feats in configs.items():
        m = fit_quantile(tr, 0.9, feats=feats, rounds=300)
        Q = predict(m, va, feats)
        o, s = oracle_eval(va, Q)
        print(f"[{name}] fit={m._fit_secs:.0f}s pinball90={pinball(y - lag, Q, 0.9):.3f}")
        print("   raw   ", fmt_score(score(g, y, lag + Q)))
        print("   oracle", fmt_score(o))
        print("   scale ", {k[-2:]: round(float(v), 3) for k, v in s.items()}, flush=True)
        if name != "base":
            imp = sorted(zip(m.feature_importance("gain"), feats), reverse=True)
            tot = sum(x for x, _ in imp)
            print("   imp", [(f, round(v / tot * 100, 1)) for v, f in imp[:15]])
