"""经验残差版条件分布 + 拉格朗日分配（最终采用的决策层）。

思路：分位数模型的 P90 作“位置/尺度”，残差 e = log(r) − log(P90) 的分布作“形状”。
形状按 (服务商 g, 离散度桶 b) 从校准窗口的真实时效中经验估计，b 由模型自身预测的尾部厚度
log(P97/P90) 分 5 桶 —— 既保留模型对“哪些包裹尾巴更长”的判断，又避免逐包裹尾部分位数的噪声。

对每个包裹，候选承诺 T_k = lag + P90·e^{z_k}（z_k 取 G_{g,b} 的细分位网格），
  a_k = P(y ≤ T_k) = G(z_k)，  b_k = P(y − T_k ≤ 24) = G(log(e^{z_k} + 24/P90))
选 argmin_k  T_k − λ_g·a_k − μ_g·b_k；可选“兜底”候选 T_floor（a、b 同样由 G 计算）。
"""
from __future__ import annotations

import numpy as np
import polars as pl

from policy import CAND_TAUS, drift_scenarios, exp_penalty, realized

REF = "q900"
DISP = ("q970", "q900")
NB = 5


class EmpShape:
    def __init__(self, g, ref, disp, r, nb=NB):
        """g: 服务商; ref: 模型 P90（剩余时长）; disp: 离散度; r: 真实剩余时长（校准窗口）。"""
        self.cuts, self.res = {}, {}
        e = np.log(np.maximum(r, 0.5)) - np.log(ref)
        for k in np.unique(g):
            m = g == k
            cuts = np.quantile(disp[m], np.linspace(0, 1, nb + 1)[1:-1]) if m.sum() > 5000 else np.array([])
            self.cuts[k] = cuts
            bk = np.searchsorted(cuts, disp[m])
            for b in range(len(cuts) + 1):
                self.res[(k, b)] = np.sort(e[m][bk == b])

    def bucket(self, k, disp):
        return np.searchsorted(self.cuts.get(k, np.array([])), disp)

    def cdf(self, key, z):
        s = self.res[key]
        return np.searchsorted(s, z, side="right") / len(s)

    def quant(self, key, taus):
        return np.quantile(self.res[key], taus)


def candidates_emp(shape: EmpShape, k, lag, ref, disp, floor_r=None, taus=CAND_TAUS):
    """单个服务商的候选承诺及其 (a, b)。返回 T (n,m), a, b。"""
    n = len(lag)
    bk = shape.bucket(k, disp)
    m_c = len(taus) + (1 if floor_r is not None else 0)
    T = np.empty((n, m_c))
    A = np.empty((n, m_c))
    B = np.empty((n, m_c))
    for b in np.unique(bk):
        key = (k, b) if (k, b) in shape.res else (k, 0)
        idx = np.where(bk == b)[0]
        z = shape.quant(key, taus)                                   # (m,)
        R = ref[idx, None] * np.exp(z)[None, :]
        if floor_r is not None:
            R = np.concatenate([floor_r[idx, None], R], axis=1)
        zz = np.log(R / ref[idx, None])
        A[idx] = shape.cdf(key, zz)
        B[idx] = shape.cdf(key, np.log((R + 24.0) / ref[idx, None]))
        T[idx] = lag[idx, None] + R
    return T, A, B


def choose(T, A, B, lam, mu):
    k = np.argmin(T - lam * A - mu * B, axis=1)
    return T[np.arange(T.shape[0]), k]


def _sub(arrs, n_max, seed=0):
    n = len(arrs[0])
    if n <= n_max:
        return arrs
    idx = np.random.default_rng(seed).choice(n, n_max, replace=False)
    return [a[idx] for a in arrs]


def fit_carrier(cands_cal, cal_y, cal_lag, cands_tgt, w_g, deltas, probs, strict=True,
                lam_grid=None, rho_grid=None, n_cal=150_000, n_tgt=100_000, win_w=None):
    """cands_cal: [(T,A,B), ...] 每个校准窗口一份；cal_y/cal_lag 同；cands_tgt: 目标集候选（只用于计算平均承诺成本）。
    最小化 w_g·mean_tgt(T) + 平均_窗口 E_δ[罚分]。为提速，大服务商在拟合时抽样（不影响最终全量应用）。"""
    lam_grid = np.geomspace(2, 30000, 32) if lam_grid is None else lam_grid
    rho_grid = np.array([0, 0.25, 0.5, 1, 2, 4, 8, 16, np.inf]) if rho_grid is None else rho_grid
    cands_tgt = tuple(_sub([x.astype(np.float32) for x in cands_tgt], n_tgt, 1))
    packed = [_sub([T.astype(np.float32), A.astype(np.float32), B.astype(np.float32), y, lag], n_cal, 2 + i)
              for i, ((T, A, B), y, lag) in enumerate(zip(cands_cal, cal_y, cal_lag))]
    cands_cal = [tuple(p[:3]) for p in packed]
    cal_y = [p[3] for p in packed]
    cal_lag = [p[4] for p in packed]
    win_w = np.ones(len(cands_cal)) if win_w is None else np.asarray(win_w, float)
    win_w = win_w / win_w.sum()

    def objective(lam, mu):
        cost = w_g * choose(*cands_tgt, lam, mu).mean()
        pen = 0.0
        for (T, A, B), y, lag, ww in zip(cands_cal, cal_y, cal_lag, win_w):
            on, sev = realized(y, lag, choose(T, A, B, lam, mu), deltas)
            pen += ww * exp_penalty(on, sev, probs, strict)
        return cost + pen, cost

    best = None
    for rho in rho_grid:
        for lam in lam_grid:
            l, m = (0.0, lam) if np.isinf(rho) else (lam, rho * lam)
            obj, cost = objective(l, m)
            if best is None or obj < best[0]:
                best = (obj, l, m, cost)
    obj0, l0, m0, _ = best
    for f in np.geomspace(0.75, 1.33, 13):
        for g2 in (0.8, 1.0, 1.25):
            l, m = l0 * f, m0 * f * g2
            obj, cost = objective(l, m)
            if obj < best[0]:
                best = (obj, l, m, cost)
    return dict(obj=best[0], lam=best[1], mu=best[2], cost=best[3])


def fit_carrier_scale(cal_list, ref_tgt, lag_tgt, w_g, deltas, probs, strict=True):
    """对照：T = lag + s·P90（单参数缩放）。cal_list: [(y, lag, ref), ...]"""
    best = None
    for s in np.geomspace(0.8, 2.5, 200):
        cost = w_g * (lag_tgt + s * ref_tgt).mean()
        pen = 0.0
        for y, lag, ref in cal_list:
            on, sev = realized(y, lag, lag + s * ref, deltas)
            pen += exp_penalty(on, sev, probs, strict)
        obj = cost + pen / len(cal_list)
        if best is None or obj < best[0]:
            best = (obj, s, cost)
    return dict(obj=best[0], s=best[1], cost=best[2])
