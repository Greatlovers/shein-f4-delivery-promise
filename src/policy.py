"""决策层：把每个包裹的条件分位数变成承诺时效 T。

记号：r = 流入→签收的剩余时长；T = lag + r_T（lag = 支付→流入，决策时已知）。
条件分布 F_i 由多分位数模型给出（τ 网格上的分位数，在对数时长上线性插值，尾部指数外推）。

策略族（每个服务商 g 一组参数）：
  uniform : T_i = lag_i + Q_i(τ_g)                               —— 统一分位数水平
  lagr    : T_i = argmin_T  T − λ_g·F_i(T) − μ_g·F_i(T+24)        —— 拉格朗日最优分配（两约束对偶）
            候选 T 取 Q_i 在细 τ 网格上的值，外加“兜底承诺” T_floor（可选，用于放弃极难预测的包裹）

参数选择（calibrate）：在校准集（上一时间窗，有真实时效）上，最小化
     w_g · mean_g(T) + 5·E_δ[ max(0, 90 − r_g(δ)) + max(0, s_g(δ) − 3) ]
  其中 δ 为“下个窗口的整体快慢漂移”情景（对剩余时长乘 e^δ），w_g 为该服务商在测试集中的包裹占比。
  这一目标与赛题得分逐项对应（ME 项对每个服务商可分），因此各服务商可以独立优化。
"""
from __future__ import annotations

import numpy as np
import polars as pl

TAUS = np.array([0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995])
QCOLS = [f"q{int(round(t * 1000)):03d}" for t in TAUS]
# 候选承诺的分位水平（细网格）
CAND_TAUS = np.concatenate([np.arange(0.50, 0.90, 0.02), np.arange(0.90, 0.99, 0.005), [0.99, 0.993, 0.995, 0.997, 0.999]])


class CondDist:
    """每个包裹的条件分布（剩余时长 r），对数空间分段线性 + 指数尾。"""

    def __init__(self, Q: np.ndarray):
        Q = np.sort(np.maximum(Q, 1.0), axis=1)          # 修正分位数交叉
        lq = np.log(Q)
        # 尾部：用 q99、q995 推指数尾斜率（对数时长每增加 beta，超越概率减半）
        beta = np.maximum((lq[:, -1] - lq[:, -2]) / np.log(2.0), 0.02)
        ext_t = np.array([0.997, 0.999, 0.9995])
        ext = lq[:, -1:] + beta[:, None] * np.log2(0.005 / (1 - ext_t))[None, :]
        # 低端：q50 以下按 q50 与 q70 的斜率外推到 τ=0.02（只用于兜底承诺的概率评估）
        lo_t = np.array([0.02, 0.1, 0.3])
        slope = (lq[:, 1] - lq[:, 0]) / (TAUS[1] - TAUS[0])
        lo = lq[:, :1] + slope[:, None] * (lo_t - 0.5)[None, :]
        self.taus = np.concatenate([lo_t, TAUS, ext_t])
        self.lq = np.concatenate([lo, lq, ext], axis=1)
        self.beta = beta

    def subset(self, m):
        new = object.__new__(CondDist)
        new.taus, new.lq, new.beta = self.taus, self.lq[m], self.beta[m]
        return new

    def quantile(self, tau: np.ndarray) -> np.ndarray:
        """tau 为 1 维数组（所有包裹相同的 τ 网格）→ (n, len(tau)) 的剩余时长。"""
        out = np.empty((self.lq.shape[0], len(tau)))
        for k, t in enumerate(tau):
            j = np.clip(np.searchsorted(self.taus, t) - 1, 0, len(self.taus) - 2)
            w = (t - self.taus[j]) / (self.taus[j + 1] - self.taus[j])
            out[:, k] = np.exp(self.lq[:, j] * (1 - w) + self.lq[:, j + 1] * w)
        return out

    def cdf(self, r: np.ndarray) -> np.ndarray:
        """r: (n, m) 剩余时长 → F(r)。"""
        lr = np.log(np.maximum(r, 1e-3))
        n, m = lr.shape
        F = np.zeros((n, m))
        lq = self.lq
        K = lq.shape[1]
        below = lr < lq[:, :1]
        F[below] = 0.0  # 低于 τ=0.02 的点（近似 0）
        for j in range(K - 1):
            a, b = lq[:, j:j + 1], lq[:, j + 1:j + 2]
            msk = (lr >= a) & (lr < b)
            w = (lr - a) / np.maximum(b - a, 1e-9)
            F = np.where(msk, self.taus[j] + w * (self.taus[j + 1] - self.taus[j]), F)
        top = lr >= lq[:, -1:]
        tail = 1 - (1 - self.taus[-1]) * np.power(2.0, -(lr - lq[:, -1:]) / self.beta[:, None])
        F = np.where(top, tail, F)
        return F


def candidates(dist: CondDist, lag: np.ndarray, floor_r: np.ndarray | None = None):
    """返回候选承诺 T (n, m)、对应 P(准时)=a、P(不严重超时)=b。"""
    R = dist.quantile(CAND_TAUS)
    if floor_r is not None:
        R = np.concatenate([floor_r[:, None], R], axis=1)
    a = dist.cdf(R)
    b = dist.cdf(R + 24.0)
    return lag[:, None] + R, a, b


def choose_lagr(T, a, b, lam, mu):
    k = np.argmin(T - lam * a - mu * b, axis=1)
    return T[np.arange(T.shape[0]), k]


def choose_uniform(dist: CondDist, lag, tau):
    return lag + dist.quantile(np.array([tau]))[:, 0]


def realized(y, lag, T, deltas):
    """在漂移情景 δ 下（剩余时长乘 e^δ）的准时率与严重超时占比（百分数）。"""
    r = y - lag
    on = np.empty(len(deltas))
    sev = np.empty(len(deltas))
    for i, d in enumerate(deltas):
        yy = lag + r * np.exp(d)
        on[i] = (yy <= T).mean() * 100
        sev[i] = ((yy - T) > 24).mean() * 100
    return on, sev


def exp_penalty(on, sev, probs, strict=True):
    p = np.maximum(0, 90 - on) + (np.maximum(0, sev - 3) if strict else 0)
    return 5 * float((p * probs).sum())


def drift_scenarios(sigma: float, bias: float = 0.0, n: int = 21):
    """正态漂移情景（对数尺度）及其概率权重。"""
    z = np.linspace(-3, 3, n)
    p = np.exp(-z ** 2 / 2)
    p /= p.sum()
    return bias + sigma * z, p


def calibrate_lagr(T, a, b, y, lag, w_g, deltas, probs, lam_grid=None, rho_grid=None, strict=True):
    """二维网格搜索 (λ, μ=ρ·λ)，最小化 w_g·mean(T) + 期望罚分。返回 (λ, μ, 目标值, meanT, on, sev)。"""
    lam_grid = np.geomspace(5, 20000, 36) if lam_grid is None else lam_grid
    rho_grid = np.array([0, 0.25, 0.5, 1, 2, 4, 8, 1e3]) if rho_grid is None else rho_grid
    best = None
    for rho in rho_grid:
        for lam in lam_grid:
            lam_eff, mu = (lam, rho * lam) if rho < 1e3 else (0.0, lam)
            Tc = choose_lagr(T, a, b, lam_eff, mu)
            on, sev = realized(y, lag, Tc, deltas)
            obj = w_g * Tc.mean() + exp_penalty(on, sev, probs, strict)
            if best is None or obj < best[2]:
                best = (lam_eff, mu, obj, Tc.mean(), on[len(on) // 2], sev[len(sev) // 2])
    # 在最优点附近细化 λ
    lam0, mu0 = best[0], best[1]
    ratio = mu0 / lam0 if lam0 > 0 else None
    base = lam0 if lam0 > 0 else mu0
    for f in np.geomspace(0.7, 1.4, 15):
        lam_eff, mu = (base * f, ratio * base * f) if ratio is not None else (0.0, base * f)
        Tc = choose_lagr(T, a, b, lam_eff, mu)
        on, sev = realized(y, lag, Tc, deltas)
        obj = w_g * Tc.mean() + exp_penalty(on, sev, probs, strict)
        if obj < best[2]:
            best = (lam_eff, mu, obj, Tc.mean(), on[len(on) // 2], sev[len(sev) // 2])
    return best


def calibrate_uniform(dist, y, lag, w_g, deltas, probs, strict=True):
    best = None
    for tau in np.concatenate([np.arange(0.85, 0.99, 0.0025), [0.99, 0.993, 0.995, 0.997, 0.999]]):
        Tc = choose_uniform(dist, lag, tau)
        on, sev = realized(y, lag, Tc, deltas)
        obj = w_g * Tc.mean() + exp_penalty(on, sev, probs, strict)
        if best is None or obj < best[1]:
            best = (tau, obj, Tc.mean(), on[len(on) // 2], sev[len(sev) // 2])
    return best
