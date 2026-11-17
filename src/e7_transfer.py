"""实验 7：多对跨窗口迁移验证 —— 在 5 对"上一窗口校准 → 下一窗口按官方公式打分"上比较决策层配置。

切分点每 3 天一个（8/22 … 9/12），目标窗口 t ∈ {8/31, 9/3, 9/6, 9/9, 9/12}，与测试流程一致：
只用 t 之前已结束的窗口校准（最近的是切分点 t-9 的窗口 [t-9, t)）。
用法：python e7_transfer.py <配置名> [...]       结果追加到 work/e7.jsonl
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from common import WORK, score
from policy import drift_scenarios
from policy_emp import DISP, REF, EmpShape, candidates_emp, choose, fit_carrier

TARGETS = ["0831", "0903", "0906", "0909", "0912"]
CARRIERS = [f"LSP_000{i}" for i in range(1, 9)]


def shift(c: str, days: int) -> str:
    return (datetime.strptime("2026" + c, "%Y%m%d") + timedelta(days=days)).strftime("%m%d")


SUF = ""   # 预测文件后缀（如 "_cal"），由命令行 --suf 设置


def avail(c: str) -> bool:
    return (WORK / f"pred_{c}{SUF}.parquet").exists()


_PAY = None


def arrays(c: str):
    global _PAY
    if _PAY is None:
        _PAY = pl.read_parquet(WORK / "all.parquet", columns=["pkg_no", "pay_time"])
    d = pl.read_parquet(WORK / f"pred_{c}{SUF}.parquet", columns=["pkg_no", "lgs_company_nm", "lag_h", "y_total_hours", REF, DISP[0]])
    d = d.join(_PAY, on="pkg_no", how="left")
    t0 = datetime.strptime("2026" + c, "%Y%m%d")
    gap = ((d["pay_time"] - t0).dt.total_seconds() / 86400.0).to_numpy()
    return dict(g=d["lgs_company_nm"].to_numpy(), lag=d["lag_h"].to_numpy(), y=d["y_total_hours"].to_numpy(),
                ref=d[REF].to_numpy(), disp=np.log(d[DISP[0]].to_numpy() / d[REF].to_numpy()), gap=gap)


def gkey(a, nb, refcuts=None):
    """形状分组键：服务商；nb>1 时再按信息滞后天数等宽分 nb 段；
    refcuts（{服务商: 切点}）不为空时改按模型 P90 水平分段（切点来自校准窗口）。"""
    if refcuts:
        gb = np.zeros(len(a["g"]), int)
        for k, cuts in refcuts.items():
            m = a["g"] == k
            gb[m] = np.searchsorted(cuts, np.log(a["ref"][m]))
        return np.char.add(np.char.add(a["g"].astype(str), "|"), gb.astype(str))
    if nb <= 1:
        return a["g"]
    gb = np.clip((a["gap"] / (9.0 / nb)).astype(int), 0, nb - 1)
    return np.char.add(np.char.add(a["g"].astype(str), "|"), gb.astype(str))


def cands(shape, a, m, nb, refcuts=None):
    """服务商子集 m 的候选；按形状分组键分别生成后按原顺序拼回。"""
    keys = gkey(a, nb, refcuts)[m]
    idx_all = np.where(m)[0]
    out = None
    for kk in np.unique(keys):
        sub = idx_all[keys == kk]
        T, A_, B_ = candidates_emp(shape, kk, a["lag"][sub], a["ref"][sub], a["disp"][sub])
        if out is None:
            out = [np.empty((len(idx_all), T.shape[1])) for _ in range(3)]
        pos = np.where(keys == kk)[0]
        out[0][pos], out[1][pos], out[2][pos] = T, A_, B_
    return tuple(out)


# ---- 配置：cal(t) → [(切分点, 权重)]；sigma（全局或按服务商字典）；bias ----
def cal_latest(t):
    return [(shift(t, -9), 1.0)]


def cal_pool(offsets, weights):
    """返回可 pickle 的描述（多进程需要），在 resolve_cal 里展开。"""
    return ("pool", tuple(offsets), tuple(weights))


def resolve_cal(cal, t):
    if callable(cal):
        return cal(t)
    _, offsets, weights = cal
    return [(shift(t, -o), w) for o, w in zip(offsets, weights) if avail(shift(t, -o))]


CONFIGS = {}
for s in (0.0, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10):
    CONFIGS[f"latest_s{s:g}"] = dict(cal=cal_latest, sigma=s)
for s in (0.04, 0.06, 0.08):
    CONFIGS[f"pool3_s{s:g}"] = dict(cal=cal_pool([9, 12, 15], [1, 1, 1]), sigma=s)
    CONFIGS[f"pool2_s{s:g}"] = dict(cal=cal_pool([9, 18], [1, 1]), sigma=s)
for s in (0.04, 0.06, 0.08):
    CONFIGS[f"gap3_s{s:g}"] = dict(cal=cal_latest, sigma=s, gapb=3)
for s in (0.05, 0.06, 0.07):
    CONFIGS[f"fresh_s{s:g}"] = dict(cal=cal_pool([9, 6, 3], [1, 1, 1]), sigma=s)
    CONFIGS[f"ref3_s{s:g}"] = dict(cal=cal_latest, sigma=s, refb=3)
for b in (0.02, 0.04):
    for s in (0.03, 0.05):
        CONFIGS[f"latest_s{s:g}_b{b:g}"] = dict(cal=cal_latest, sigma=s, bias=b)


def run_one(args):
    name, t, cfg, suf = args
    global SUF
    SUF = suf
    cals = resolve_cal(cfg["cal"], t)
    if not cals or not all(avail(c) for c, _ in cals):
        return None
    A = []
    for c, _ in cals:
        a = arrays(c)
        # 只保留目标窗口开始前支付的包裹（较新的切分点窗口会伸进目标窗口，必须截断）
        keep = a["gap"] < (datetime.strptime("2026" + t, "%Y%m%d") - datetime.strptime("2026" + c, "%Y%m%d")).days
        A.append({k: v[keep] for k, v in a.items()})
    w = [ww for _, ww in cals]
    tgt = arrays(t)
    nb = cfg.get("gapb", 1)
    pool = {k: np.concatenate([a[k] for a in A]) for k in ("ref", "disp", "y", "lag")}
    refcuts = None
    if cfg.get("refb", 1) > 1:
        gg = np.concatenate([a["g"] for a in A])
        qs = np.linspace(0, 1, cfg["refb"] + 1)[1:-1]
        refcuts = {k: np.quantile(np.log(pool["ref"][gg == k]), qs) for k in np.unique(gg) if (gg == k).sum() > 30000}
    pool["g"] = np.concatenate([gkey(a, nb, refcuts) for a in A])
    shape = EmpShape(pool["g"], pool["ref"], pool["disp"], pool["y"] - pool["lag"])
    T = np.empty(len(tgt["lag"]))
    params = {}
    for k in np.unique(tgt["g"]):
        mt = tgt["g"] == k
        sig = cfg["sigma"][k] if isinstance(cfg["sigma"], dict) else cfg["sigma"]
        deltas, probs = drift_scenarios(sig, cfg.get("bias", 0.0)) if sig > 0 else (np.array([cfg.get("bias", 0.0)]), np.array([1.0]))
        cc, ys, ls = [], [], []
        for a in A:
            m = a["g"] == k
            cc.append(cands(shape, a, m, nb, refcuts))
            ys.append(a["y"][m])
            ls.append(a["lag"][m])
        ct = cands(shape, tgt, mt, nb, refcuts)
        p = fit_carrier(cc, ys, ls, ct, float(mt.mean()), deltas, probs, win_w=w)
        T[mt] = choose(*ct, p["lam"], p["mu"])
        params[k] = (p["lam"], p["mu"])
    res = score(tgt["g"], tgt["y"], T)
    per = {k: dict(on=v[0] * 100, sev=v[1] * 100, n=v[2], meanT=float(T[tgt["g"] == k].mean()),
                   meanY=float(tgt["y"][tgt["g"] == k].mean())) for k, v in res["per"].items()}
    return dict(cfg=name + suf, tgt=t, A=res["A"], ME=res["ME"], pen_on=res["pen_on"], pen_sev=res["pen_sev"],
                meanT=res["meanT"], meanY=res["meanY"], per=per, params=params)


def main(names, workers=4, suf="", targets=None):
    targets = targets or TARGETS
    done = set()
    if (WORK / "e7.jsonl").exists():
        for line in open(WORK / "e7.jsonl", encoding="utf-8"):
            r = json.loads(line)
            done.add((r["cfg"], r["tgt"]))
    jobs = [(n, t, CONFIGS[n], suf) for n in names for t in targets if (n + suf, t) not in done]
    print(f"{len(jobs)} 个任务（跳过已完成 {len(done)}）", flush=True)
    with ProcessPoolExecutor(workers) as ex, open(WORK / "e7.jsonl", "a", encoding="utf-8") as f:
        for r in ex.map(run_one, jobs):
            if r is None:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{r['cfg']:22s} {r['tgt']}  A={r['A']:7.2f}  ME={r['ME']:6.2f}  on={r['pen_on']:5.2f}  sev={r['pen_sev']:5.2f}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    suf = ""
    if "--suf" in args:
        i = args.index("--suf")
        suf = args[i + 1]
        args = args[:i] + args[i + 2:]
    tg = None
    if "--targets" in args:
        i = args.index("--targets")
        tg = args[i + 1].split(",")
        args = args[:i] + args[i + 2:]
    main(args, suf=suf, targets=tg)
