"""第二轮实验后更新 docs/results.json 中的报告数字。

  - 迁移验证（work/e7.jsonl）→ 第六节表格、文字、图 5 数据、作品说明单关键结果
  - 最终提交（outputs/params.json、outputs/test_detail.parquet）→ 第四节参数表、第七节测试集解读
用法：python update_results.py
"""
import json

import numpy as np
import polars as pl

from common import ROOT, WORK

P = ROOT / "docs" / "results.json"
R = json.loads(P.read_text(encoding="utf-8"))
T = R["tokens"]

# ---------------- 迁移验证 ----------------
rows = [json.loads(l) for l in open(WORK / "e7.jsonl", encoding="utf-8")]
E = {}
for r in rows:
    E.setdefault(r["cfg"], {})[r["tgt"]] = r
MAIN = ["0831", "0903", "0906", "0909", "0912"]
EXTRA = ["0915", "0918"]
lab = lambda t: f"{int(t[:2])}/{int(t[2:])}"


def mean5(c):
    return float(np.mean([E[c][t]["A"] for t in MAIN]))


SIG_ROWS = [("latest_s0", "不留余量 σ=0"), ("latest_s0.04", "σ=0.04"), ("latest_s0.05", "σ=0.05"),
            ("latest_s0.06", "<b>σ=0.06（采用）</b>"), ("latest_s0.07", "σ=0.07"), ("latest_s0.08", "σ=0.08")]
VAR_ROWS = [("pool3_s0.06", "合并最近 3 个窗口（σ=0.06）"), ("pool2_s0.06", "合并 t−9、t−18 两个窗口（σ=0.06）"),
            ("fresh_s0.05", "再加入更新鲜的 6 天、3 天窗口（σ=0.05）"), ("gap3_s0.06", "残差形状按信息滞后细分（σ=0.06）"),
            ("ref3_s0.05", "残差形状按 P90 水平细分（σ=0.05）"), ("latest_s0.05_b0.02", "漂移情景加偏置 +0.02（σ=0.05）")]
html = []
for c, name in SIG_ROWS + VAR_ROWS:
    if c not in E or not all(t in E[c] for t in MAIN):
        continue
    cells = "".join(f"<td class='num'>{E[c][t]['A']:.1f}</td>" for t in MAIN)
    m = mean5(c)
    bold = c == "latest_s0.06"
    cells += f"<td class='num'>{'<b>' if bold else ''}{m:.1f}{'</b>' if bold else ''}</td>"
    cells += "".join(f"<td class='num'>{E[c][t]['A']:.1f}</td>" if t in E[c] else "<td class='num'>—</td>" for t in EXTRA)
    sep = " style='border-top:1.5px solid #999'" if c == VAR_ROWS[0][0] else ""
    html.append(f"<tr{sep}><td>{name}</td>{cells}</tr>")
T["TRANSFER_ROWS"] = "\n".join(html)

m = {c: mean5(c) for c, _ in SIG_ROWS if c in E and all(t in E[c] for t in MAIN)}
fin = E["latest_s0.06"]
lo, hi = min(fin[t]["A"] for t in MAIN), max(fin[t]["A"] for t in MAIN)
s0 = m.get("latest_s0")
T["TRANSFER_TEXT"] = (
    "流程与测试完全一致：在目标窗口之前已结束的 9 天窗口上（用当时的模型预测 + 真实时效）选 (λ, μ)，用到下一窗口并按官方公式打分。"
    "第二轮把切分点加密到每 3 天一个，得到 5 对完整窗口 + 2 对不完整窗口（9/15 为 6 天、9/18 为 3 天，表中 * 列）。"
    + (f"不留余量时 5 对平均 {s0:.1f}，冲击窗口（9/12、9/15）罚分很重；" if s0 else "")
    + f"σ 在 0.05–0.06 最好（{m['latest_s0.05']:.1f} / {m['latest_s0.06']:.1f}，7 对平均 78.5 / 78.8，差距在噪声内），"
    "σ=0.06 在冲击窗口上更稳，且测试期临近国庆，故保留 0.06。损失曲线不对称：留少了罚得重，留多了只是慢一些。"
    "下半部分的 6 种校准方式都不如\"只用最近一个窗口\"，差距主要出在冲击窗口 9/12——更旧或更细的数据会让决策层对最近的冲击反应变慢。"
)
R["transfer"] = {lab(t): {"0" if c == "latest_s0" else c.split("_s")[1]: E[c][t]["A"] for c, _ in SIG_ROWS
                          if c in E and t in E[c]} for t in MAIN + EXTRA}
R["final"]["transfer_mean5"] = m["latest_s0.06"]

T["FINAL_A"] = f"~{m['latest_s0.06']:.0f}（与测试流程一致的 5 对迁移验证平均）"
kr = T["KEY_RESULTS_ROWS"].split("\n")
kr[0] = (f"<tr><td>严格版 A：可实现的迁移评估（5 对平均）</td><td>249.2（渠道 P90 常数，三折平均）</td>"
         f"<td><b>{m['latest_s0.06']:.1f}</b>（{lo:.1f}–{hi:.1f}）</td><td>报告第 8 页；work/e7.jsonl、b0 基线日志</td></tr>")
if s0:
    kr[2] = (f"<tr><td>漂移余量的价值（5 对平均）</td><td>{s0:.1f}（σ=0）</td><td>{m['latest_s0.06']:.1f}（σ=0.06）</td>"
             f"<td>报告第 8 页表；work/e7.jsonl</td></tr>")
T["KEY_RESULTS_ROWS"] = "\n".join(kr)
T["HIGHLIGHT_1"] = T["HIGHLIGHT_1"].replace("降到 ~78", f"降到 ~{m['latest_s0.06']:.0f}")
if s0:
    T["HIGHLIGHT_2"] = ("同一验证折上逐步对照（第 7 页）：模型化使事后得分 111→67；新决策层再降至 66.0（统一分位 70.8）；"
                        f"漂移余量把可实现得分（5 对迁移验证平均）从 {s0:.1f} 降到 {m['latest_s0.06']:.1f}。"
                        "所有数字可由 work/*.log、work/e7.jsonl 复现，本地评分函数与官方 score.py 逐位一致。")

it = T["ITER_ROWS"].split("\n")
it = [r for r in it if not any(r.startswith(f"<tr><td>{i}</td>") for i in range(13, 20))]   # 可重复运行
it += [
    "<tr><td>13</td><td>可实现：三窗等权合并，σ=0.05</td><td class='num'>—</td><td class='num'>—</td><td class='num'>95.65</td>"
    "<td>老窗口稀释最近信息，不如单窗（80.07）</td></tr>",
    "<tr><td>14</td><td>第二轮：迁移验证扩到 5+2 对，重选 σ 与校准方式</td><td class='num'>—</td><td class='num'>—</td><td class='num'>—</td>"
    f"<td>σ=0.05/0.06 并列最好（{m['latest_s0.05']:.1f}/{m['latest_s0.06']:.1f}）；合并、细分、更新鲜窗口、加偏置均更差（见第六节）</td></tr>",
    "<tr><td>15</td><td>第二轮：预测层 14 项改进尝试（事后）</td><td class='num'>—</td><td class='num'>—</td><td class='num'>65.59</td>"
    "<td>仓库负荷、目的地近况等新特征与加权、调参共 10 项无增益；换 CatBoost/XGBoost 单模更差，融合仅 −0.3，弃用；"
    "3 个随机种子平均（P90 缩放 67.02→66.67，完整决策层 66.04→65.59），采用</td></tr>",
    "<tr><td>16</td><td><b>最终：最近窗口校准，σ=0.06，P90/P97 三种子平均</b></td><td class='num'>—</td><td class='num'><b>77.35</b></td>"
    f"<td class='num'><b>78.10</b></td><td>5 对迁移平均 {m['latest_s0.06']:.1f}（{lo:.1f}–{hi:.1f}）；迁移验证用单种子模型</td></tr>",
]
T["ITER_ROWS"] = "\n".join(it)

# ---------------- 最终提交 ----------------
par = json.loads((ROOT / "outputs" / "params.json").read_text(encoding="utf-8"))
det = pl.read_parquet(ROOT / "outputs" / "test_detail.parquet")
share = det.group_by("lgs_company_nm").agg(n=pl.len(), meanT=pl.col("pred_hours").mean()).with_columns(
    s=pl.col("n") / pl.col("n").sum()).sort("n", descending=True)
pr = []
for k, n, mt, s in share.iter_rows():
    c = par["carriers"][k]
    on, sev = c["calib_on_sev"][0]
    pr.append(f"<tr><td>{k}</td><td class='num'>{s * 100:.2f}%</td><td class='num'>{c['lam']:.0f}</td><td class='num'>{c['mu']:.0f}</td>"
              f"<td class='num'>{mt:.1f}</td><td class='num'>{on:.1f}% / {sev:.1f}%</td></tr>")
pr.append("<tr><td colspan='6' class='small'>λ、μ 单位为“小时 / 每 1 单位概率”；μ=0 表示该服务商的准时率约束更紧、严重超时自然满足。"
          "LSP_0001 的平均承诺含新线路（兜底 P99.5×1.15）。</td></tr>")
T["PARAM_ROWS"] = "\n".join(pr)

T_ = det["pred_hours"].to_numpy()
nr = det.filter(~pl.col("rare")).with_columns(add=pl.col("pred_hours") - pl.col("lag_h") - pl.col("q900"),
                                              ratio=(pl.col("pred_hours") - pl.col("lag_h")) / pl.col("q900"))
qb = nr.with_columns(b=pl.col("q900").qcut(5, labels=[str(i) for i in range(5)])).group_by("b").agg(
    add=pl.col("add").median(), q=pl.col("q900").median()).sort("b")
r5, r95 = nr["ratio"].quantile(0.05), nr["ratio"].quantile(0.95)
pm = {k: v for k, _, v, _ in share.iter_rows()}
on6, sev6 = par["carriers"]["LSP_0006"]["calib_on_sev"][0]
T["TEST_TEXT"] = (
    f"测试集 911,415 件的平均承诺 {T_.mean():.1f}h、中位 {np.median(T_):.1f}h、最小 {T_.min():.1f}h，没有不合理的极小承诺。"
    f"决策层的效果在左图清晰可见：同一服务商内，承诺相对模型 P90 的加成从短时效件的中位 +{qb['add'][0]:.0f}h（P90≈{qb['q'][0]:.0f}h）"
    f"到长时效件的 +{qb['add'][-1]:.0f}h（P90≈{qb['q'][-1]:.0f}h）逐级递增，比值集中在 {r5:.2f}–{r95:.2f}——短件靠“24h 宽限”自然满足严重超时约束，"
    f"长件按比例多收保险费。各服务商平均承诺差异很大：LSP_0006 {pm['LSP_0006']:.0f}h，LSP_0004 因长尾极重达 {pm['LSP_0004']:.0f}h；"
    f"最大的 LSP_0006 在校准窗口上卡在 {on6:.1f}% / {sev6:.1f}%，小服务商（LSP_0005/0008）则留出明显余量——正是“杠杆不对称”的定量结果。"
    f"若测试期平均实际时效与 9 月相近（~170h），预期 ME 约 {T_.mean() - 170:.0f}h。")
T["MIN_T"] = f"{T_.min():.1f}"
kr = T["KEY_RESULTS_ROWS"].split("\n")
assert kr[5].startswith("<tr><td>测试集平均承诺")
kr[5] = (f"<tr><td>测试集平均承诺 / 最小承诺</td><td>—</td><td>{T_.mean():.1f}h / {T_.min():.1f}h</td>"
         "<td>outputs/submission.csv（911,415 行，格式已校验）</td></tr>")
T["KEY_RESULTS_ROWS"] = "\n".join(kr)
R["final"]["mean_T"] = float(T_.mean())

P.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print("updated", P, "mean5(σ=0.06)=", round(m["latest_s0.06"], 2), "σ0=", s0)
