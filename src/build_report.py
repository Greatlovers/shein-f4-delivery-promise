"""把 docs/results.json 的数字填进 docs/report_template.html，生成 docs/report.html，并用 Edge 无头打印 PDF。"""
import json
import subprocess
from pathlib import Path

from common import ROOT

DOCS = ROOT / "docs"
R = json.loads((DOCS / "results.json").read_text(encoding="utf-8"))
html = (DOCS / "report_template.html").read_text(encoding="utf-8")

PIPELINE = """
<svg viewBox="0 0 760 160" width="100%" xmlns="http://www.w3.org/2000/svg" style="font-family:Microsoft YaHei;font-size:11px">
 <defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#6b7280"/></marker></defs>
 <g stroke="#cbd5e1" fill="#f8fafc">
  <rect x="5" y="10" width="135" height="60" rx="6"/><rect x="5" y="82" width="135" height="58" rx="6"/>
  <rect x="170" y="10" width="160" height="60" rx="6" fill="#eff6ff" stroke="#93c5fd"/>
  <rect x="170" y="82" width="160" height="58" rx="6" fill="#fef2f2" stroke="#fca5a5"/>
  <rect x="360" y="10" width="170" height="130" rx="6" fill="#f0fdfa" stroke="#5eead4"/>
  <rect x="560" y="10" width="195" height="130" rx="6" fill="#eff6ff" stroke="#93c5fd"/>
 </g>
 <g fill="#1f2937">
  <text x="72" y="30" text-anchor="middle" font-weight="700">决策时点信息</text>
  <text x="72" y="47" text-anchor="middle">包裹属性 · lag · 流入时段</text>
  <text x="72" y="62" text-anchor="middle">近况特征（滞后 0–9 天）</text>
  <text x="72" y="102" text-anchor="middle" font-weight="700">历史标签（≤ t0+45 天）</text>
  <text x="72" y="119" text-anchor="middle">三个校准窗口的</text>
  <text x="72" y="134" text-anchor="middle">“当时模型预测 + 真实时效”</text>
  <text x="250" y="30" text-anchor="middle" font-weight="700">① 分位数模型 ×11</text>
  <text x="250" y="47" text-anchor="middle">LightGBM，log 剩余时长</text>
  <text x="250" y="62" text-anchor="middle">输出 P50 … P99.5</text>
  <text x="250" y="102" text-anchor="middle" font-weight="700">② 新线路兜底</text>
  <text x="250" y="119" text-anchor="middle">去渠道/服务商身份的同行模型</text>
  <text x="250" y="134" text-anchor="middle">留渠道验证定保险系数</text>
  <text x="445" y="32" text-anchor="middle" font-weight="700">③ 经验条件分布</text>
  <text x="445" y="52" text-anchor="middle">尺度 = 模型 P90</text>
  <text x="445" y="70" text-anchor="middle">形状 = 真实残差经验分布</text>
  <text x="445" y="88" text-anchor="middle">按 服务商 × 尾部厚度 分桶</text>
  <text x="445" y="112" text-anchor="middle" fill="#0f766e">每包裹 43 个候选承诺</text>
  <text x="445" y="128" text-anchor="middle" fill="#0f766e">及其 P(准时)、P(不晚一天)</text>
  <text x="657" y="32" text-anchor="middle" font-weight="700">④ 拉格朗日决策（每服务商）</text>
  <text x="657" y="52" text-anchor="middle">T = argmin T − λ·a − μ·b</text>
  <text x="657" y="72" text-anchor="middle">(λ, μ) 最小化：</text>
  <text x="657" y="88" text-anchor="middle">占比 × 平均承诺</text>
  <text x="657" y="104" text-anchor="middle">+ 漂移情景下期望罚分</text>
  <text x="657" y="128" text-anchor="middle" fill="#1d4ed8" font-weight="700">→ submission.csv</text>
 </g>
 <g stroke="#6b7280" stroke-width="1.3" fill="none" marker-end="url(#ar)">
  <path d="M140,40 L168,40"/><path d="M72,140 L72,152 L350,152 L350,118 L358,118"/><path d="M330,40 L358,40"/><path d="M330,111 L345,111 L345,60 L358,60"/>
  <path d="M530,75 L558,75"/>
 </g>
</svg>"""

rep = {"$PIPELINE_SVG$": PIPELINE}
for k, v in R["tokens"].items():
    rep[f"${k}$"] = str(v)
for k, v in rep.items():
    html = html.replace(k, v)
left = [t for t in set(__import__("re").findall(r"\$[A-Z_0-9]+\$", html))]
assert not left, f"未填的占位符：{left}"
out = DOCS / "report.html"
out.write_text(html, encoding="utf-8")

chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"   # Edge 无头模式异步写文件，改用 Chrome
tmp_pdf = (DOCS / "_report_tmp.pdf").resolve()
prof = (ROOT / "work" / "_chrome_profile").resolve()
if tmp_pdf.exists():
    tmp_pdf.unlink()
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={prof}",
                "--no-pdf-header-footer", f"--print-to-pdf={tmp_pdf}", out.resolve().as_uri()], timeout=180,
               capture_output=True)
pdf = DOCS / "F4_方案报告.pdf"
tmp_pdf.replace(pdf)
print("PDF:", pdf, pdf.stat().st_size)
