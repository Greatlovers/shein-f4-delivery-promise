"""汇总 work/e7.jsonl：每个配置在 5 个完整目标窗口上的 A 与平均（另列 9/15、9/18 两个不完整窗口）；
可选按服务商列出准时率/严重超时。用法：python e7_summary.py [配置名 ...]"""
import json
import sys
from collections import defaultdict

from common import WORK

sys.stdout.reconfigure(encoding="utf-8")
rows = [json.loads(l) for l in open(WORK / "e7.jsonl", encoding="utf-8")]
by = defaultdict(dict)
for r in rows:
    by[r["cfg"]][r["tgt"]] = r
MAIN = ["0831", "0903", "0906", "0909", "0912"]
EXTRA = ["0915", "0918"]
print(f"{'配置':22s}" + "".join(f"{t:>8s}" for t in MAIN) + f"{'5对平均':>9s}{'ME':>7s}{'罚分':>7s}" + "".join(f"{t:>8s}" for t in EXTRA))


def avg(d, key):
    v = [d[t][key] if key != "pen" else d[t]["pen_on"] + d[t]["pen_sev"] for t in MAIN if t in d]
    return sum(v) / len(v) if v else float("nan")


for c, d in sorted(by.items(), key=lambda x: (sum(t in x[1] for t in MAIN) < 5, avg(x[1], "A"))):
    n = sum(t in d for t in MAIN)
    line = f"{c:22s}" + "".join(f"{d[t]['A']:8.2f}" if t in d else f"{'':>8s}" for t in MAIN)
    line += f"{avg(d, 'A'):9.2f}{avg(d, 'ME'):7.2f}{avg(d, 'pen'):7.2f}"
    line += "".join(f"{d[t]['A']:8.2f}" if t in d else f"{'':>8s}" for t in EXTRA)
    print(line + ("" if n == 5 else f"  (仅 {n} 对)"))
for c in sys.argv[1:]:
    print(f"\n{c}：各服务商 准时%/严重超时%（目标窗口）")
    for t in MAIN + EXTRA:
        if t in by[c]:
            p = by[c][t]["per"]
            print(f"  {t} " + " | ".join(f"{k[-2:]}:{v['on']:.1f}/{v['sev']:.1f}" for k, v in sorted(p.items())))
