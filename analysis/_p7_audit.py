# -*- coding: utf-8 -*-
"""P7: submission-readiness audit of CRANE-Z v29.8."""
import re

t = open("../CRANE-Z_manuscript.tex", encoding="utf-8").read()
print("=== P7 可投稿性审计 ===")
print("\n[1] 写作规范层 (H40-H54)")
sections = ["abstract", "introduction", "results", "discussion", "methods",
            "data availability", "code availability", "acknowledg", "competing"]
pat = re.compile(r"\\(?:section|subsection)\*?\{", re.I)
all_secs = [s.strip().lower() for s in pat.findall(t)]
sec_text = " ".join(all_secs)
for s in sections:
    found = s in t.lower()
    print(f"  [{s}]: {'OK' if found else 'MISSING'}")

cites = set()
for m in re.findall(r"\\cite\{([^}]*)\}", t):
    for k in m.split(","):
        cites.add(k.strip())
bib = open("../crane_refs.bib", encoding="utf-8").read()
bkeys = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
print(f"\n[2] 引用: {len(cites)} cited / {len(bkeys)} bib | 悬空: {len(cites-bkeys)} 未引: {len(bkeys-cites)}")
figs = re.findall(r"\\includegraphics", t)
tabs = re.findall(r"\\begin\{table", t)
print(f"  图: {len(figs)} 张 | 表: {len(tabs)} 个")

print("\n[3] 关键证据链数字 (cross-check with paper text)")
checks = {
    "内部 OPS 7.426": "7.426" in t,
    "Ridge 7.400": "7.400" in t,
    "AA -14.69 年": "14.69" in t,
    "Cohen d 1.18": "1.18" in t,
    "Spearman 0.816": "0.816" in t,
    "cohort2 delta -17.89": "-17.89" in t,
    "cohort2 d 1.39": "1.39" in t,
    "跨平台 18.2%": "18.2" in t,
    "Peters d 0.55": "0.55" in t,
    "ssGSEA d 1.06": "1.06" in t,
    "ssGSEA d 1.14": "1.14" in t,
}
allok = True
for k, v in checks.items():
    allok &= v
    print(f"  [{'OK' if v else 'MISSING'}] {k}")

print(f"\n[4] 结论: 证据链 {'完整' if allok else '有缺口'}")
