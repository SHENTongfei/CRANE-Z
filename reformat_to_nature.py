# -*- coding: utf-8 -*-
"""Reformat CRANE-Z manuscript to Nature Aging submission style.

Input:  CRANE-Z_manuscript.tex (OUP / Bioinformatics template)
Output: CRANE-Z_nature.tex (Nature Aging style, article + natbib)
"""
import os, re

SRC = r"C:/Users/TS/Desktop/crane/CRANE-Z_manuscript.tex"
DST = r"C:/Users/TS/Desktop/crane/CRANE-Z_nature.tex"

t = open(SRC, encoding="utf-8").read()

# ---- 1. preamble ----
# IMPORTANT: use lambda for replacement to avoid re.sub's \b backspace conversion
PREAMBLE = (
    r"\documentclass[11pt,a4paper]{article}" + "\n"
    r"\usepackage[utf8]{inputenc}" + "\n"
    r"\usepackage[T1]{fontenc}" + "\n"
    r"\usepackage{graphicx}" + "\n"
    r"\usepackage{amsmath,amssymb}" + "\n"
    r"\usepackage[sort&compress,square,comma]{natbib}" + "\n"
    r"\usepackage{url}" + "\n"
    r"\usepackage[margin=2cm]{geometry}" + "\n"
    r"\usepackage[colorlinks=true,linkcolor=black,citecolor=black,urlcolor=black]{hyperref}" + "\n"
    r"\bibpunct{[}{]}{,}{n}{}{,}" + "\n"
)
t = re.sub(
    r"\\documentclass\[unnumsec,webpdf,contemporary,large\]\{oup-authoring-template\}",
    lambda m: PREAMBLE,
    t, count=1
)
t = re.sub(r"\\graphicspath\{\{Fig/\}\}\n?", "", t)

# ---- 2. balance-aware scrub of OUP-specific fields (handles nested braces) ----
def strip_arg(text, cmd):
    out = []; i = 0
    while i < len(text):
        j = text.find(cmd, i)
        if j < 0:
            out.append(text[i:]); break
        out.append(text[i:j])
        k = j + len(cmd)
        while k < len(text) and text[k] != "{": k += 1
        if k >= len(text):
            out.append(text[j:]); break
        depth = 1; k += 1
        while k < len(text) and depth > 0:
            if text[k] == "{": depth += 1
            elif text[k] == "}": depth -= 1
            k += 1
        i = k
    return "".join(out)

def strip_command(text, cmd):
    """Remove a whole \cmd{...} including its argument, brace-balanced."""
    out = []; i = 0
    while i < len(text):
        j = text.find("\\" + cmd, i)
        if j < 0:
            out.append(text[i:]); break
        # skip potential \cmd<sep> form -- here our targets are simple \cmd
        if j > 0 and text[j-1].isalpha():
            out.append(text[i:j+len(cmd)+1]); i = j+len(cmd)+1; continue
        out.append(text[i:j])
        k = j + len(cmd) + 1
        if k < len(text) and text[k] == "[":
            depth = 1; k += 1
            while k < len(text) and depth > 0:
                if text[k] == "[": depth += 1
                elif text[k] == "]": depth -= 1
                k += 1
        if k < len(text) and text[k] == "{":
            depth = 1; k += 1
            while k < len(text) and depth > 0:
                if text[k] == "{": depth += 1
                elif text[k] == "}": depth -= 1
                k += 1
        i = k
    return "".join(out)

for cmd in ["orgdiv", "orgname", "orgaddress", "street", "postcode", "state", "country"]:
    t = strip_arg(t, "\\" + cmd)
for cmd in ["journaltitle", "DOI", "copyrightyear", "pubyear", "access",
            "appnotes", "firstpage", "received", "revised", "accepted",
            "corresp", "authormark", "address"]:
    t = strip_command(t, cmd)

# remove leftover empty braces / commas from stripped blocks
t = re.sub(r",\s*,", ",", t)
t = re.sub(r"\{\s*\}", "", t)

# remove all remaining OUP author lines
t = re.sub(r"\\author\[[^\]]*\]\{[^}]*\}", "", t)
t = re.sub(r"\\author\{[^}]*\}", "", t)

# ---- 3. build Nature-style author + affiliation block ----
# Use \thanks{} (article standard) instead of \affil (revtex-only)
author_block = """\\author{Tongfei Shen}
\\thanks{School of Information Science and Engineering, Qingdao Huanghai University, Qingdao, China.}
\\author{Xueqin Lei}
\\thanks{College of Physics and Electronic Information Engineering, Zhejiang Normal University, Jinhua, China.}
\\author{Ka Hung Lok}
\\thanks{The Centre for Innovation and Entrepreneurship, The Hang Seng University of Hong Kong, Hong Kong, China.}
\\author{Huaicheng Li}
\\thanks{Ability R\\&D Energy Research Centre, School of Energy and Environment, City University of Hong Kong, Hong Kong, China.}
\\author{Zhongze Pan}
\\thanks{School of Engineering, Nanfang College Guangzhou, Guangzhou, China.}
\\author{Xuepeng Fu}
\\thanks{Department of Life Science and Agroforestry, Qiqihar University, Qiqihar, China.}
\\author{Corresponding author: Tongfei Shen. \\href{mailto:02383@qqhru.edu.cn}{02383@qqhru.edu.cn}}
\\thanks{Correspondence and requests for materials should be addressed to T.S. (email: 02383@qqhru.edu.cn).}
"""
t = t.replace("\\maketitle", author_block + "\\maketitle", 1)

# ---- 4. abstract -> Nature ~150-word single paragraph ----
abstract_match = re.search(r"\\abstract\{(.+?)\n\\keywords", t, re.DOTALL)
if abstract_match:
    abs_body = abstract_match.group(1)
    # strip \textbf{...:} structured heads (colon is INSIDE braces)
    plain = re.sub(r"\\textbf\{[^}]*:\}\s*", "", abs_body)
    plain = re.sub(r"\\\\\n", " ", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    words = plain.split()
    if len(words) > 170:
        plain = " ".join(words[:160]).rstrip(",.") + "."
    new_abs = "\\begin{abstract}\n" + plain + "\n\n\\textbf{Keywords:} " + t.split("\\keywords{")[1].split("}")[0] + ".\n\\end{abstract}\n"
    t = t.replace(abstract_match.group(0), new_abs, 1)

# ---- 5. figures/tables: figure* -> figure ----
t = re.sub(r"\\begin\{figure\*\}", r"\\begin{figure}", t)
t = re.sub(r"\\end\{figure\*\}", r"\\end{figure}", t)
t = re.sub(r"\\begin\{table\*\}", r"\\begin{table}", t)
t = re.sub(r"\\end\{table\*\}", r"\\end{table}", t)
t = re.sub(r"\\includegraphics\[width=0\\.95\\textwidth\]",
           r"\\includegraphics[width=\\textwidth]", t)

# ---- 5b. structural cleanup (mandatory post-fixes) ----
# bibpunct must have 6 args; dedup if present twice
t = t.replace(r"\bibpunct{[}{]}{,}{n}{,}", r"\bibpunct{[}{]}{,}{n}{}{,}")
t = t.replace(r"\bibpunct{[}{]}{,}{n}{}{,}\n\n\bibpunct{[}{]}{,}{n}{}{,}",
              r"\bibpunct{[}{]}{,}{n}{}{,}", 1)
# remove \received/\revised/\accepted multi-arg remnants like {0}{2026}
t = re.sub(r"\n\{0\}\{2026\}", "", t)
# remove leftover \keywords{...} line duplicated after abstract
t = re.sub(r"\n\{longevity, [^}]*\}", "", t)
t = re.sub(r",\s*,\s*,", ",", t)
# graphicspath needed for figures (was stripped with OUP scaffolding)
t = t.replace(r"\usepackage{graphicx}",
              "\\usepackage{graphicx}\n\\graphicspath{{Fig/}}", 1)

# ---- 6. bibliography style: unsrtnat -> plainnat (Vancouver-numeric) ----
t = re.sub(r"\\bibliographystyle\{unsrtnat\}", r"\\bibliographystyle{plainnat}", t)

# write
with open(DST, "w", encoding="utf-8") as f:
    f.write(t)
print(f"Nature reformat -> {DST} ({len(t)} bytes, {len(t.splitlines())} lines)")