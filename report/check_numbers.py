"""Confirm that every number in main.tex's prose, and in the generated tables, comes from a results file.

Independent of make_tables.py's reader and formatter:
  1. each traced value is re-read with the csv / json modules and compared at the printed precision;
  2. each macro in tables/numbers.tex must print exactly its traced value;
  3. each number in the generated tables must be one of that table's traced values, or a structural label;
  4. main.tex is stripped to prose and every literal number must match a written justification below. A literal is
     never accepted because the same digits happen to occur somewhere in a results file.

Writes report/number_check.txt and exits 1 on any failure.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# a number that is not part of a name such as A5, v2, 2014J, DAS3H, NDCG@5
NUMBER = re.compile(r"(?<![A-Za-z0-9_@.])\d+(?:\{?,\}?\d{3})*(?:\.\d+)?(?![A-Za-z0-9_])")

from number_allowlist import JUSTIFIED  # noqa: E402


def source_value(p: dict):
    path = ROOT / p["file"]
    if path.suffix == ".json":
        d = json.loads(path.read_text(encoding="utf-8"))
        for part in p["col"].split("/"):
            d = d[int(part)] if isinstance(d, list) else d[part]
        return d

    def same(a, b):
        if a == b:
            return True
        try:
            return float(a) == float(b)
        except ValueError:
            return False

    with path.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if all(same(r[k], v) for k, v in p["where"])]
    if len(rows) != 1:
        raise LookupError(f"{len(rows)} rows selected")
    return rows[0][p["col"]]


def agrees(text: str, raw, style: str) -> bool:
    v = float(raw)
    if style == "p" and text.startswith("<"):
        return v < float(text[1:])
    t = text.replace(",", "")
    if style.startswith("pct"):
        v, t = 100 * v, t.rstrip("%")
    if style.startswith("abs"):
        v = abs(v)
    decimals = len(t.split(".")[1]) if "." in t else 0
    return abs(v - float(t)) <= 0.5 * 10 ** -decimals + 1e-9


def plain(s: str) -> str:
    return s.replace("{,}", ",").replace("\\%", "%").replace("$", "")


def prose(tex: str) -> str:
    body = re.sub(r"(?<!\\)%.*", "", tex)
    body = body.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
    body = re.sub(r"\\(?:input|cite|ref|eqref|label|bibliography|bibliographystyle)\{[^}]*\}", " ", body)
    body = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", " ", body)
    body = re.sub(r"\\(?:begin|end)\{[^}]*\}(?:\[[^\]]*\])?", " ", body)
    body = re.sub(r"\\(?:vspace|hspace|setlength)\*?\{[^}]*\}(?:\{[^}]*\})?", " ", body)
    return re.sub(r"\\\\(?:\[[^\]]*\])?", " ", body)


def main():
    prov = json.loads((HERE / "tables/provenance.json").read_text(encoding="utf-8"))
    bad, lines = [], []

    n_source = 0
    by_table, by_macro = defaultdict(set), {}
    for p in prov:
        where = f"{p['file']} {dict(p['where'])} {p['col']}"
        try:
            raw = source_value(p)
            ok = agrees(p["text"], raw, p["style"])
        except Exception as e:
            raw, ok = repr(e), False
        n_source += ok
        if not ok:
            bad.append(f"SOURCE MISMATCH {p['used_in']}: {p['text']} vs {raw} ({where})")
        kind, name, *_ = p["used_in"].split(":")
        if kind == "macro":
            by_macro[name] = (p["text"], where)
        else:
            by_table[name].update(x.replace(",", "") for x in NUMBER.findall(p["text"]))

    defined = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{\\mbox\{(.*)\}\}",
                              (HERE / "tables/numbers.tex").read_text(encoding="utf-8")))
    for name, body in defined.items():
        if name not in by_macro or plain(body) != by_macro[name][0]:
            bad.append(f"MACRO MISMATCH \\{name}: prints {body}, traced {by_macro.get(name)}")

    n_table = 0
    structural = {"1", "0.002", "95", "0"}         # 'topic 1', the preregistered margin, CI level, 'gap = 0'
    for key, allowed in by_table.items():
        t = re.sub(r"(?m)^%.*$", "", (HERE / f"tables/{key}.tex").read_text(encoding="utf-8"))
        t = re.sub(r"\\(?:vspace|parbox)\{[^}]*\}", " ", t)
        t = re.sub(r"\\(?:label|setlength|begin|end)\{[^}]*\}(?:\{[^\n]*\})?(?:\[[^\]]*\])?", " ", t)
        for m in NUMBER.finditer(t):
            x = m.group(0).replace(",", "")
            if x in allowed or x in structural or (key == "leak" and x == "0.5"):
                n_table += 1
            else:
                bad.append(f"UNTRACED TABLE NUMBER {key}: {x} in '{t[max(0, m.start() - 35):m.end() + 35]!r}'")

    body = prose((HERE / "main.tex").read_text(encoding="utf-8"))
    used = Counter(re.findall(r"\\([A-Za-z]+)", body))
    used = {k: v for k, v in used.items() if k in defined}
    for k in sorted(used):
        lines.append(f"RESULT    \\{k} = {plain(defined[k])}  x{used[k]}  <- {by_macro[k][1]}")
    body = " ".join(body.split())
    allowed_at = {}
    for before, number, after, why in JUSTIFIED:
        at = body.find(before + number + after)
        while at >= 0:
            allowed_at[at + len(before)] = (number, why)
            at = body.find(before + number + after, at + 1)
    n_just = 0
    for m in NUMBER.finditer(body):
        ctx = body[max(0, m.start() - 70):m.end() + 70]
        ok = allowed_at.get(m.start())
        if ok and ok[0] == m.group(0):
            n_just += 1
            lines.append(f"JUSTIFIED {m.group(0)}: {ok[1]}  |  ...{ctx}...")
        else:
            bad.append(f"UNMATCHED PROSE NUMBER {m.group(0)}  |  ...{ctx}...")

    head = [
        "Number check for report/main.tex (python report/check_numbers.py)",
        f"  traced values re-read from their source file and matched : {n_source} of {len(prov)}",
        f"  numbers in generated tables traced                      : {n_table}",
        f"  result numbers in the prose (macro uses)                : {sum(used.values())} ({len(used)} distinct)",
        f"  other prose numbers, each justified as not a result     : {n_just}",
        f"  failures                                                : {len(bad)}",
        "  values from report/audit/ are read-only audit counts. They are kept outside results/, which is frozen.",
        "",
    ]
    (HERE / "number_check.txt").write_text("\n".join(head + bad + lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(head + bad[:40]))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
