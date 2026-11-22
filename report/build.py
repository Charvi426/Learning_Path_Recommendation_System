"""Build the report: tables, figures, number check, PDF (if a TeX engine exists) and overleaf.zip.

  python report/build.py

Engines are tried in this order: latexmk, pdflatex + bibtex, tectonic (on PATH, or the TECTONIC variable).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_OVERFULL_PT = 5.0


def overleaf_zip():
    files = [HERE / "main.tex", HERE / "references.bib", *sorted((HERE / "tables").glob("*.tex")),
             *sorted((HERE / "figures").glob("*.pdf")), *sorted((HERE / "figures").glob("demo_*.png"))]
    with zipfile.ZipFile(HERE / "overleaf.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            info = zipfile.ZipInfo(f.relative_to(HERE).as_posix(), (1980, 1, 1, 0, 0, 0))      # reproducible
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, f.read_bytes())
    print(f"overleaf.zip: {len(files)} files")


def compile_pdf() -> str | None:
    def run(*cmd):
        return subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")

    if shutil.which("latexmk") and run("latexmk", "-pdf", "-interaction=nonstopmode", "main.tex").returncode == 0:
        return "latexmk"
    if shutil.which("pdflatex") and shutil.which("bibtex"):
        steps = [("pdflatex", "-interaction=nonstopmode", "main.tex"), ("bibtex", "main")]
        steps += [steps[0], steps[0]]
        if all(run(*s).returncode == 0 for s in steps):
            return "pdflatex + bibtex"
    exe = os.environ.get("TECTONIC") or shutil.which("tectonic")
    if exe:
        r = run(exe, "--keep-logs", "main.tex")
        if r.returncode == 0:
            return "tectonic"
        print(r.stderr[-3000:])
    return None


def check_log() -> list[str]:
    log = (HERE / "main.log").read_text(encoding="utf-8", errors="replace")
    bad = re.findall(r"LaTeX Warning: (?:Reference|Citation) .*? undefined", log)
    bad += re.findall(r"There were undefined (?:references|citations)", log)
    for kind, pt in re.findall(r"Overfull \\([hv])box \(([\d.]+)pt", log):
        if float(pt) > MAX_OVERFULL_PT:
            bad.append(f"overfull {kind}box: {pt}pt")
    return bad


def page_count() -> int | None:
    try:
        import pymupdf
    except ImportError:
        m = re.search(r"Output written on main\.\w+ \((\d+) pages?", (HERE / "main.log").read_text(errors="replace"))
        return int(m.group(1)) if m else None
    with pymupdf.open(HERE / "main.pdf") as d:
        return d.page_count


def main():
    for script in ("make_tables.py", "make_figures.py", "check_numbers.py"):
        if subprocess.run([sys.executable, str(HERE / script)]).returncode:
            raise SystemExit(f"{script} failed")
    overleaf_zip()
    engine = compile_pdf()
    if not engine:
        print("No working TeX engine: upload report/overleaf.zip to Overleaf.")
        return 0
    problems = check_log()
    print(f"main.pdf built with {engine}: {page_count()} pages; {len(problems)} layout or reference problems")
    for p in problems:
        print("  ", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
