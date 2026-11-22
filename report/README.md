# Report

`main.pdf` is the built report: 8 pages, IEEEtran conference format. `overleaf.zip` holds the same sources, ready to
upload to Overleaf (set the main file to `main.tex`).

> The report presents the frozen results **and** four defects that make them unreliable (Section V-C). Do not cut
> that section, or the labels in the captions, without fixing the analysis.

## Build

From the repository root, after `pip install -r requirements-demo.txt` and `python demo/build_demo_data.py` (the
forgetting figure reads the demo data):

```bash
python report/build.py            # or: make -C report
```

It runs, in order:

| Step | Output |
| --- | --- |
| `make_tables.py` | `tables/*.tex`, `tables/numbers.tex` (prose numbers), `tables/provenance.json` |
| `make_figures.py` | `figures/*.pdf`, `figures/demo_crop.png` |
| `check_numbers.py` | `number_check.txt`; fails on any untraced number |
| zip | `overleaf.zip` |
| TeX | `main.pdf`; fails on undefined references or boxes more than 5 pt too wide |

The TeX engine is the first of `latexmk`, `pdflatex` + `bibtex`, or `tectonic` (on PATH, or set `TECTONIC` to the
executable). `audit_defects.py` regenerates `audit/` from the raw and intermediate data. It is read-only.

To change a number, do not edit it in `main.tex`: it is a macro that points at a results cell in `make_tables.py`.
A new literal that is not a result needs an entry, with a reason, in `number_allowlist.py`.

## Figures

| File | Content | Made by |
| --- | --- | --- |
| `pipeline.pdf` | pipeline diagram | `make_figures.py` |
| `forgetting.pdf` | one student, v2 and v3 | `make_figures.py` |
| `exam_diagnosis.pdf` | delta and reset rate by topic | `make_figures.py` |
| `ndcg_holdout.pdf` | holdout NDCG@5 with intervals | `make_figures.py` |
| `criteria.pdf` | preregistered criteria | `make_figures.py` |
| `demo_student.png`, `demo_forgetting.png`, `demo_planner.png` | real screenshots | `demo/screenshots.py` |
| `demo_crop.png` | readable crop used in the report | `make_figures.py` |

No figure is a placeholder.

## References a human must check

Only details found in the project's own files were used, and nothing was looked up.

**Likely errors in the midterm's bibliography**

- **[Nagatani et al., 2019]** The midterm's author list (Nagatani, Zhang, Sato, Sonoda, Chen, Xiang, Sun) does not
  match the assistant's recollection of this paper after the first three names. It was shortened to
  "K. Nagatani et al." rather than copied. Check it against the proceedings.
- **[Zhang, Feng, Wang]** No venue, year, volume or pages in the midterm. All left out.
- **[Fan, Tong, Li, 2026]** Confirm that the article exists with that volume and article number. It could not be
  confirmed from memory.

**Incomplete entries**

| Entry | Left out, because it is not in the project files |
| --- | --- |
| Pavlik and Anderson 2005 (ACT-R) | **title, venue, pages, initials**; the entry says "to be confirmed" |
| Pavlik, Cen, Koedinger 2009 | initials, pages, DOI |
| Vie and Kashima 2019 | initials, pages, DOI |
| Choffin, Popineau, Bourda, Vie 2019 | initials, pages |
| Settles and Meeder 2016 | initials, pages, DOI |
| Piech et al. 2015 | full author list, pages |
| Gervet, Koedinger, Schneider, Mitchell 2020 | initials, pages, DOI |
| Liu et al. 2022 (pyKT) | full author list, pages |
| Liu et al. 2019 (CSEAL) | full author list, pages, DOI |
| Ghosh, Heffernan, Lan 2020 (AKT) | initials, pages, DOI |
| Shen et al. 2021 (LPKT) | full author list, pages, DOI |
| Hidi and Renninger 2006 | initials, pages, DOI |
| Gruber, Gelman, Ranganath 2014 | initials, pages, DOI |
| Da Silva et al. 2023; Rahayu et al. 2023; Abdelrahman et al. 2023 | DOI |
| Zhao et al. 2023; Kuzilek et al. 2017 | DOI |
| Kuzilek et al., UCI repository | year, URL, DOI |

The one-line descriptions of cited work in Section II come from the midterm and `docs/Methodology.md`, or from the
cited titles. The papers were not read.
