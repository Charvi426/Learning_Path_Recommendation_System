# Demo and report: handover

Branch `demo-report`, from `v3-revision`. Nothing under `results/` or `config/` was changed.

## Read this first: the results are not submission-ready

An audit made while writing the report found bugs in the analysis (`DECISIONS.md` DR-1 to DR-8, evidence in
`report/audit/`). They were disclosed, not fixed, because this run was told not to change results.

| | Defect | Evidence | What it invalidates or weakens |
| --- | --- | --- | --- |
| D1 | **The label is inside the `ability` feature** | 57% of DDD/FFF test rows, 18-70% on holdout modules. On topic 1 the feature alone scores AUC 0.875 (DDD) and 0.920 (FFF), where a clean one scores 0.5 | every model metric in v1, v2 and v3, criterion (a), and everything tuned on those metrics. Rankings indirectly |
| D2 | **The exam day is 7-25 days too late** | presentation's last day used instead of `date_submitted` | every exam check, criterion (b), the planner (3 of its 4 CCC 2014J weeks are after the exam) |
| D3 | **v3 was not scored as preregistered** | raw K used instead of K = M*R. Recorded plans and A5 scores reproduce exactly with raw K | all v3 ranking outside validation, criterion (c), the planner evaluations |
| D4 | **Cohort features look ahead** | 27 DDD/FFF and 28 CCC sites change class with revision-period clicks, contrary to the v3 log | topic map (all stages), attribution (v3) |

Statements in earlier summaries and hand-overs that are **withdrawn**:

- "no leakage";
- "nothing on or after the exam is used";
- "v3 was scored with the frozen configuration";
- "v3's ranking gain did not transfer";
- "the memory-model improvement held";
- "attribution never uses revision-period clicks".

`results/SUMMARY.md`, `results/v2/SUMMARY_v2.md` and `results/v3/SUMMARY_v3.md` are frozen and still contain them.
Do not paste from those files.

### The fix (not done; needs a decision)

1. In `compute_state`, cut the features of a due-date row for topic k at min(t_due, t_submit) and exclude k from
   ability. Add a test that flips y_k and asserts that no feature of row k changes.
2. Set the exam day to each student's `studentAssessment.date_submitted`.
3. Build cohort features from earlier presentations, or from days before t.
4. Pass the frozen `knowledge_scale` in `score_outside_validation.py` and assert it.
5. Rerun v1 to v3 on a new branch, as a declared correction. The test sets have been seen, so the rerun is not
   confirmatory.

The earlier full runs took a few hours in total. Conclusions may change in either direction.

## Working prototype (added after the hand-over)

*My learning path* computes a path live from a student's scores, VLE clicks and chosen topics. It uses a separate
model trained on presentations before 2014J. It does not change or correct any v1-v3 result, and the report does not
describe it. Build it with `python demo/build_prototype.py`. See `demo/README.md` and `DECISIONS.md` DR-18 to DR-20.

## Run the demo

```bash
pip install -r requirements-demo.txt      # preferably in a virtualenv
python demo/build_demo_data.py            # raw CSVs in data/raw/; about 1-5 minutes
streamlit run demo/app.py                 # http://localhost:8501
```

- Besides the prototype, four pages: student view, pre-exam planner, how it works, results with their status.
- Measured start to a rendered student view: 6-15 seconds (limit 30).
- A two-minute presentation script is in `demo/README.md`.
- The v3 paths follow the frozen configuration (K = M*R), so the recorded v3 ranking scores do not evaluate them.

## Report

- PDF: `report/main.pdf`. Overleaf upload: `report/overleaf.zip` (compiles on its own).
- **8 pages**, including references and the appendix of full tables.
- No undefined references or citations, and no overfull box over 5 pt.
- Rebuild with `python report/build.py` (see `report/README.md`).
- **Number check** (`report/number_check.txt`): 461 of 461 traced values match their source cells, 394 table numbers
  are traced, 87 result numbers in the prose are generated, 37 other literals are each justified as settings or
  labels, **0 failures**. The checker was also confirmed to fail on a planted number and on an altered one. It proves
  that the numbers match the files, not that the files are right (see above).

## Checks run

| Check | Result |
| --- | --- |
| `pytest` with `requirements-demo.txt` | 81 passed |
| `pytest` in the original environment (no Streamlit) | 67 passed, 14 page-render tests skipped |
| Clean clone + raw CSVs: demo build | works (281 s); demo data equal to the working copy |
| Clean clone: app start | works, 6-15 s; also works with only `demo/` present |
| Clean clone: `make_tables.py`, `make_figures.py` | work; tables identical, vector figures pixel-identical |
| `git diff v3-revision --stat -- results config` | empty |
| PDF build in the clean clone | not rerun; the Overleaf zip was compiled in an empty folder instead |

The existing leakage tests pass **despite D1**. Passing tests are not evidence that the evaluation is clean.

## Skipped or done differently

- `docs/draft_sections.md` contains only the IEEE template, so there was no draft to start from. Results, Discussion
  and Limitations were written from the SUMMARY files, corrected for the defects.
- No TeX was installed. A checksum-verified portable Tectonic was used from a temporary folder, so rebuilding the
  PDF needs a TeX engine or Overleaf.
- The demo shows configuration A5 only. The planner is v3 only, because v2 had none.
- A path can have fewer than 5 items when fewer topics pass the frozen filters.
- `docs/` is left untracked, as supplied.
- Screenshots worked, so there are no figure placeholders.

## A human must review

1. **Whether to submit before fixing D1-D4.** Both authors should read Section V-C of the report and agree to it,
   and tell the instructor if any earlier numbers were already shared.
2. **Authors:** Charvi (2310110452, single name as in the midterm) and Anushi Gupta (2310110419). No e-mail
   addresses, as in the midterm.
3. **References:** the Nagatani author list in the midterm looks wrong, Zhang et al. has no venue, Fan et al. 2026 is
   unconfirmed, and Pavlik and Anderson 2005 has no title. The full list is in `report/README.md`.
4. **Midterm wording:** it says OULAD has "22 courses". It has 7 modules in 22 presentations, as the report says.
5. **AI assistance:** check the course's rules on disclosing it. The analysis, code and report text were produced
   with an AI coding assistant, and the report does not currently say so.
6. **Repository:** v1 accidentally committed 124 MB of student-level sensitivity parquet files
   (`data/processed/sens_*`). Decide before pushing whether to remove them from history.
7. **Global Python:** earlier runs installed pyarrow, CPU torch and pykt globally and, as an unreported side effect,
   upgraded setuptools from 75.6 to 78.1. pykt's `wandb` dependency is missing by choice, so `pip check` flags it.
   This run installed nothing globally.
