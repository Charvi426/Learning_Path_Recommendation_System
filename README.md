# Forgetting- and interest-aware learning path recommendation on OULAD

Implementation of `Methodology.md`: topics from assessment windows, a forgetting-aware PFA knowledge model with a
learned half-life, an interpretable hybrid recommender, and an offline evaluation with ablations A0–A7.

Results: `results/SUMMARY.md`. Design decisions and deviations: `DECISIONS.md`. Status: `PROGRESS.md`.

## Setup

Python 3.12, CPU only.

```bash
pip install -r requirements.txt
# optional: register a Jupyter kernel for the notebooks
python -m ipykernel install --user --name fds312
```

Put the 7 OULAD CSVs (courses, assessments, studentAssessment, vle, studentVle, studentRegistration,
studentInfo) in `data/raw/`. They are not in git (`studentVle.csv` is 450 MB).

Peak memory is about 2 GB. `studentVle` is read in 2M-row chunks and filtered to the selected modules.

## Reproduce every table (five commands)

| Stage | Command | Runtime* | Writes |
| --- | --- | --- | --- |
| 1 Preprocess + audit (§3) | `python scripts/01_preprocess.py` | 0.5 min | `data/processed/{topics,topic_map,interactions,vle_daily,students}.parquet`, `results/tables/audit_*` |
| 2 Factors (§4) | `python scripts/02_build_features.py` | 1.5 min | `data/processed/[sens_*/]{state_*,features_*,queries}.parquet`, `prereq_edges.csv`, `results/tables/interest_validity.csv` |
| 3 Knowledge models (§5) | `python scripts/03_train_models.py` | 1.5 min | `models/fitted_models.joblib`, `results/tables/coefficients.csv`, `model_metrics_val.csv`, `delta_identification.json`, `data/processed/knowledge_state.parquet` |
| 4 Recommender tuning (§6) | `python scripts/04_tune_recommender.py` | 8 min | `config/final_config.yaml`, `results/tables/recommender_tuned_val.csv` |
| 5 Test evaluation (§7) | `python scripts/05_evaluate.py` | ~25 min | `results/tables/*_test.csv`, `exam_retention_check.csv`, `wilcoxon_holm.csv`, `sensitivity.csv`, `fairness*.csv`, figures, `results/sample_paths/` |

\*On a 16-core laptop. Stage 5 includes the two sensitivity variants (each refits the models and retunes A3/A5 on
validation). `python scripts/05_evaluate.py --skip-sensitivity` skips them, and `--dry-run-val` runs the whole
stage-5 code path on validation (written to `results/tables/dryrun_val/`) for debugging without touching the test split.

Optional DKT baseline (Methodology 5.4): `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then
`pip install --no-deps pykt-toolkit` (its `wandb` dependency is not needed), then `python scripts/03b_optional_dkt.py`
(about 6 min; writes `results/tables/dkt_comparison.csv`).

Everything is seeded (`seed: 42` in `config/config.yaml`). Test-split (2014J) metrics are computed only by stage 5 and
the optional 03b. Stage 3 applies the fitted models to test features to write the knowledge state, but scores nothing.

Notebooks (optional, regenerated from code): `python notebooks/make_notebooks.py` builds and executes
`01_eda`, `02_topic_windows`, `03_factor_checks` and `04_results`.

Tests: `python -m pytest -q` (topic construction, no-leakage perturbation tests, metric implementations with
hand-computed NDCG cases, path construction).

## Layout

```
config/          config.yaml (inputs), final_config.yaml (frozen after validation tuning)
data/raw/        OULAD CSVs (not in git)      data/interim/, data/processed/  generated
src/data/        load, clean, topics, split
src/features/    engagement (click cube), interest, prerequisites, difficulty, build, pipeline
src/models/      baselines, pfa, das3h (main forgetting-aware PFA + DAS3H benchmark), decay, knowledge_state, train
src/recommend/   candidates, scoring, path, explain, engine (vectorised)
src/evaluation/  model_metrics, relevance, ranking_metrics, ablation, stats, evaluate
scripts/         01–05
notebooks/       make_notebooks.py + generated .ipynb
models/          fitted models, bootstrap draws
results/         tables/, figures/, sample_paths/, SUMMARY.md
tests/           test_topics.py, test_no_leakage.py, test_metrics.py
```

## Data choices in brief

Modules **DDD** and **FFF** (6 topics in each of 2013B, 2013J, 2014B, 2014J). Train = 2013B + 2013J,
validation = 2014B, test = 2014J. See `DECISIONS.md` for why the other modules were excluded and for every other
deviation from the methodology.
