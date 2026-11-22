# PROGRESS

> **KNOWN DEFECTS (found 2026-11-22, not fixed):** (1) label leakage in `ability` invalidates every model metric in v1-v3; (2) the exam day is 7-25 days late, which undermines exam checks and planner; (3) v3 holdout/2014J ranking and planner were scored with raw K instead of the preregistered M·R, so criterion (c) did not test the preregistered configuration. See `DECISIONS.md`, DR-1 to DR-8.

| Milestone | Status | Notes |
| --- | --- | --- |
| M1 Data audit + preprocessing | done | DDD + FFF, 6 topics per presentation; `topic_map`, `interactions`, `vle_daily` written; notebooks 01, 02 |
| M2 Factors | done | engagement windows, ability, interest proxy (+ validity checks), prerequisites (no data edges at pass mark 40), difficulty; leakage tests pass; sensitivity variants built |
| M3 Models | done | Δ not narrow (no fallback); baselines, PFA, 4 forgetting variants, DAS3H; 200-resample bootstrap; knowledge_state.parquet; DKT deferred |
| M4 Recommender | done | candidates, scoring, vectorised path builder, weight grid (671), tuning A2–A7 on validation; final_config.yaml frozen |
| M5 Evaluation | done | test split evaluated once; model metrics, ranking A0–A7, simulated gain, exam check, Wilcoxon/Holm, sensitivity, fairness; optional DKT done |
| M6 Write-up | done | README, SUMMARY, sample paths, 4 notebooks, figures; pytest 18 pass; clean-clone rerun reproduces all 39 tables |

Blocked items: none. Skipped: nothing mandatory. Optional DKT was run (`scripts/03b_optional_dkt.py`).
See `results/SUMMARY.md` for results and `DECISIONS.md` for every deviation.

## v2 (post-hoc fixes, branch `v2-fixes`)

| Fix | Status | Notes |
| --- | --- | --- |
| Fix 1 exam check | done | no bug (Δ audited, tests added); diagnosis in results/v2/exam_check_diagnosis.md |
| Fix 2 relative filters | done | within-student mode chosen; revision filter works; mastery filter can only be nearly inactive (upcoming-topic rule) |
| Fix 3 popularity term + A1 audit | done | A1 not leaky (unchanged); Pop = training-window share; A8, A9 tuned |
| Fix 4 holdout modules | done | AAA, BBB, CCC, EEE, GGG; frozen settings; results/v2/holdout/ |
| Freeze v2 config, 2014J post-hoc, SUMMARY_v2, reproducibility | done | frozen 2026-11-20 before holdout/2014J; 2014J scored once (post-hoc); 30 tests pass; clean clone reproduces 81/81 CSV + 21/21 JSON; v1 outputs unchanged |

## v3 (post-hoc revision component, branch `v3-revision`)

| Step | Status | Notes |
| --- | --- | --- |
| 1 Resource classes | done | results/v3/resource_classes.csv |
| 2 Attribution | done | direct / coaccess / shared(H); revision visible for 55–60% on topics 2–5 (shared) vs 15–47% (direct) |
| 3 Memory models (validation) | done | V3 (ACT-R spacing, shared 0.4) selected by the fallback rule; val Δρ −0.008 vs −0.097 (v2) |
| 4 Planner (validation) | done | exam Δρ improvement +0.089 [0.048, 0.134] vs v2; observational flagged-share coef −21.5 (not causal) |
| 5 Recommender (validation) | done | v3 A5 0.802 vs v2 A5 0.778 val NDCG@5; config/final_config_v3.yaml frozen |
| Preregistration | done | PREREGISTRATION_v3.md committed before any outside-validation scoring |
| Holdout / CCC / 2014J scoring | done | criteria (a) met, (b) met, (c) not met |
| SUMMARY_v3, reproducibility | done | 42 tests pass; clean clone reproduces 26/26 CSV + 3/3 JSON; v1/v2 outputs unchanged |

## Demo and report (branch `demo-report`)

| Part | Status | Notes |
| --- | --- | --- |
| Read-only audit | done | four defects (DR-1 to DR-7); disclosed, **not fixed** |
| 1 Streamlit demo | done | 7 modules x 300 students; 4 pages; starts in 6-15 s; tests; screenshots |
| 2 IEEE report | done | `report/main.pdf`, 8 pages; `report/overleaf.zip`; number check 0 failures |
| Final checks | done | 81 tests; clean-clone build; results and config unchanged |
| Working prototype | done | live path from scores + VLE clicks + chosen topics; separate leak-free model (DR-18 to DR-20) |
| Corrected evaluation | **not started** | needs the owner's decision; see `DEMO_AND_REPORT.md` |
