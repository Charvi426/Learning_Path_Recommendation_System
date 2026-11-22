# DECISIONS

Dated log of assumptions, deviations from `Methodology.md`, fallbacks and skipped items.
"Methodology" below refers to `Methodology.md` (the file is named `Methodology.md`, not `METHODOLOGY.md`).

## 2026-11-19 — M1 data audit and preprocessing

1. **Project root.** Section 2 shows a root folder `fds-learning-path/`; the project lives directly in the working directory `FDS_Project/` with exactly the section 2 sub-layout.
2. **Environment.** `pyarrow` was not installed; installed `pyarrow 26.0.1` with pip. Notebooks run on a registered kernel `fds312` (the default Jupyter kernel pointed to a Python 3.13 without pyarrow).
3. **Raw data matches the expected schema.** All 7 CSVs have the column names in Methodology 3.1. Missing values are coded as empty strings; `?` is also treated as missing. Presentations: 2013B, 2013J, 2014B, 2014J. 11 assessments have no date (all Exams), filled with `module_presentation_length`.
4. **Module choice (deviation: data decides).** Methodology 3.2 keeps modules with >= 5 non-exam assessments. The audit (`results/tables/audit_module_selection.csv`) shows:
   - CCC: only 2014B/2014J, so no training presentation. Excluded.
   - EEE: 4 non-exam assessments. Excluded.
   - GGG: 9 assessments but 6 CMAs share one due date, so only 4 topics after the same-day merge (3.3), and no 2013B. Excluded.
   - AAA: 5 TMAs but runs only in 2013J/2014J (would need the 50/50 fallback for val/test, about 180 students each). Excluded to keep a single clean temporal design.
   - BBB: runs in all 4 presentations, but 2014J (the test presentation) has 5 TMAs and no CMAs, while 2013B/2013J/2014B have 7 topics. Topic indices and topic intercepts would not transfer from train to test. Excluded.
   - **DDD and FFF selected**: both run in all four presentations and have 6 topics in every presentation, so topic k means the same assessment slot in train, validation and test.
   The 50/50 `id_student` fallback split is implemented in `src/data/split.py` but not triggered.
5. **DDD 2013B CMAs dropped from topic construction.** DDD 2013B has 7 CMAs due 2 days before each TMA; no other DDD presentation has CMAs. Keeping them would give 13 topics (or 8 if near-dates were merged) and break topic alignment. Config `drop_assessment_types: {DDD_2013B: [CMA]}`.
6. **FFF CMA bundle.** FFF's 7 CMAs all share the exam date (weight 0) and are merged into topic 6 (the same-day merge rule). Weights are 0, so the topic score is the plain mean of the CMA scores.
7. **Merged-topic timing.** For a merged topic, `t_submit` = last submission among its parts (the topic is complete when all parts are in), `score` = weighted mean of submitted parts (plain mean if weights sum to 0). `is_submitted` = at least one part has a score.
8. **First window.** Topic 1's window is (t0 - 1, t_1] with t0 = first day of VLE activity in that presentation, so day t0 itself belongs to topic 1.
9. **Banked / missing scores.** A student-topic row whose result is banked, or whose only submission has no score, is removed entirely (not treated as a non-submission): 949 banked and 82 no-score rows.
10. **Click caps** (99.5th percentile of daily clicks per student, site and day, per activity type) are computed on the **training presentations only**, then applied to all splits (`results/tables/audit_click_caps.csv`).
11. **Navigation excluded from topics.** `homepage` sites (4 per module, 26% of DDD and 14% of FFF clicks) are course-wide navigation with no topic content; they get no topic. Their clicks are kept in `data/interim/total_daily_clicks.parquet` for "total activity" checks.
12. **Site → topic mapping by cohort clicks** uses all students of the same presentation (no outcomes, only aggregate click timing). This is treated as course-structure metadata, comparable to the published study calendar, and is not a per-student leak. Shares mapped by each rule are in `results/tables/topic_map_method_share.csv`: DDD 0.1% of sites by `week_from` (97.5% by cohort clicks), FFF 42% by `week_from` (57% by cohort clicks). Clicks after the last due date are ignored in the argmax unless a site has no other clicks. 53 sites with no clicks and no `week_from` stay unmapped.
13. **Outcome imbalance.** Among submitted topic assessments, 94.2% pass at 40. Non-submission (20% of rows) is the main form of "failure". Methodology's default (non-submission excluded from targets) is kept; the "non-submission = fail" sensitivity run (7.6) is reported alongside.
14. **Caveat on topic 1.** Course-wide resources clicked mostly in the first window (for example general forums) map to topic 1. About 55% of topic-1 clicks happen after its window (`results/tables/topic_click_share_inside_window.csv`), so topic-1 "post" and revision signals are inflated.
15. **EDA as notebooks.** `notebooks/make_notebooks.py` builds and executes `01_eda.ipynb` and `02_topic_windows.ipynb` from cells held in that script, so they regenerate deterministically.

## 2026-11-19 — M2 factors

16. **Factor engine.** All factors come from a dense student × topic × day click cube with cumulative sums (`src/features/engagement.py`). A query (s, t) reads only days <= t-1 and outcomes with `t_submit < t`. `tests/test_no_leakage.py` perturbs every event on or after t (clicks and outcomes) and asserts that no feature changes. It also recomputes sampled real validation queries from truncated data.
17. **Windows.** Features are computed for W = {7, 14, 30, ∞}. Models use {7, 30, ∞} by default; {7, 14, 30, ∞} is a tuned/sensitivity alternative for the DAS3H benchmark (the only model that uses windows).
18. **Assessment event time** for succ/fail windows and the forgetting clock is the submission day of the prerequisite (`t_submit`, the last part for merged topics). The result is treated as known from that day.
19. **Δ (forgetting clock)** = t − max(t_submit_j, last day before t with >= 3 clicks on j's resources, if later than t_submit_j). **Δt_{s,k}** (4.2) = t − max(last click on k, submission of k).
20. **ability_s(t)** = (Σ y over topics assessed before t + 2·μ) / (n + 2), μ = training pass rate of the module.
21. **Interest proxy details.**
    - "Opened topics" J_s = topics whose window has opened before t, or that the student has clicked before t.
    - `post` is 0 for topics not yet assessed.
    - The cohort z-score reference is per (module, topic, stage), where stage = number of topic windows open at t. The share mechanically depends on how many topics are open, so comparing it with the cohort at the same stage removes that artefact. Cells with < 30 training rows fall back to (module, topic).
    - I = 0.5 for topics outside J_s and for students with no raw signal yet.
22. **Interest validity** (`results/tables/interest_validity.csv`). Correlations of I with total clicks are small (|r| = 0.03 with the student's total clicks, 0.23 with clicks on the same topic), so the "measures activity" fallback is **not** triggered and vol is kept. Raw correlations between the signals are mixed: vol–post +0.59, but vol–early −0.16 and early–post −0.17. The negative ones are mechanical (early topics have long post periods, late topics have long early periods). Within (module, topic, stage) all three are positive (+0.53, +0.56, +0.20), and both versions are reported. Split-half (odd vs even days) r = 0.49, Spearman–Brown 0.66.
23. **Data-derived prerequisite edges: none at the default setting.** With 94% of submissions passing, no topic has the required >= 200 training students failing it (the maximum is 180). Following Methodology 4.5 exactly, the data graph equals the chain graph, so **A7 is identical to A6 at the default setting**. With pass mark 70, 30 edges qualify (all ordered pairs), and with non-submission = fail, 26. The chain vs data comparison is therefore reported in the pass-70 sensitivity run (7.6), not by relaxing the rule. The rule is not relaxed, to avoid a post-hoc threshold.
24. **Sensitivity variants.**
    - `pass70` re-labels y = 1[score >= 70].
    - `nonsub_fail` turns every non-submission into a submitted fail with score 0 at the due date. It then counts in targets, ability, succ/fail and M.
    Both variants re-derive every training-split statistic (difficulty, cohort mean, z-stats, data edges).

## 2026-11-19 — M3 models

25. **Training rows** are (s, k) with t = t_due_k and an observed outcome (non-submissions excluded by default, 3.2 step 4). Validation and test rows are built in exactly the same way.
26. **Missing prerequisite.** If a prerequisite j was never submitted before t, it carries M = 0 (K = 0) into K̄. Topic 1 has no prerequisite, so K̄ = 0 there and the topic intercept absorbs it.
27. **Identification check (5.1).** The decision rule was fixed before fitting: "narrow" if the median within-topic IQR of Δ is < 7 days. Result: median within-topic IQR = 12 days, SD = 17.9 days, 58% of the variance is within topic, and 66% of prerequisite pairs were revised (>= 3 clicks after submission). **Not narrow, so no 8.2 fallback**; the learned half-life stays the main forgetting model. Caveat: for target topic 2 the prerequisite is topic 1, whose course-wide resources are clicked almost daily, so Δ ≈ 1–2 days there (`results/tables/delta_by_topic.csv`).
28. **Regularisation grid extended.** The first run picked the smallest L2 (1e-4, the grid edge), so the grid was extended to {1e-5, …, 0.1} before any test-split use. L2 penalises θ and φ_c, φ_d, φ_I, but not β_k or φ_0. Inputs (ability, c, I, and pair-level c_j, d_j, I_j) are standardised with training means and SDs. θ_Ic multiplies the standardised I and c.
29. **Half-life bounds.** h is kept in [1, 365] by clipping log2 h (zero gradient outside), with box bounds on φ_0. Three starts for φ_0 (log2 7, 30, 120) are run from the no-forgetting θ, and the best training loss is kept. The analytic gradient was checked against finite differences (max error 1e-7).
30. **Model variants vs. Methodology naming.** 5.1 calls the model *with* φ_I, θ_I, θ_Ic the main model, while 5.2 calls the learned half-life without φ_I "main". Here:
    - `learned` = learned half-life (φ_0, φ_c, φ_d), no interest terms → used for A5;
    - `learnedI` = the full 5.1 equation (+ φ_I, θ_I, θ_Ic) → used for A6;
    - `none` = no forgetting (K̄ = mean M) → A2/A3;
    - `fixed` = exp(−Δ/S) → A4.
    S was chosen on validation log-loss from {7, 14, 30, 60, 120}: **S = 120** (the grid edge; longer is better, again pointing to slow forgetting).
31. **Baselines.**
    - Topic difficulty, PFA and DAS3H use `sklearn.LogisticRegression` with topic one-hot (no intercept), and C tuned on validation log-loss over {0.01, 0.1, 1, 10}.
    - PFA = ability + succ_∞ + fail_∞ + c_∞.
    - DAS3H window set was tuned on validation ({7, 30, ∞} vs {7, 14, 30, ∞}): {7, 14, 30, ∞} won by 0.0002 log-loss and is used as the benchmark and simulator.
32. **Bootstrap.** 200 resamples of training students. A resample is implemented as integer sample weights per student (equivalent to duplicating rows), with standardisation fixed from the full fit so the coefficients stay comparable. It took about 1 minute in total, so no reduction to 100 was needed.
33. **φ_c < 0 and φ_d > 0.** Clicks and active days on a topic are strongly collinear, so their separate signs are not interpretable. Their sum effect is what the model uses. Reported as is.
34. **"RMSE/MAE for ỹ".** No separate score-regression model was fitted; RMSE_score / MAE_score compare the predicted pass probability with the normalised score ỹ (a rough proxy, labelled as such).
35. **DKT (optional)** is deferred until everything else works (see later entries).

## 2026-11-19 — M4 recommender

36. **Decision points.** One per (s, k), k = 2..6, at t = (submission day of k−1) + 1, kept only if t <= t_due_k and the student is still registered for k. k_now = k (the next assessment due).
37. **Relevance (7.2).** The window is [t_decision, t_due_k].
    - Rel-A applies to students who passed k; Rel-B to students who submitted k and scored < 40.
    - Non-submitters of k have no relevance set by default (in the `nonsub_fail` variant they count as Rel-B).
    - "A topic in both sets counts 2" can never happen as Rel-A and Rel-B are defined (disjoint students), so graded relevance reduces to binary. The tuning objective pools Rel-A and Rel-B decision points ("combined").
    - Metrics are averaged within student first, then over students. Decision points without any relevant topic are skipped.
38. **Path ordering (6.4 steps 3–4).** Items are taken in score order. When the next item is a new topic with unplaced in-path prerequisites, its earliest such prerequisite is placed first (recursively): "insert prerequisites before it". Revision items are not reordered. Unit-tested in `tests/test_metrics.py`.
39. **Score ties** (for example every revision item has Q = 1) are broken by the lower topic index, identically for all configurations.
40. **A2 "Mastery only"** uses no prerequisite information at all: no Q filter and no prerequisite ordering. A3–A7 use the Q filter and ordering with their graph.
41. **A0 / A1** rank the universe {assessed topics} ∪ {unassessed topics <= k_now + 2} with no thresholds and no revision cap (pure rankers). A0 uses random scores (seed 42). A1 scores topic j by the mean clicks of **training** students on j in the same (module, k_now) decision window.
42. **Threshold grid extended (deviation, validation-only).** With the Methodology grid {0.4, …, 0.8}, the tuned τ_m, τ_r and p* all landed on the grid edge (0.8). Diagnosis: with a 94% pass rate, the predicted mastery of unassessed topics is usually > 0.9, so τ_m <= 0.8 removed the current topic from almost every candidate set (paths averaged about 1.2 new items; the current topic led the path in about 3% of cases). The grid was extended to τ ∈ {0.4, …, 0.8, 0.9, 1.0} and p* ∈ {0.6, 0.7, 0.8, 0.9} **before any test-split evaluation**. Validation then picks τ_m = τ_r = 1.0 for A2–A7, which switches off the mastery and retention filters: ranking does the work, and only the readiness filter Q >= τ_q stays active. The first-grid run was not committed. Its validation results (copied from that run's console output) were: NDCG@5 / Recall@5 = A2 0.330/0.417, A3 0.429/0.401, A4 0.458/0.424, A5 0.456/0.423, A6 0.457/0.425 (τ_m = 0.8, τ_q = 0.4, τ_r = 0.8, p* = 0.8 for A3–A6). The same ordering A4 ≈ A5 ≈ A6 > A3 held there. The forgetting term F and the interest weight ε were both tuned to 0 in every configuration. Forgetting influences A4/A5 only through K (in Q and D), and A6 differs from A5 only through the learnedI model's K and M.
43. **A7 = A6** at the default setting (no data-derived edges, #23). final_config records `identical_to: A6`.
44. **Frozen configuration**: `config/final_config.yaml` (written by `scripts/04_tune_recommender.py`).

## 2026-11-19 — M5 evaluation

45. **Validation dry run before the test run.** `scripts/05_evaluate.py --dry-run-val` executes the complete stage-5 code path on validation and writes to `results/tables/dryrun_val/`, so bugs could be fixed without touching the test split. The dry run surfaced no bug. One validation observation was investigated: A3–A7 have a much smaller simulated gain than A0–A2. Cause: the readiness filter Q >= τ_q removes the current topic for the roughly 6% of decision points where the student failed or missed prerequisite k−1. Those are the low-p cases where the logistic simulator gives the largest gain for added clicks. This is a property of the design, so nothing was changed. The test split was then evaluated **once** (`results/eval_test_run.log`).
46. **Simulated gain (7.3).**
    - Simulator: the DAS3H window model with the validation-chosen windows {7, 14, 30, ∞}.
    - "Do nothing" = the target topic k's features at the decision time.
    - A Learn/Practice item for k adds the training median of per-topic clicks (module-specific) to every click window of k.
    - A Revise item for a prerequisite j of k moves j's outcome inside every succ/fail window, as if it were just practised (the window-model analogue of resetting Δ to 0).
    - Items on other topics cannot change P(y_k) in this simulator.
    - Gains are averaged within student, then over students. Labelled model-based.
47. **Exam check (7.4).** For each test student with an exam score: mean M and mean K over all 6 topics at the exam date. Unassessed topics take M = K = the model prediction (5.5). Reported for the fixed, learned and learnedI variants, with student-bootstrap CIs on K−M differences. A validation copy is written as supplementary (`exam_retention_check_val.csv`).
48. **Wilcoxon + Holm.** The primary family is the three main comparisons on per-student NDCG@5 with combined relevance. Rel-A-only and Rel-B-only versions, and the simulated-gain versions, are reported as supplementary families, each Holm-corrected within itself.
49. **Fairness.** NDCG@5 (combined) for A3 and A6, and AUC for their models (`none` and `learnedI`), by gender, IMD band (missing as its own group) and disability. The largest gap is computed over groups with >= 30 students/rows.
50. **Sensitivity.** Each variant (pass mark 70; non-submission = fail) refits all knowledge models on train (no bootstrap), retunes A3 and A5 on its own validation set (and A3/A5 with the data-derived graph, which exists in both variants), then evaluates test once. "Windows {7, 14, 30, ∞}" only changes the DAS3H simulator, since A3/A5 use no windows, so it is reported as the A5−A3 simulated-gain difference under each simulator.
51. **DKT (optional) done.**
    - `torch 2.15 (CPU)` was installed from the PyTorch CPU index. `pykt-toolkit 0.0.38` was installed with `--no-deps`: its `wandb` dependency is only needed by pyKT's training scripts, not by the model class.
    - `scripts/03b_optional_dkt.py` uses pyKT's `DKT` class (emb_size 200, dropout 0.1) with Adam (lr 1e-3), early-stopped on validation AUC (best epoch at 16).
    - DKT cannot predict a sequence's first step, so it is compared with the other models on exactly the rows with >= 1 earlier response (`results/tables/dkt_comparison.csv`).
    - DKT has fixed defaults and was not adjusted after its test numbers were seen. It ran after the main test evaluation had started.
52. **Added after the test run (descriptive only).** The paired student-bootstrap of AUC and log-loss differences between models (`model_paired_diffs_test.csv`) was added to `scripts/05_evaluate.py` after the single test run, because the marginal model CIs overlap. It uses the already-fitted, frozen models and changes no model, threshold or weight. It was first computed with a one-off call on the same objects; a clean rerun of stage 5 reproduces it.
53. **`topic_click_share_inside_window.csv`** was first written by notebook 02. It is now written by `scripts/01_preprocess.py` (same numbers), so scripts 01–05 alone regenerate every table in `results/tables/`. The DKT tables (`dkt_*`) come from the optional `scripts/03b_optional_dkt.py`.

## 2026-11-19 — M6 write-up and quality checks

54. **Test-split rerun for the reproducibility check.** The quality checks require scripts 01–05 to reproduce `results/tables/` from a clean clone, so the clone reruns stage 5 on the test split. Nothing was changed between runs (same code, same frozen configuration, same seed), and the rerun is only compared with the first run, never used to choose anything.
55. **Sensitivity of the data-derived graph.** Where data edges qualify (pass 70, non-submission = fail), they make the graph almost complete and both A3 and A5 get worse; A5 then falls below A3. This is reported as a negative result for data-derived prerequisites. Nothing was retuned in response.

# v2 (post-hoc)

All entries below were made on branch `v2-fixes` after the v1 test split had been scored (tag `v1` = commit e91c04d on `master`). Each is tagged **BUG-FIX** (code did not do what Methodology.md specified) or **CHANGE** (new design choice). v1 outputs (`results/tables/`, `config/final_config.yaml`, `results/sample_paths/`, `results/SUMMARY.md`) are not modified. v2 outputs go to `results/v2/` and `config/final_config_v2.yaml`.

## 2026-11-20 — v2 Fix 1: exam retention check

v2-1. **Branch naming.** The v1 branch is `master`, not `main`, so the final v1-unchanged check is `git diff master --stat -- results/tables config/final_config.yaml`. Tag `v1` marks the v1 state.
v2-2. **Audit, no bug (no tag needed).** Validation shows the same negative sign as test (learned Δρ = −0.097 [−0.142, −0.054]). Δ at the exam date was recomputed independently for all 2,930 validation pairs and matches the pipeline exactly. Revision-period clicks reset Δ, and events on/after the exam date are not used. Δ-reset unit tests were added (`tests/test_v2_exam_delta.py`). **No BUG-FIX**; the exam-check numbers are unchanged.
v2-3. **Scope correction (documentation).** OULAD has exam scores only for CCC and DDD, so the v1 exam check is DDD-only (FFF has none). v1 `results/SUMMARY.md` did not say this; it is stated in the v2 summary.
v2-4. **Diagnosis.** Topics 2–5 drive the negative Δρ: they are rarely re-opened after submission, so their median Δ at the exam is 83–188 days. R for them carries a much weaker exam signal than the score. Revision appears to go through topic-1 (course-wide) and topic-6 resources. Details: `results/v2/exam_check_diagnosis.md`. No further decay variants were tried, as instructed.

## 2026-11-20 — v2 Fix 2: relative candidate filters

v2-5. **CHANGE: relative filters replace τ_m / τ_r** (`src/recommend/v2.py`). Both options were implemented:
    - within-student: revise if K_{s,j} is among the lowest ceil(q·n) of the student's n assessed topics; exclude a new topic as "already mastered" if it is among the floor(q·n) highest predicted-mastery unassessed topics within the horizon;
    - cohort quantile: revise if K < the q-quantile of training-cohort K for that topic; exclude a new topic if its predicted mastery is above the (1−q)-quantile of the training cohort's predicted mastery for that topic. Quantiles come from the training split's decision points (`data/processed/v2/knowledge_state_train.parquet`).
    The new-topic rule mirrors the revision rule (bottom q% → revise; top q% → mastered), which is the relative analogue of M < τ_m.
    Grid: mode × q_rev × q_new ∈ {10, …, 50} × τ_q (v1 grid) × p* (v1 grid). Tuned on validation exactly as in v1: filters with equal weights → weight grid → filters again.
v2-6. **CHANGE: admissibility rule** (from the run instructions), checked on validation for every filter setting during tuning:
    - reject if the mastery filter removes the upcoming topic at > 10% of decision points;
    - reject if the revision filter flags a topic at < 1% of decision points (among those with any assessed topic);
    - reject if the mastery filter excludes a topic at < 1% of decision points ("flags nothing").
    The 1% floor is my operationalisation of "flags nothing".
v2-7. **Result (validation).**
    - Within-student mode wins for A2–A6: q_rev = 50 (grid edge; the grid was fixed by the instructions) and q_new = 30.
    - **Every cohort-quantile setting is inadmissible**: its mastery filter removes the upcoming topic at 22–77% of decision points. Validation students' predicted mastery of the upcoming topic exceeds the training cohort's quantiles.
    - The only admissible mastery filters are almost inactive: within-student q_new = 30 excludes a topic at 1.4% of decision points, because floor(0.3·n) >= 1 only when >= 4 unassessed topics are in the horizon. q_new >= 40 removes the upcoming topic at 55–63%.
    - Conclusion: **the revision filter does real work** (1.66 revision candidates per decision point vs. every assessed topic in v1). **A relative mastery filter cannot do real work without dropping the upcoming topic**, because the upcoming topic usually has the highest predicted mastery among a student's unassessed topics.
    - Diagnostics per configuration: `results/v2/validation/tuned_A*.json` → `val_filter_diagnostics`.

## 2026-11-20 — v2 Fix 3: popularity term and A1 audit

v2-8. **A1 audit: no leakage, no BUG-FIX.** v1's A1 (`scripts/04_tune_recommender.py`, `load_rec_inputs`) scores topic j by the mean clicks of **training-split** students (2013B/2013J) on j in the same (module, k_now) decision window. Those clicks come after the decision time, but from earlier cohorts, so this is historical data. The scored (validation/test) presentation is never read, and the score is identical for every student at the same k_now. **Corrected A1 = v1 A1** (numbers unchanged). A1's strength comes from Rel-A being defined by engagement in the window, which is what cohort popularity predicts, not from leakage.
v2-9. **CHANGE: Pop(k, t) term.** Four leakage-free definitions were compared on validation by the NDCG@5 (combined) of a Pop-only ranking (`results/v2/validation/pop_definition_selection.csv`):
    - `train_window`: v1 A1 scores, row-max normalised to [0, 1] → 0.945;
    - current presentation's cohort clicks in [t − w, t − 1] for w = 7, 14, 30 → 0.889, 0.889, 0.887.
    **`train_window` chosen.** Tests: `tests/test_v2_pop_and_filters.py` asserts that the cohort Pop is unchanged when every event at/after t is altered, that it matches a manual window sum, and that `train_window` reads only training-split windows.
v2-10. **CHANGE: six-term score** S = αG + βQ + γF + δD + εI + ζPop (weights >= 0, sum 1, step 0.1, ε <= 0.2): 1,001 weight vectors for A8 and 2,211 for A9. Tuned on validation NDCG@5 as in v1, with the Fix 2 relative filters.
    - **A8 = A5 + Pop:** validation NDCG@5 0.786 (A5: 0.778).
    - **A9 = A6 + Pop:** 0.790 (A6: 0.781).
    - Tuned weights for both: G 0.3, F 0.2, D 0.3, Pop 0.2. With popularity in the score, the forgetting term F gets non-zero weight for the first time.

## 2026-11-21 — v2 Fix 4: holdout modules; 2014J post-hoc run

v2-11. **CHANGE: holdout modules as a fresh test set** (`scripts/v2/fix4_holdout.py`). All five modules excluded in v1 (AAA, BBB, CCC, EEE, GGG) meet the rule ">= 4 topics in every presentation and >= 2 presentations" (`results/v2/holdout/holdout_module_eligibility.csv`), so none was dropped.
    - Topic windows are per presentation with the same-day merge and no assessment-type drops. BBB has 7, 7, 7 topics in training and 5 in 2014J; topic intercepts are keyed by (module, k), and all 5 test topics have training intercepts.
    - Each module runs independently: train = earlier presentations, test = 2014J, no validation split.
    - Model coefficients were refitted per module with **frozen** v1 hyper-parameters (L2 per variant, S = 120, C and DAS3H windows {7, 14, 30, ∞}). Recommender settings are frozen from `final_config.yaml` (v1) and `final_config_v2.yaml` (v2).
    - Quantities defined *relative to the training cohort* are computed from the module's own training presentations with the frozen definitions: v2 cohort quantiles (not used, since within-student mode was chosen), the training-window Pop, the simulator's median clicks, difficulty and the interest z-stats.
    - **Nothing was tuned on holdout data.** Chain prerequisites only.
v2-12. **Implementation details (holdout).**
    - Topics that do not exist in a student's presentation (e.g. BBB 2014J topics 6–7) are masked: never candidates, never relevant, and excluded from the popularity universe.
    - The interest-validity checks were skipped for holdout modules (memory; they are diagnostics, not inputs).
    - The exam check is possible only for CCC (the only holdout module with exam scores in OULAD).
    - Pooled metrics concatenate students across modules (bootstrap over students). Pooled AUC mixes modules with different base rates: the constant "majority" model gets pooled AUC 0.46, not 0.5. Per-module AUCs are therefore reported too (`model_metrics_holdout_by_module.csv`).
v2-13. **2014J (DDD/FFF test) scored once with the frozen v2 config** (`scripts/v2/eval_dddfff.py --split test`), after `final_config_v2.yaml` was frozen (commit 482ee8f) and after the holdout run. All 2014J v2 numbers are **post-hoc / exploratory**. The v1 rows recomputed in that run reproduce the v1 test tables exactly (e.g. A3 0.828, A5 0.840, A1 0.946 NDCG@5).
v2-14. **Reproducibility.** A clean clone of `v2-fixes` rerunning v1 01–03 plus all v2 scripts reproduced 81/81 CSVs (max |diff| = 0) and 21/21 JSONs in `results/v2/`. This rerun scored 2014J a second time with the same frozen config (identical numbers); it is a reproducibility check, not a new evaluation.

# v3 (post-hoc)

Branch `v3-revision` (from `v2-fixes`). All v3 work is post-hoc: every exam set (DDD 2014J, CCC holdout) and the 2014J test split have been looked at before. Design and tuning use only DDD/FFF validation (2014B), including the DDD 2014B exam scores. v1/v2 outputs are not modified; v3 outputs go to `results/v3/` and `config/final_config_v3.yaml`.

## 2026-11-21 — v3 Step 1: resource classes

v3-1. **CHANGE: NAVIGATION vs CONTENT** (`src/v3/attribution.py`, `results/v3/resource_classes.csv`). The rule was fixed before any modelling and not tuned:
    - NAVIGATION = every `homepage`, plus `forumng`, `subpage` and `glossary` sites whose cohort clicks are spread evenly over the course weeks (normalised weekly-click entropy >= 0.80);
    - everything else is CONTENT. Content types (oucontent, resource, url, quiz, page, …) are never navigation, even when clicked evenly: revisiting a quiz or content page is practice.
    Evenness uses the aggregate clicks of the site's own presentation (click timing only, no outcomes). Navigation clicks count as practice of no topic.
v3-2. **Result.** 344 navigation sites (homepage 22, forumng 86, subpage 224, glossary 15). They carry 32–61% of each module's clicks; 59% in DDD and 33% in FFF. v1/v2 mapped forums and subpages to topics (mostly topic 1), which is part of why topic 1 looked "course-wide" (v1 #14).

## 2026-11-21 — v3 Step 2: attribution of practice to topics

v3-3. **CHANGE: attributed practice** (`src/v3/attribution.py`, `src/v3/data.py`). Three rules: direct, coaccess and shared (H ∈ {0.4, 0.6, 0.8}).
    - Weights use only clicks inside topic windows (t_open_k, t_due_k], so they are pre-assessment by construction; clicks after the last due date (revision period) are never used. Tests: `tests/test_v3_attribution.py`.
    - **Deviation: "training cohort".** OULAD site ids are presentation-specific (0 of 6,364 sites appear in two presentations), so validation, test and holdout sites have no training-cohort clicks. Weights therefore use the aggregate pre-assessment clicks of the site's own presentation. This is the same cohort-metadata convention as the v1 site→topic mapping (#12), with no outcomes involved.
    - Sites with no window clicks fall back to their v1 topic mapping.
v3-4. **CHANGE: shared rule details.**
    - "Final-window" = sites mapped to the last topic of the presentation.
    - "Exam preparation" = > 50% of the site's cohort clicks after the last due date.
    - A shared site's click on day d is spread uniformly over the topics assessed before d **plus the topic whose window contains d**. Without the second part, final-window content clicked during the final window would credit no topic, not even its own.
v3-5. **Note: "direct" in v3 is not identical to v1/v2.** v3's direct attribution excludes navigation sites (Step 1), while v1/v2 mapped forums and subpages to topics. V0 (the reference) keeps the exact v2 features.
v3-6. **Dose definition.** Dose on day d = Σ_site w(site, j[, d]) · log1p(clicks of that site on day d), using the v1 capped clicks.
v3-7. **Diagnostic** (`results/v3/attribution_revision_diagnostics_val.csv`, DDD validation exam-takers). Share with a revision-period dose >= log1p(3) on topics 2–5:
    - direct: 15–47%;
    - shared (any H): 55–60%.
    Revision that was invisible under direct mapping becomes visible.

## 2026-11-21 — v3 Step 3: history-based memory (validation only)

v3-8. **CHANGE: ACT-R activation + learned K** (`src/v3/memory.py`).
    - Events: every day with attributed practice dose > 0, plus the submission as one event with dose x_sub.
    - m = ln Σ x_i (t − t_i + 1)^(−d_i), using events strictly before the cut-off.
    - Spacing rule: d_i = c·exp(m(t_i)) + a, where exp(m(t_i)) = the summed decayed dose of strictly earlier days.
    - K = σ(b0 + bM·logit M + bm·z(m) [+ bI·z(I)]), M clipped to [0.01, 0.99].
    - The outcome model uses mean K over P(k) in place of K̄. A prerequisite never submitted before t gives K = 0, as in v1.
    - **Fitting split:** coefficients (β, θ, b) are fitted on train. Decay parameters (d, or a and c), x_sub, the attribution rule and H are hyper-parameters chosen on validation log-loss. This mirrors v1, where S was chosen on validation. On the holdout, coefficients are refitted and the hyper-parameters stay frozen.
    - Tests: `tests/test_v3_memory.py` (monotone decay, rise after practice, spacing effect, no events at/after the cut-off).
v3-9. **Fix to my own tuning procedure (before preregistration, validation only).** The first outer L-BFGS-B run used finite-difference steps of 1e-3, at which the validation log-loss is flat to ~1e-6, so it stopped at its starting point (d = 0.5, a = 0.177, c = 0.217, x_sub = e). The search now runs a coarse grid first, then L-BFGS-B refinement (eps = 0.02):
    - fixed rule: d ∈ {0.1, …, 1.5} × x_sub ∈ {0.5, 2.7, 10};
    - spacing rule: a ∈ {0.05, 0.177, 0.35, 0.6} × c ∈ {0, 0.1, 0.217, 0.4} × x_sub ∈ {0.5, 2.7, 10}.
    Only the second run is used. For transparency, the first run gave V2 0.15836 / V3 0.15806 validation log-loss and Δρ −0.009 / −0.010.
v3-10. **Attribution choice per variant.**
    - V1 and V4: best of coaccess / shared(0.4, 0.6, 0.8) by validation log-loss.
    - V3: attribution chosen with the fixed rule (shared 0.4), then fixed vs spacing compared on that attribution.
v3-11. **Result (`results/v3/variant_comparison_val.csv`, DDD/FFF validation).** Validation log-loss / within-course AUC / DDD-2014B exam Δρ:
    - V0 (v2 learned half-life): 0.1613 / 0.898 / −0.097;
    - V1 (learned half-life, attributed): 0.1616 / 0.898 / −0.041;
    - V2 (ACT-R fixed, direct): **0.1578** / **0.903** / −0.014;
    - V3 (ACT-R spacing, shared 0.4): 0.1580 / 0.902 / **−0.008**;
    - V4 (DAS3H on attributed): 0.1623 / 0.885 / −0.168.
    No variant reaches Δρ >= 0, so the preregistered fallback applies: highest Δρ among variants within 0.002 of the best log-loss (V2, V3) → **V3 selected**. Frozen parameters: a = 0.60 (a grid value), c = 0.217, x_sub = 10.0.
    Bound and edge notes:
    - V2's d went to the upper bound 2.0 (very fast decay of single practice days).
    - V3's a = 0.6 is the top grid value, and refinement did not move it.
    - "Within-course" AUC = row-weighted mean of per-module AUCs.

## 2026-11-21 — v3 Step 4: pre-exam revision planner (validation)

v3-12. **CHANGE: planner** (`src/v3/planner.py`).
    - Decision points at exam − 28, − 21, − 14 and − 7 days for every exam-taker.
    - Topics = those submitted before the point. K(exam) uses events before the point only, aged to the exam date.
    - Top 3 by priority M − K(exam), priority > 0. Interest terms are not used (A5-type model).
v3-13. **CHANGE: planner and recommender use the mastery-scale knowledge K_score = M·R**, where R = K(t)/K(t_submit + 1) clipped to [0, 1], as the methodology defines R from the learned K.
    - The learned K = σ(b0 + bM·logit M + bm·z(m)) is identified only up to θ_K and sits well below M (b0 ≈ −3.8), so it cannot serve as a pass-probability-scale value in D or in the "estimated knowledge" text.
    - With K_score, F = M − K_score = M(1 − R), exactly Methodology 6.2.
    - Chosen on validation by v3 A5 NDCG@5 (`results/v3/knowledge_scale_selection_val.csv`): raw K 0.726 (weights collapse to Q only) vs K_score 0.802.
    - The **outcome model and the exam check keep the learned K** (unchanged definition from Step 3).
v3-14. **Validation results.**
    - Exam check, DDD 2014B: v3 Δρ −0.008 vs v2 −0.097; paired improvement +0.089 [+0.048, +0.134].
    - Observational regression (not causal, n = 472; 49 exam-takers with no revision practice on assessed topics excluded): exam ~ mean M + log1p(28-day clicks) + share of revision on topics flagged at week −4. The flagged-share coefficient is −21.5 [−37.5, −7.1]. Students who spent more of their revision on flagged topics scored lower, given M and activity. Plausibly confounded: struggling students revise what they found hard.
    - The planner always fills 3 slots (every assessed topic loses some predicted knowledge).
    - 5 sample plans: `results/v3/sample_revision_plans.md` (validation students).

## 2026-11-21 — v3 Step 5: recommender with the v3 knowledge state (validation)

v3-15. **CHANGE: weights re-tuned, filters frozen.**
    - Filters (within-student, q_rev 50, q_new 30, τ_q, p*) are frozen per configuration from `final_config_v2.yaml`; only the six-term weights were re-tuned on validation NDCG@5.
    - Models: A3 = `none` (unchanged; reproduces v2 A3 exactly); A5/A8 = v3 (selected variant); A6/A9 = v3 with interest terms (θ_I, θ_Ic, bI; L2 1e-4 as v1 learnedI).
    - Tuned weights: A5/A6 D 1.0; A8/A9 G 0.2, F 0.2, D 0.2, Pop 0.4.
v3-16. **Validation NDCG@5 (combined):**
    - v3 A5 0.802 vs v2 A5 0.778 (+0.024, p_Holm < 1e-4); v3 A5 vs v3 A3 +0.029;
    - v3 A8 0.825; v3 A9 0.828 (A9 − A8 +0.003).
    On Rel-B, v3 A5 is worse than v2 A5 (0.703 vs 0.757). Frozen in `config/final_config_v3.yaml`.

## 2026-11-21 — v3 scoring after preregistration (commit 9c44407)

v3-17. **Holdout (primary) and CCC/2014J (exploratory) were each scored once** with `scripts/v3/score_outside_validation.py`. Nothing was changed after scoring. The recomputed v2 references reproduce v2's published values: CCC Δρ −0.0955 (v2: −0.096), holdout v2 A5 NDCG@5 0.718, DDD 2014J Δρ −0.088.
v3-18. **Preregistered criteria** (`results/v3/preregistered_criteria.csv`): (a) met, (b) met, (c) **not met**. Details in `results/v3/SUMMARY_v3.md`.
v3-19. **Reproducibility.** A clean clone of `v3-revision` (v1 01–03, v2 holdout preparation, all v3 scripts) reproduced 26/26 CSVs and 3/3 JSONs in `results/v3/` exactly. This rerun re-scored the holdout and 2014J with the same frozen configuration; it is a reproducibility check only.

# Demo and report

Branch `demo-report` (from `v3-revision`). No experiment, refit for evaluation, re-tuning or change under `results/` or `config/`.

## 2026-11-22 — defects found while preparing the report (read-only audit: `report/audit_defects.py`, `report/audit/`)

These correct earlier entries in this file and statements in the three SUMMARY files, which are frozen and therefore still contain them. The instruction for this run was to change no result, so the defects are **disclosed in the demo and the report, not fixed**.

DR-1. **BUG, critical: the label is inside `ability`** (`src/features/build.py`, `compute_state`).
- Outcome rows for topic k are evaluated at t = t_due_k, and `ability` averages y over every topic with t_submit < t. A student who submitted k before the due date therefore has y_k in their own feature.
- Affected rows: 57% of DDD/FFF test rows, and 18–70% of test rows on the holdout modules (`label_in_ability.csv`).
- On topic 1, where no history exists, the stored feature alone has test AUC 0.875 (DDD) and 0.920 (FFF). In CCC topic 1, the only case with no affected row, it is 0.51 (`first_topic_ability_auc.csv`).
- The target topic's clicks, windows and interest also include post-submission days on those rows.
- **Consequences:**
  - every AUC, log-loss, Brier score, calibration and coefficient in v1/v2/v3 is invalid as an estimate of predictive performance;
  - the comparison with DKT, which does not use the feature, is not like for like;
  - hyper-parameter and variant selection, including v3's, used contaminated losses;
  - rankings, exam checks and the planner are affected indirectly, through fitted coefficients and tuned settings. Only A0 and A1 use no fitted model.
- **Withdrawn claims:** "no leakage" (M2, #16, the v1 summary). The leakage tests pass because they perturb only events on or after t, and this event is before t.

DR-2. **BUG, critical for the exam check and planner: wrong exam day** (#3, following Methodology 3.2).
- `t_exam` is the presentation's last day. The median recorded sitting (`studentAssessment.date_submitted`) is 7–25 days earlier (`exam_day.csv`).
- "Knowledge at the exam" therefore includes post-exam clicks and ageing.
- Planner weeks that fall after the median exam: 1 of 4 on DDD 2014B, 2 of 4 on DDD 2014J, 3 of 4 on CCC 2014J.
- The observational planner regressions largely measure post-exam behaviour.
- **Withdrawn claim:** v2-2, "events on/after the exam date are not used". That audit compared against the wrong date.

DR-3. **False claim in v3-3: "revision-period clicks are never used".** The shared rule's exam-preparation test uses clicks after the last due date. They change the class of 27 DDD/FFF and 28 CCC sites (`attribution_revision_clicks.csv`). `tests/test_v3_attribution.py` checked this only for co-access.

DR-4. **Transductive cohort features** (#12, v2-11, v3-1, v3-3). The site-to-topic map, the navigation classes and the attribution weights use the scored presentation's own whole-course clicks, including days after the prediction time. This breaks the "training split only, strictly before t" rule. It was logged as "metadata", which understated it.

DR-5. **Interpretation corrections.**
- (a) Criterion (b) is also satisfied by K = M, which has Δρ = 0 by definition. Meeting it does not show that modelling forgetting helps. The fitted CCC activation weight is bm = 0.084, against bM = 2.25 on logit M.
- (b) v3 A5 puts all weight on D = 1 − |K − p*|/p*. For K below p* = 0.9 this increases with K, so it ranks the *best retained* revision topic first.
- (c) Rel-B is a fixed function of k (k, k−1, k−2), so a rule with no personalisation scores NDCG 1.0 on it.
- (d) A0/A1 rank an unfiltered universe, while the other configurations return filtered, often shorter lists.
- (e) The holdout was not fresh for v3: its v2 results had been seen and motivated v3.
- (f) The rise in "visible revision" under shared attribution is partly mechanical, because one click is credited to every assessed topic.
- (g) `SUMMARY_v3.md` says the exam check uses K "as the planner uses it". The planner uses M·R and the check uses the learned K.
- (h) `SUMMARY_v3.md` says 15–47% is "under the v1/v2 mapping". It is v3's content-only direct mapping (v3-5).

DR-6. **Supplied inputs.** `docs/draft_sections.md` is the unmodified IEEE conference template converted to Markdown, with no Results, Discussion or Limitations draft in it. Those sections are written from the SUMMARY files, corrected by DR-1 to DR-5. `docs/methodology.md` is `docs/Methodology.md`, identical to the root copy. `docs/` is left untracked as supplied, because the PDF carries personal details and nothing needs it at build time.
DR-7. **BUG, critical: v3 was not scored as preregistered.**
- `PREREGISTRATION_v3.md` and `config/final_config_v3.yaml` fix the recommender and planner knowledge scale to K_score = M·R.
- `scripts/v3/score_outside_validation.py` was first committed at Step 3 (92188a0), before that option existed, and was never updated. It calls `C.inject(...)` and `PN.plans(...)` with their default, **raw K** (lines 86, 87, 104, 172, 181, 182). The preregistration commit contained both the text and the contradicting code.
- Verified: the recorded CCC plans equal the raw-scale plans (max difference 1e-16; 0.385 against M·R), and the recorded 2014J A5 NDCG@5 (0.7303) is reproduced exactly with raw K.
- **Affected:** every v3 A5/A6/A8/A9 ranking number on the holdout and on 2014J, their Wilcoxon tests, **criterion (c)**, and the CCC and DDD 2014J plans and observational regressions. Validation used M·R and is consistent with the preregistration. The scale does not affect criteria (a) and (b), which have DR-1, DR-2 and DR-5a instead.
- **Withdrawn claims:** v3-17 ("with the frozen configuration"), v3-18's reading of (c), and SUMMARY_v3's "the validation gain in ranking did not transfer".
- Criterion (c) stays recorded as **not met**, because results are frozen and the criterion is not reinterpreted. It is labelled "scored on a configuration that was not the preregistered one". The correctly configured result has deliberately **not** been computed and is unknown.
- The clean-clone check could not catch this: it reproduced the same bug.

DR-8. **Demo and report policy that follows from DR-1 to DR-7.**
- Recorded numbers are shown unchanged and generated from the CSVs, each with a validity label, and the defects are stated in the abstract, the captions and the demo.
- No finding is presented as established.
- The demo's v3 paths and plans follow the frozen configuration (M·R), as "frozen configs" requires. They are illustrations and are labelled as different from what was scored. They are never compared with outcomes, so they create no new result.
- The recorded status of each criterion (met, met, not met) is kept, next to the reason it cannot be relied on.

## 2026-11-22 — Part 1: demo

DR-9. **Demo choices.**
- **Configuration:** A5 under v2 and v3, the configuration criterion (c) names. Week = a frozen decision point, labelled by its week and day (the only times at which the pipeline recommends).
- **Status:** "Fading" = the frozen within-student revision flag, so no new threshold. "Not started" = not assessed yet.
- **Selection:** 300 test students per module (seed 42), including forced no-click and withdrawn cases, under new ids such as `DDD-001`. No original ids, demographics or outcomes are written.
- **Default student:** chosen by rule (DDD, sat the exam, not withdrawn, most decision points, lowest id), not by appearance.
- **Planner:** v3 only, as v2 had no planner. Each student's recorded sitting day is shown and late weeks are marked.
DR-10. **Reproduced fits.** The fitted objects are not in git, so `demo/build_demo_data.py` refits with the frozen hyper-parameters and stops unless the coefficients and the recorded A5 scores are reproduced. It tunes nothing. `demo/data/` (about 15 MB) is generated, not committed.
DR-11. **Tools.** The demo was built and tested in an isolated virtualenv, with nothing installed globally. Playwright Chromium worked, and the three screenshots are in `report/figures/`.

## 2026-11-22 — Part 2: report

DR-12. **Content.** Same title and author block as the midterm. The Results, Discussion and Limitations were written from the three SUMMARY files, corrected by DR-1 to DR-7, because `docs/draft_sections.md` holds only the IEEE template (DR-6). Where a SUMMARY's paste-ready paragraph conflicts with a verified defect, the defect takes precedence, and its recorded numbers are kept.
DR-13. **Numbers.** All table cells and all result numbers in the prose are generated from CSV/JSON cells. `check_numbers.py` re-reads them independently. Other literals are pinned to their exact phrase in `number_allowlist.py`. Audit counts come from `report/audit/`, outside the frozen `results/`.
DR-14. **TeX.** latexmk, pdflatex and tectonic were absent on Windows, WSL and Docker. The official portable Tectonic 0.17.0 was downloaded to the scratchpad, its SHA-256 checked against the GitHub release, and nothing was installed. `[T1]{fontenc}` was added so that IEEEtran gets Times under XeTeX. Result: **8 pages including references and appendix**, no undefined reference or citation, no box more than 5 pt too wide. `overleaf.zip` is built as well, because the machine has no TeX of its own.
DR-15. **References.** Built from the project files only. The midterm's suspect Nagatani author list was shortened to "et al." and flagged. Incomplete entries are listed in `report/README.md`.

## 2026-11-22 — final checks

DR-16. **Housekeeping found in the final checks, not changed.**
- v1's `git add -A` committed 124 MB of `data/processed/sens_*` parquet files, because the ignore pattern did not cover sub-folders. History was not rewritten.
- The v1 torch install upgraded the global setuptools (75.6 to 78.1) without being logged. Not reverted.
DR-17. **Final checks.** 81 tests pass with the demo requirements. A clean clone builds the demo data, starts the app (6-15 s) and regenerates identical tables and pixel-identical vector figures. `git diff v3-revision --stat -- results config` is empty.

## 2026-11-22 - working prototype (requested after the hand-over)

DR-18. **A separate model, not the frozen one.** The v2/v3 coefficients carry DR-1, so `demo/build_prototype.py` fits a new logistic regression per module for the app only.
- Target: submit and pass. Rows: weekly, up to the earlier of submission and due date. Features: events before that day, from the same function that serves the app.
- Trained on presentations before 2014J, C = 1, no tuning. One validation run: AUC 0.78-0.86, against 0.52-0.65 for the calendar alone.
- 2014J outcomes are not used: the model file is byte-identical when all 2014J scores are randomised.
- `results/`, `config/`, `src/` and the report are untouched.
DR-19. **Assumed or rule-based, and labelled so in the app.** Half-life 60 days (adjustable); revise below K = 0.6; urgent within 7 days; chain prerequisites. The tuned v3 weights are not used, because they rank the best-retained topic first (DR-5b). No interest term. The resource-to-topic map uses only cohort clicks before "today" (avoids DR-4).
DR-20. **Leakage tests that would have caught DR-1 and DR-4** are in `tests/test_prototype.py`, and were confirmed to fail on planted versions of both bugs.
