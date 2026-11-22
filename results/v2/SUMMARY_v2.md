# v2 summary (post-hoc fixes)

Branch `v2-fixes`, built on v1 (tag `v1`, branch `master`). **Everything in this document is post-hoc**: v1's test
split (DDD/FFF 2014J) had already been scored when this work started. To keep the new evidence honest:
- every new choice was tuned on **validation only** (DDD/FFF 2014B) and frozen in `config/final_config_v2.yaml`;
- the **primary new evidence comes from five holdout modules never used in v1** (AAA, BBB, CCC, EEE, GGG; 2014J of each);
- DDD/FFF 2014J was scored **once** with the frozen v2 config, and those numbers are labelled **post-hoc / exploratory** throughout.

v1 outputs are unchanged: `git diff master --stat -- results/tables config/final_config.yaml` is empty. Every
change is logged in `DECISIONS.md` under "v2 (post-hoc)", tagged BUG-FIX or CHANGE. **No BUG-FIX was needed**: all
four fixes are CHANGEs or audits. Intervals are 95% bootstrap intervals over students (1,000 resamples). "Combined"
relevance pools Rel-A and Rel-B decision points, as in v1.

---

## The four fixes

### Fix 1 — exam retention check: audit + diagnosis → no bug

- **Validation shows the same sign as test.** ρ(mean K, exam) − ρ(mean M, exam):
  - learned decay −0.097 [−0.142, −0.054] (test −0.088);
  - fixed decay −0.139 (test −0.152).
- **Audit.** Δ at the exam date was recomputed independently for all 2,930 validation (student, topic) pairs and matches the pipeline exactly. Revision-period clicks do reset Δ, and nothing on or after the exam date is used. 5 new unit tests cover this (`tests/test_v2_exam_delta.py`). **No BUG-FIX.**
- **Scope v1 missed.** OULAD has exam scores only for CCC and DDD, so the v1 exam check was DDD-only.
- **Diagnosis** (`exam_check_diagnosis.md`):
  - Topics 2–5 drive the negative sign. Only 20–34% of exam-takers re-open them with >= 3 clicks in the revision period, so Δ is 83–188 days and R ≈ 0.5–0.6.
  - R carries a much weaker exam signal than the score: ρ(R, exam) = 0.11–0.22 vs ρ(M, exam) = 0.46–0.63 for those topics.
  - 92% of exam-takers revise through topic-1 (course-wide) and topic-6 resources, which does not reset Δ for topics 2–5.
  - So the decay as modelled measures "did the student re-open this window's pages", which misses most revision.
- **Holdout replicates it.** In CCC (the only holdout module with exam scores), Δρ = −0.096 [−0.113, −0.078] for the learned decay and −0.298 [−0.337, −0.262] for fixed.

### Fix 2 — relative candidate filters → CHANGE; the revision filter works, the mastery filter cannot

- Both options were implemented: within-student bottom-q% and training-cohort q-quantile, with q ∈ {10, …, 50}, for revision (low K) and new topics (top q% predicted mastery excluded as already mastered). Admissibility rule: the mastery filter may not remove the upcoming topic from > 10% of decision points, and no filter may be inactive (< 1% of decision points).
- **Validation chose within-student, q_rev = 50, q_new = 30** for every configuration.
- **Every cohort-quantile setting was inadmissible**: it removed the upcoming topic at 22–77% of decision points.

| Validation (A5) | v1 (absolute τ, filters off) | v2 (relative) |
| --- | --- | --- |
| Candidates per decision point | 4.89 | 3.81 |
| Revision candidates per decision point | 2.73 (every assessed topic) | 1.66 |
| Paths with >= 1 Revise item | 100% | 100% |
| Paths containing the upcoming topic | 93.1% | 93.0% |
| Upcoming topic removed by mastery filter | 0% | 0% |
| Decision points where the mastery filter excludes anything | 0% | 1.4% |

- **The revision filter now does real work** (39% fewer revision candidates; mean path length 4.30 → 3.81).
- **No relative mastery filter can do real work without dropping the upcoming topic.** That topic usually has the highest predicted mastery among a student's unassessed topics, because they have already been working on it. The admissible setting is nearly inactive.
- **Effect.** The filters shorten paths and **lower NDCG@5** against v1's filters-off setting:
  - validation: A5 0.852 → 0.778; A3 0.839 → 0.773;
  - holdout: v2 vs v1 A5 −0.011 (p_Holm < 1e-4), A3 +0.006 (n.s.).

### Fix 3 — popularity: A1 audit → no leakage; Pop term → CHANGE

- **A1 audit: no leakage, so no BUG-FIX.** v1's A1 scores topics with the clicks of *training-presentation* students in the same decision window (past cohorts). It never reads the scored presentation. Corrected A1 = v1 A1 (test 0.946, unchanged). A1 is strong because Rel-A is defined by engagement in the window, which cohort popularity predicts.
- **Pop(k, t) chosen on validation** (Pop-only NDCG@5): training-window share 0.945, versus the current cohort's clicks in the last 7/14/30 days before t at 0.889/0.889/0.887. Leakage tests are in `tests/test_v2_pop_and_filters.py`.
- **Six-term score.** Validation picked G 0.3, F 0.2, D 0.3, Pop 0.2 (ε = 0) for both A8 and A9. With popularity in the score, the forgetting term F gets weight for the first time.
- **Effect.** Adding Pop helps a little, but the model-based recommender stays far below pure popularity:
  - A8 − A5: +0.008 validation, +0.010 holdout;
  - A8 − A1: −0.159 validation, −0.165 holdout (p_Holm < 1e-4).

### Fix 4 — holdout modules → CHANGE (new primary evidence)

- All 5 excluded modules qualify (>= 4 topics per presentation, >= 2 presentations): AAA (5/5 topics), BBB (7/7/7/5), CCC (8/8), EEE (4/4/4), GGG (4/4/4).
- Each was trained on its earlier presentations and tested on 2014J, with coefficients refitted per module and every hyper-parameter, threshold and weight frozen.
- Size: 5,465 test students with outcomes; 24,122 topic outcomes; 19,764 decision points; 4,526 students with a relevance set.
- Results below and in `results/v2/holdout/`.

---

## v1 vs v2: one comparison table

NDCG@5 [95% CI], combined relevance (Rel-A / Rel-B point estimates in brackets after the CI).
Validation = DDD/FFF 2014B. **Holdout = AAA, BBB, CCC, EEE, GGG 2014J (primary).** 2014J = DDD/FFF test, **v2 rows are
post-hoc / exploratory**.

| Config | Validation | Holdout (primary) | DDD/FFF 2014J |
| --- | --- | --- | --- |
| A1 popularity (v1 = v2) | 0.945 [0.942, 0.948] (A 0.953 / B 0.849) | **0.893** [0.889, 0.896] (0.893 / 0.879) | 0.946 [0.944, 0.949] (0.952 / 0.824) |
| v1 A3 | 0.839 [0.833, 0.845] (0.852 / 0.612) | 0.702 [0.696, 0.708] (0.700 / 0.635) | 0.828 [0.823, 0.833] (0.840 / 0.553) |
| v1 A5 | 0.852 [0.847, 0.857] (0.850 / 0.813) | 0.729 [0.724, 0.733] (0.722 / 0.768) | 0.840 [0.835, 0.844] (0.838 / 0.805) |
| v1 A6 | 0.848 [0.843, 0.854] (0.847 / 0.802) | 0.745 [0.740, 0.749] (0.738 / 0.775) | 0.837 [0.833, 0.842] (0.836 / 0.802) |
| v2 A3 | 0.773 [0.767, 0.780] (0.764 / 0.794) | 0.708 [0.703, 0.713] (0.704 / 0.720) | *0.784 [0.780, 0.789] (0.783 / 0.756)* |
| v2 A5 | 0.778 [0.771, 0.785] (0.774 / 0.757) | 0.718 [0.712, 0.723] (0.714 / 0.717) | *0.784 [0.779, 0.789] (0.784 / 0.723)* |
| v2 A6 | 0.781 [0.775, 0.788] (0.776 / 0.761) | 0.735 [0.730, 0.739] (0.731 / 0.725) | *0.786 [0.782, 0.791] (0.786 / 0.718)* |
| v2 A8 = A5 + Pop | 0.786 [0.780, 0.793] (0.782 / 0.760) | 0.728 [0.723, 0.733] (0.724 / 0.731) | *0.794 [0.790, 0.799] (0.795 / 0.723)* |
| v2 A9 = A6 + Pop | 0.790 [0.784, 0.796] (0.785 / 0.765) | **0.742** [0.737, 0.746] (0.738 / 0.734) | *0.797 [0.792, 0.802] (0.798 / 0.721)* |

*Italic = post-hoc / exploratory.* NDCG@3, P@K and R@K for K ∈ {3, 5}: `results/v2/validation/eval/ranking_metrics.csv`,
`results/v2/holdout/ranking_metrics_holdout.csv` (pooled and per module), and `results/v2/posthoc_2014J/ranking_metrics.csv`.

**Paired Wilcoxon, Holm-corrected, per-student NDCG@5 (combined):**

| Comparison | Validation | Holdout (primary) | 2014J (post-hoc) |
| --- | --- | --- | --- |
| v2 A5 vs A3 (forgetting) | +0.005, p_Holm = 2e-4 | **+0.009**, p_Holm < 1e-4 | +0.000, p_Holm = 0.31 |
| v2 A8 vs A1 (Pop term vs popularity) | −0.159, p < 1e-4 | **−0.165**, p < 1e-4 | −0.152, p < 1e-4 |
| v2 A9 vs A8 (interest) | +0.003, p < 1e-4 | **+0.014**, p < 1e-4 | +0.003, p < 1e-4 |
| v2 A5 vs A3, Rel-B only | −0.037, p < 1e-4 | −0.004, p_Holm = 0.95 | −0.033, p_Holm = 7e-4 |

**Knowledge models on the holdout** (`model_metrics_holdout.csv`, `model_paired_diffs_holdout.csv`, per module in
`model_metrics_holdout_by_module.csv`). Pooled AUC [95% CI] over 24,122 outcomes:

| | Holdout AUC | Log-loss | DDD/FFF test AUC (v1) |
| --- | --- | --- | --- |
| Topic difficulty | 0.821 [0.809, 0.833] | 0.197 | 0.848 |
| PFA | 0.854 [0.843, 0.865] | 0.184 | 0.890 |
| No forgetting | 0.848 [0.838, 0.857] | 0.202 | 0.900 |
| Fixed S | 0.847 [0.837, 0.857] | 0.203 | 0.900 |
| Learned half-life | 0.847 [0.837, 0.857] | 0.203 | **0.902** |
| Learned + interest | 0.847 [0.837, 0.857] | 0.201 | 0.900 |
| DAS3H windows | **0.858** [0.846, 0.869] | **0.179** | 0.889 |

- Paired differences:
  - learned − none: −0.0004 [−0.0011, +0.0002];
  - learned − fixed: +0.0004 [−0.0004, +0.0010];
  - learnedI − learned: +0.0001 [−0.0009, +0.0011];
  - learned − DAS3H: −0.011 [−0.014, −0.007].
- **On the holdout, DAS3H beats the main model by more than 0.01 AUC.** Following Methodology 5.3 this must be reported.
- Pooled AUCs mix modules with different base rates (the constant model gets 0.46). Per module, the forgetting-aware model is best in AAA, CCC, EEE and GGG, and DAS3H/PFA in BBB (0.892/0.882 vs 0.835).

**Simulated gain** (DAS3H simulator refitted per module):
- holdout: every model-based configuration, v1 and v2, gives about +0.030 (e.g. v2 A9 0.0297 [0.0288, 0.0307]), against popularity +0.041;
- DDD/FFF validation and 2014J: unchanged from v1 (A3–A9 ≈ +0.001).

**Exam check (holdout, CCC only):**
- learned decay: ρ(K) − ρ(M) = −0.096 [−0.113, −0.078]; AUC(K) − AUC(M) = −0.020 [−0.030, −0.009];
- fixed decay: −0.298 [−0.337, −0.262].

---

## Updated answers to the research questions

**RQ1 — Does forgetting help?** **Not for prediction, inconsistently for ranking, and not for the exam check.**
- **Prediction.** On fresh modules the learned decay adds nothing over no decay (−0.0004 AUC [−0.0011, +0.0002]). The +0.002 seen on DDD/FFF test does not replicate.
- **Ranking.** Forgetting-aware ranking helps on the holdout:
  - v2 A5 vs A3: +0.009 NDCG@5, p_Holm < 1e-4;
  - v1 A5 vs v1 A3 on the holdout: 0.729 vs 0.702.
  But the gain comes from students who passed (Rel-A). For students who failed (Rel-B) it is −0.004 (n.s.) on the holdout, and negative on validation and 2014J in v2. That reverses v1's main ranking finding: v1's large Rel-B gain (A3 0.553 → A5 0.805 on test) was measured against a weak v1 A3. Validation had tuned that A3 to rank by readiness Q alone, with filters off, so most ties fell back to topic order. With relative filters, v2's A3 re-tunes to G 0.7 / D 0.3 and reaches Rel-B 0.794 on validation, and A5's Rel-B advantage disappears.
- **Exam check.** Negative again on a new module (CCC). The diagnosis shows the decay mostly measures whether a student re-opened an old window's pages, while most revision happens elsewhere.

**RQ2 — Learned vs. fixed decay?** The learned half-life is consistently **less wrong** than the fixed decay on the exam check (CCC: −0.096 vs −0.298; DDD: −0.088 vs −0.152). It is indistinguishable for prediction on the holdout (+0.0004 AUC, CI includes 0). Ranking comparisons involving fixed decay (A4) were not part of the holdout set, so this answer rests on prediction and the exam check.

**RQ3 — Does interest help?** **Weakly, for ranking on fresh data; not for prediction.**
- On the holdout the interest variant ranks better: v2 A9 vs A8 +0.014, p_Holm < 1e-4 (v2 A6 0.735 vs A5 0.718; v1 A6 0.745 vs A5 0.729). Validation and 2014J show the same direction (+0.003).
- The interest model adds no AUC on the holdout (+0.0001 [−0.0009, +0.0011]).
- Because ε = 0 in every tuned configuration, the ranking gain comes from the interest model's K/M (φ_I, θ_I), not from the I term in the score.

---

## What changed in the story since v1 (paste-ready)

The v1 evaluation suggested that modelling forgetting helped both prediction and, especially, recommendations for
struggling students. A post-hoc review weakens both claims; it was designed to avoid tuning on the v1 test data and
added five modules never used before as a fresh test set.

On those modules, the learned half-life adds no predictive accuracy over the same model without decay, and a
DAS3H-style window model now predicts best. The forgetting-aware recommender still ranks slightly better than its
no-forgetting counterpart (+0.009 NDCG@5). However, the large gain v1 reported for students who failed their next
assessment was measured against a weak baseline: validation had tuned it to rank by prerequisite readiness alone,
with filters off. Once revision candidates are restricted to each student's weakest topics and the baseline is
re-tuned, the gain disappears.

The exam-based retention check is negative on a second module as well. An audit found no implementation error.
Instead, with assessment-window topics, Δ counts only clicks on a window's own pages, and most students revise
through course-wide and final-window resources.

Interest-aware knowledge estimates give a small, replicated ranking gain. A leakage-free popularity term helps a
little, but a pure training-cohort popularity ranking remains far ahead of every model-based configuration on
these offline relevance definitions. An audit confirmed that this popularity baseline uses no future information.
The honest conclusion is that, on OULAD's assessment-level data, forgetting is hard to observe. Offline relevance
mostly rewards what cohorts typically do next, so a live study would be needed to test the recommender's value.

---

## Reproducibility

- **Setup.** Fresh `git clone -b v2-fixes` into a scratch folder, the 7 raw CSVs copied in, then:
  - v1 `scripts/01–03` to regenerate the gitignored intermediate data (v1 `final_config.yaml` and the knowledge models' inputs);
  - `scripts/v2/fix1_exam_diagnosis.py`;
  - `fix23_tune_validation.py --part fix2 / fix3 / freeze`;
  - `fix4_holdout.py`;
  - `eval_dddfff.py --split val` and `--split test`.
  It took about 25 minutes.
- **Result:** all **81 CSV tables in `results/v2/` match** (max |diff| = 0, well within the 1e-6 tolerance, text columns identical), and all **21 JSON files** are identical. All 102 files were rewritten during the run, none left over from the clone.
- The only differences git saw in the clone are the `frozen_at` timestamp in `config/final_config_v2.yaml` and the run time in `models/train_summary.json`.
- **v1 unchanged:** `git diff master --stat -- results/tables config/final_config.yaml` prints nothing. Neither do `results/sample_paths`, `results/SUMMARY.md`, the v1 sensitivity configs or `models/`.

## Tests

`python -m pytest -q`: **30 passed**. That is v1's 18 plus 12 new:
- `test_v2_exam_delta.py` (5): Δ reset at the exam date;
- `test_v2_pop_and_filters.py` (7): Pop uses no events at/after t, Pop matches a manual window sum, train-window Pop reads only training data, within-student and cohort-quantile filters, the 6-term grid (2,211 / 1,001 vectors) and the admissibility rule.

## Files

| Path | Content |
| --- | --- |
| `config/final_config_v2.yaml` | frozen v2 choices (validation only) |
| `results/v2/exam_check_diagnosis.md`, `results/v2/exam_check/` | Fix 1 |
| `results/v2/validation/` | Fix 2/3 tuning (`tuned_A*.json`, histories, Pop selection), `eval/` = v1 vs v2 on validation |
| `results/v2/holdout/` | Fix 4: pooled and per-module tables, Wilcoxon, simulated gain, exam check (CCC), eligibility |
| `results/v2/posthoc_2014J/` | DDD/FFF 2014J, scored once, **post-hoc / exploratory** |
| `scripts/v2/` | `fix1_exam_diagnosis.py`, `fix23_tune_validation.py`, `fix4_holdout.py`, `eval_dddfff.py` |
