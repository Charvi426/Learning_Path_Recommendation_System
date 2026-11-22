# Results summary

All numbers below come from code that ran (`scripts/01`–`05`, plus the optional `03b` for DKT). They are read
from `results/tables/`. Test = OULAD presentation 2014J of modules DDD and FFF (3,064 students with 14,976
student-topic outcomes; 12,529 decision points from 2,996 students). It was evaluated once with the configuration frozen in
`config/final_config.yaml`. Intervals are 95% bootstrap intervals over students: 1,000 resamples for test
metrics, 200 for model coefficients. "Combined" relevance pools the Rel-A decision points (students who passed
the next assessment) and the Rel-B decision points (students who failed it).

**Data context that shapes every result.** At the pass mark of 40, 94% of submitted topic assessments are passed.
Failure mostly shows up as non-submission, which the methodology excludes from targets by default. Prediction is
therefore about ranking a small minority of fails, and most students' knowledge state is near ceiling.

---

## RQ1 — Does modelling forgetting help?

**Answer: barely for prediction, clearly for spotting what struggling students need, and not on the exam check.**

| Evidence | Without forgetting | With learned forgetting | Difference [95% CI] |
| --- | --- | --- | --- |
| Test AUC, P(pass topic) | 0.900 [0.888, 0.912] (`none`) | 0.902 [0.890, 0.914] (`learned`) | +0.0018 [+0.0002, +0.0037] (paired) |
| Test log-loss | 0.1287 | 0.1276 | −0.0011 [−0.0016, −0.0007] (paired) |
| NDCG@5, combined (A3 vs A5) | 0.828 [0.823, 0.833] | 0.840 [0.835, 0.844] | +0.012, Wilcoxon p_Holm < 1e-4 (2,754 students) |
| NDCG@5, Rel-A (passed) | 0.840 [0.835, 0.845] | 0.838 [0.833, 0.843] | −0.001, p_Holm = 0.97 (n.s.) |
| NDCG@5, Rel-B (failed) | 0.553 [0.530, 0.578] | 0.805 [0.787, 0.821] | +0.251, p_Holm < 1e-4 (424 students) |
| Exam check: Spearman with exam score | mean M: 0.658 | mean K: 0.570 | −0.088 [−0.123, −0.050] |
| Exam check: AUC for exam pass | mean M: 0.849 | mean K: 0.817 | −0.032 [−0.061, −0.006] |

- **Prediction.** The learned-forgetting model is the best knowledge model on test (AUC 0.902, log-loss 0.128),
  but its edge over the same model without decay is about 0.002 AUC. Both clearly beat PFA (learned − PFA =
  +0.012 [+0.008, +0.016] AUC) and the DAS3H window benchmark (+0.014 [+0.007, +0.020]). So the parametric decay
  costs nothing against windows, and DAS3H does not beat the main model by more than 0.01, which means its
  predictions are not used for K (Methodology 5.3).
- **Coefficients** (training fit, 200 student resamples):
  - θ_K = 2.19 [1.87, 2.55]: knowledge carried in from the prerequisite helps, as expected.
  - Half-life at average inputs: 2^φ₀ = 2^6.52 ≈ 92 days [50, 350]. The median fitted h over training pairs is 88 days (`results/figures/decay_curves.png`).
  - Forgetting is slow relative to the 20–50-day gaps between assessments, which is why prediction gains are small.
  - Validation picked S = 120 days for the fixed decay, at the edge of the grid, which again points to slow forgetting.
- **Ranking.** Forgetting-aware K changes the ranking mainly for students who then fail: A5 recommends the failed
  topic and its prerequisites much more often than A3 (Rel-B NDCG@5 0.805 vs 0.553). Among students who pass,
  there is no difference.
- **Exam check (7.4)** is the strongest test the methodology names, and it goes the wrong way. Decayed knowledge K
  predicts the cumulative exam *worse* than undecayed mastery M, for every decay variant (fixed −0.152, learned
  −0.088, learned+interest −0.096 in Spearman ρ; all intervals exclude 0). Validation shows the same pattern
  (`exam_retention_check_val.csv`). The learned decay hurts least, but the evidence that the modelled decay is
  "real" retention loss at exam time is **negative**.

- **Robustness.** A5 > A3 also holds with pass mark 70 (+0.013) and with non-submission counted as a fail (+0.020), but not with data-derived prerequisites (see Sensitivity).

**Caveats.**
- Δ varies enough to identify h: the median within-topic IQR is 12 days and 66% of prerequisites are revised after submission (`delta_identification.json`). But for target topic 2 the prerequisite is topic 1, whose course-wide resources are clicked almost daily, so Δ ≈ 1–2 days there.
- φ_c (−1.08) and φ_d (+1.28) have opposite signs because clicks and active days are collinear; only their combination is interpretable.
- The gain on Rel-B rests on 424 students who failed a submitted assessment.
- Popularity (A1) beats every model-based configuration on Rel-A and on combined relevance (see RQ2 and the limitations below).

## RQ2 — Learned vs. fixed decay?

**Answer: the learned half-life predicts slightly better, but the fixed decay gives slightly better paths. Neither difference is practically large.**

| Evidence | Fixed S (A4 / `fixed`) | Learned h (A5 / `learned`) | Difference [95% CI] |
| --- | --- | --- | --- |
| Test AUC | 0.900 [0.888, 0.911] | 0.902 [0.890, 0.914] | +0.0023 [+0.0014, +0.0033] (paired) |
| Test log-loss | 0.1291 | 0.1276 | −0.0014 [−0.0017, −0.0012] |
| NDCG@5, combined | 0.843 [0.839, 0.848] | 0.840 [0.835, 0.844] | −0.004, Wilcoxon p_Holm < 1e-4 |
| NDCG@5, Rel-B | 0.824 [0.808, 0.839] | 0.805 [0.787, 0.821] | −0.019, p_Holm < 1e-4 |
| Exam check, ρ(K) − ρ(M) | −0.152 [−0.192, −0.104] | −0.088 [−0.123, −0.050] | learned is less wrong |

- The fixed S = 120 days corresponds to a half-life of 83 days, almost the same as the learned median of 88 days.
  The learned model adds per-student/topic variation in h (10th–90th percentile 50–171 days).
- That variation slightly improves calibrated prediction and the exam correlation, but not the offline ranking.
  The Wilcoxon tests are significant because 2,754 paired students give high power to differences of 0.004 NDCG.

## RQ3 — Does interest help?

**Answer: the interest terms are statistically detectable in the model, but they do not improve prediction or ranking.**

| Evidence | Value [95% CI] |
| --- | --- |
| φ_I (interest → longer half-life) | +0.566 [+0.058, +0.966]: excludes 0 |
| θ_I (direct effect on passing) | +0.144 [+0.059, +0.237]: excludes 0 |
| θ_Ic (interest × clicks, "learns faster") | +0.035 [−0.013, +0.098]: includes 0 |
| Test AUC, learned+interest − learned | −0.0020 [−0.0035, −0.0005] (paired) |
| NDCG@5 combined, A6 − A5 | 0.837 vs 0.840: −0.003, Wilcoxon p_Holm < 1e-4 |
| NDCG@5 Rel-B, A6 − A5 | 0.802 vs 0.805: −0.002, p_Holm = 0.49 (n.s.) |
| Tuned interest weight ε | 0 (validation chose no interest term in the score) |

- Of the methodology's two hypotheses, "retains longer" (φ_I > 0) is supported, and "learns faster" (θ_Ic > 0) is not.
- With three extra parameters (θ_I, θ_Ic, φ_I), the interest model is slightly *worse* out of sample. The A6 path
  differs from A5 only through the interest model's K and M (ε = 0), and it is slightly worse too.
- **Proxy validity** (`interest_validity.csv`, training split):
  - I is nearly independent of activity: r = 0.03 with the student's total clicks and 0.23 with clicks on the same topic (target |r| < 0.3), so the 8.2 fallback was not needed.
  - Split-half reliability (odd vs even days) r = 0.49, Spearman–Brown 0.66.
  - The three raw signals correlate positively within (module, topic, stage) (+0.53, +0.56, +0.20). Pooled, early correlates negatively with vol and post (−0.16, −0.17) because of the course calendar.
  - Following the methodology, it should be called a **behavioural interest proxy**.
- **Fairness** (`fairness_gaps.csv`): the largest gap in NDCG@5 between groups is 0.029 by gender for A6 vs 0.015 for A3, and 0.029 vs 0.028 by IMD band. So the interest proxy roughly doubles the (small) gender gap. AUC gaps by IMD band are 0.070 (`learnedI`) vs 0.062 (`none`).

---

## Other results

**Knowledge models on test** (`model_metrics_test.csv`, `calibration_test.csv`, `results/figures/model_auc_test.png`,
`calibration_test.png`).

| Model | AUC [95% CI] | Log-loss | Brier | RMSE | ECE |
| --- | --- | --- | --- | --- | --- |
| Majority | 0.500 | 0.186 | 0.043 | 0.208 | 0.018 |
| Topic difficulty (IRT-like) | 0.848 [0.833, 0.863] | 0.138 | 0.034 | 0.185 | 0.016 |
| PFA | 0.890 [0.876, 0.903] | 0.131 | 0.033 | 0.183 | 0.025 |
| Forgetting-aware PFA, no forgetting | 0.900 [0.888, 0.912] | 0.129 | 0.033 | 0.183 | 0.021 |
| — fixed S = 120 d | 0.900 [0.888, 0.911] | 0.129 | 0.034 | 0.183 | 0.022 |
| — learned half-life | **0.902** [0.890, 0.914] | **0.128** | 0.033 | 0.182 | 0.020 |
| — learned half-life + interest (full 5.1) | 0.900 [0.888, 0.912] | 0.129 | 0.033 | 0.183 | 0.021 |
| DAS3H windows {7,14,30,∞} | 0.889 [0.875, 0.902] | 0.133 | 0.033 | 0.183 | 0.026 |
| DKT (pyKT), rows with history only* | 0.783 [0.764, 0.801] | 0.171 | 0.044 | 0.209 | – |

\*DKT cannot predict a sequence's first step. On the same 11,912 rows: learned 0.898 [0.886, 0.910], PFA 0.882
(`dkt_comparison.csv`). As Gervet et al. (2020) lead one to expect for sequences of at most 6 steps, DKT does not
reach PFA.

Calibration: all models under-predict in the 0.1–0.5 bins, because the 2014J cohort passed more often than the
training cohorts. With ECE ≈ 0.02, M and K are usable as probabilities.

**Ranking metrics, test, K ∈ {3, 5}** (`ranking_metrics_test.csv`, `ranking_metrics_test_wide.csv`,
`results/figures/ranking_ndcg5_test.png`). NDCG@5 [95% CI]:

| ID | Configuration | Combined | Rel-A | Rel-B | NDCG@3 comb. | P@5 comb. | R@5 comb. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | Random | 0.739 [0.734, 0.744] | 0.737 | 0.726 | 0.607 | 0.494 | 0.904 |
| A1 | Popularity | **0.946** [0.944, 0.949] | **0.952** | **0.824** | **0.934** | 0.520 | 0.956 |
| A2 | Mastery only | 0.653 [0.648, 0.658] | 0.648 | 0.698 | 0.472 | 0.492 | 0.911 |
| A3 | Mastery + prerequisites | 0.828 [0.823, 0.833] | 0.840 | 0.553 | 0.765 | 0.474 | 0.884 |
| A4 | + fixed forgetting | 0.843 [0.839, 0.848] | 0.841 | **0.824** | 0.827 | 0.501 | 0.928 |
| A5 | + learned forgetting (proposed) | 0.840 [0.835, 0.844] | 0.838 | 0.805 | 0.817 | 0.500 | 0.926 |
| A6 | + interest | 0.837 [0.833, 0.842] | 0.836 | 0.802 | 0.812 | 0.499 | 0.925 |
| A7 | A6 + data-derived prerequisites | = A6 (no data edge qualifies at pass mark 40) | | | | | |

Full P@K / R@K / NDCG@K for every configuration and relevance set are in the CSV. **Popularity is the risk
named in 8.2 and it materialised.** A1 ranks topics by how much training students clicked them in the same
window, and Rel-A rewards exactly that, so A1 wins on Rel-A by 0.11 NDCG. Following the 8.2 fallback, the report
should lead with Rel-B, the simulated gain and the exam check. On Rel-B, A4 ties A1 (0.824 vs 0.824), and A5/A6
are just below.

**Simulated learning gain (model-based, DAS3H simulator)** (`simulated_gain_test.csv`). These are mean predicted
gains in P(pass next assessment) over doing nothing:
- A0 0.010, A1 0.012, A2 0.012;
- A3–A7 0.0009 [0.0008, 0.0010].

The gain comes almost entirely from paths that contain the upcoming topic for students with a low predicted pass
probability. The readiness filter Q >= τ_q (A3–A7) drops the upcoming topic for students whose prerequisite
knowledge is low, so those students lose exactly the item the simulator rewards. This is a limitation of the
readiness filter, not of forgetting: A5 − A3 = −0.00002, statistically significant (p_Holm < 1e-16) but negligible (`wilcoxon_holm_simulated_gain.csv`).

**Statistics** (`wilcoxon_holm.csv`). Primary family: per-student NDCG@5 (combined), Holm-corrected.
- A5 > A3: +0.0118, p_Holm < 1e-4.
- A5 < A4: −0.0036, p_Holm < 1e-4.
- A6 < A5: −0.0028, p_Holm < 1e-4.

All three are significant, but only A5 vs A3 is in the hypothesised direction. On Rel-A alone, A5 vs A3 is n.s.
(p_Holm = 0.97).

**Sensitivity** (`sensitivity.csv`; each variant retuned on its own validation split, then evaluated once on test):

| Setting | Comparison | x (NDCG@5 combined) | y | x − y | Wilcoxon p | Does A5 > A3 hold? |
| --- | --- | --- | --- | --- | --- | --- |
| Default (pass 40, non-submission excluded, chain) | A5 vs A3 | 0.840 [0.835, 0.844] | 0.828 [0.823, 0.833] | +0.012 | < 1e-4 | **yes** |
| Pass mark 70 | A5 vs A3 | 0.863 [0.859, 0.867] | 0.850 [0.846, 0.854] | +0.013 | < 1e-4 | **yes** |
| Non-submission = fail | A5 vs A3 | 0.802 [0.797, 0.808] | 0.782 [0.777, 0.787] | +0.020 | < 1e-4 | **yes** |
| Pass 70, chain + data-derived prerequisites | A5 vs A3 (data graph) | 0.753 [0.747, 0.758] | 0.844 [0.839, 0.848] | −0.091 | < 1e-4 | **no** |
| Non-submission = fail, chain + data prerequisites | A5 vs A3 (data graph) | 0.720 [0.714, 0.725] | 0.773 [0.768, 0.779] | −0.054 | < 1e-4 | **no** |
| Windows {7,30,∞} vs {7,14,30,∞} (simulator only) | A5 − A3 simulated gain | 0.0009 | 0.0009 | ≈ −0.00002 under both | < 1e-4 | no difference; simulator AUC 0.889 under both |

- **The forgetting conclusion (A5 > A3 with the chain graph) is robust** to the pass mark and to how non-submission is treated. The gain is largest when non-submission counts as a fail (+0.020).
- **Data-derived prerequisites hurt.** In the variants where edges qualify (pass 70: 30 module-level pairs, i.e. every ordered pair of topics; non-submission = fail: 26), the data graph is almost fully connected. Readiness Q then becomes the minimum over all earlier topics, and both configurations drop. A5 falls below A3 because decayed K for old topics now enters Q for every later topic. **Chain prerequisites should be preferred**, so A7-style data edges are not recommended.
- Window choice has no effect on the A5 vs A3 conclusion; the two DAS3H window sets are almost identical in AUC (0.8885 vs 0.8885).

**Fairness** (`fairness.csv`, `fairness_gaps.csv`). Largest gap between groups (groups with >= 30 students):

| Attribute | NDCG@5 A3 | NDCG@5 A6 | AUC `none` (A3 model) | AUC `learnedI` (A6 model) |
| --- | --- | --- | --- | --- |
| Gender | 0.015 | 0.029 | 0.005 | 0.004 |
| IMD band | 0.028 | 0.029 | 0.062 | 0.070 |
| Disability | 0.011 | 0.008 | 0.014 | 0.017 |

**Sample paths** with explanations (10 test students: 3 strong, 4 average, 3 struggling):
`results/sample_paths/sample_paths_A6.md` and `.json`. The same students under A5 are in `sample_paths_A5.*`. With
the tuned weights (G 0.3, Q 0.1, D 0.6), most explanations come from the difficulty-fit term D.

---

## What was not implemented, or implemented differently (Methodology re-read section by section)

| § | Methodology | What was done | Why |
| --- | --- | --- | --- |
| 2 | root folder `fds-learning-path/`; notebooks hand-written | the project root is the working folder; notebooks are generated and executed by `notebooks/make_notebooks.py` | reproducibility |
| 2 | `features_{split}.parquet` holds all factors | factors are stored in long form (`state_{split}.parquet`, one row per query × topic); `features_{split}` is a wide copy for inspection | the forgetting model needs one row per prerequisite pair |
| 2 | `report/` | left empty | final report source is out of scope for this run |
| 3.2.1 | keep modules with >= 5 non-exam assessments | DDD and FFF only | CCC has no training presentation; EEE has 4 assessments; GGG has 4 topics after merging; AAA has no 2014B; BBB 2014J has a different structure (5 vs 7 topics) |
| 3.2 | – | DDD 2013B CMAs dropped; `homepage` sites not mapped to topics | topic alignment across presentations; navigation has no topic content |
| 3.2.6 | clicks before day 0 → first topic | topic 1's window starts at the first VLE day; mapping is per site, not per click | the same effect for in-window clicks; per-site mapping is needed for "early"/"post" signals |
| 3.3 | cohort-click mapping | uses the same presentation's cohort clicks (no outcomes) | treated as course metadata (`DECISIONS.md` #12) |
| 4.4 step 5 | cohort z-score per (module, presentation, topic) from train | per (module, topic, stage) from train | val/test presentations have no training statistics of their own; the share depends mechanically on the number of open topics |
| 4.5 | data-derived edges | rule implemented exactly; **no edge qualifies at pass 40** (max 180 failing students < 200) | data; A7 = A6. Chain vs data compared in the pass-70 / non-submission sensitivity runs, where 30 / 26 edges qualify |
| 4.6 | difficulty feeds the topic intercept's prior | β_k is fitted unpenalised; diff_k is stored in the knowledge state | the intercepts are well identified (> 1,900 training rows per topic) |
| 4.2, 4.3 | Δt_{s,k}, prev_attempts | computed (`dt_last`, `prev_attempts`) but not used by any model | the 5.1 equation does not include them |
| 5.1 / 5.2 | "main" model naming is inconsistent | `learned` (φ₀, φ_c, φ_d) = A5; `learnedI` (+φ_I, θ_I, θ_Ic) = A6 | both are reported |
| 5.1 | tune L2 on validation | grid extended to 1e-5 after the first fit chose the edge | before any test use |
| 5.4 | DKT via pyKT "default configuration" | pyKT `DKT` class with its defaults, own Adam loop with early stopping; evaluated on rows with history | pyKT's training scripts need wandb and its benchmark folders |
| 6.1 | thresholds tuned over {0.4 … 0.8}, p* over {0.6, 0.7, 0.8} | grid extended to {…, 0.9, 1.0} and p* 0.9 | every tuned value hit the edge; with a 94% pass rate, τ_m <= 0.8 removed the upcoming topic from almost every path (`DECISIONS.md` #42). Validation then chose τ_m = τ_r = 1.0 (filters off) |
| 6.2 | F term | implemented; validation gave γ = 0 in every configuration | data |
| 7.1 | RMSE/MAE "for ỹ in the score variant" | no separate score model; predicted p compared with ỹ (`RMSE_score`) | simplicity |
| 7.2 | graded relevance "a topic in both sets counts 2" | cannot occur: Rel-A and Rel-B are defined on disjoint students | definitions as written |
| 7.5 | A0, A1, A2 | A0/A1 are pure rankers over all eligible topics (no revision cap); A2 has no Q filter and no ordering | "no components" means none of the path machinery |
| 7.6 | window sensitivity | windows only enter the DAS3H model; reported as DAS3H AUC and as the A5−A3 simulated gain under each simulator | A3/A5 do not use windows |
| 7.1 | paired comparison of models | added a paired student-bootstrap of AUC/log-loss differences (`model_paired_diffs_test.csv`) after the test run | the marginal CIs overlap; a descriptive statistic only, nothing was re-tuned |

## Limitations to state in the report

- Topics are assessment windows; prerequisites are course order. Topic 1 absorbs course-wide resources.
- With 94% passing, "mastery" is near ceiling for most students, and the informative failures are non-submissions (see the sensitivity rows).
- Offline relevance is a proxy, and Rel-A favours popularity. The simulated gain is model-based (DAS3H simulator) and blind to items other than the upcoming topic and its prerequisites.
- The exam check does not support the decay as modelled. A real effect of forgetting would need a live study or finer-grained (per-question) data.
- Interest is behavioural and inferred from clicks; it cannot be separated from motivation or deadlines.

## Quality checks

- `python -m pytest -q`: 18 tests pass (topic construction; no-leakage perturbation and truncation tests; cohort statistics from train only; metric implementations including hand-computed NDCG@3/5 cases; the 671-weight grid; revision cap and prerequisite insertion in paths).
- **Clean-clone reproduction.** Cloned the repo, copied the 7 raw CSVs into `data/raw/`, ran `scripts/01`–`05` (about 25 min). All 39 CSVs in `results/tables/` were reproduced to within 1e-6, and git saw no content change in any of them. The only differences were the `frozen_at` timestamp in `config/final_config.yaml` and run-time fields in `models/*train_summary.json`. The DKT tables come from the optional `scripts/03b_optional_dkt.py` and were not part of that rerun.
- `Methodology.md` was re-read section by section; differences are listed in the table above and, with dates, in `DECISIONS.md`.
