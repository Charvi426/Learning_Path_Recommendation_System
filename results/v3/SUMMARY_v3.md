# v3 summary: making revision visible to the forgetting model (post-hoc)

Branch `v3-revision`, built on `v2-fixes`. **All of v3 is post-hoc.** The CCC and DDD exams, DDD/FFF 2014J and the
holdout modules had been looked at before. Everything was designed and tuned on DDD/FFF validation (2014B, including
the DDD 2014B exam), then **preregistered** (`PREREGISTRATION_v3.md`, commit 9c44407) before anything else was
scored:
- **primary evidence** comes from the five holdout modules (AAA, BBB, CCC, EEE, GGG, 2014J);
- the CCC exam check and DDD/FFF 2014J are **exploratory**, scored once.

Intervals are 95% bootstrap intervals over students (1,000 resamples). Decisions are logged in `DECISIONS.md` under
"v3 (post-hoc)". No BUG-FIX was needed; all entries are CHANGEs or corrections to my own tuning procedure, made
before preregistration.

## What changed

1. **Navigation vs content** (`resource_classes.csv`). Homepages, plus forums, subpages and glossaries that are
   clicked evenly all course, are navigation (32–61% of each module's clicks) and never count as practice.
2. **Attribution of practice to topics.** A content site's clicks are credited to topics with weights estimated from
   pre-assessment cohort clicks. Spread-out, final-window and exam-preparation resources are shared over all topics
   assessed so far. Effect on DDD validation exam-takers: 55–60% now show a revision-period dose on each of topics
   2–5, against 15–47% under the v1/v2 mapping (`attribution_revision_diagnostics_val.csv`).
3. **History-based memory** (ACT-R activation, Pavlik & Anderson 2005): m = ln Σ x_i (t − t_i + 1)^(−d_i), with
   spacing-sensitive decay d_i = c·exp(m_i) + a. It replaces Δ-based retention, and the multiplicative K = M·R is
   replaced by a learned K = σ(b0 + bM·logit M + bm·m).
4. **Pre-exam revision planner**: weekly decision points over the last 4 weeks, recommending the 3 topics with the
   largest expected loss M − K(exam). Example plans: `sample_revision_plans.md`.
5. **Recommender** re-tuned (weights only; v2 filters frozen) on the v3 knowledge state, using the mastery-scale
   K_score = M·R. The learned K is not on the score scale: on validation, raw K gave A5 0.726 against 0.802 for
   K_score.

## Variant comparison (validation, DDD/FFF 2014B)

`variant_comparison_val.csv`; coefficients fitted on train, decay hyper-parameters and attribution chosen on validation.

| ID | Memory model | Attribution | Val log-loss | Val within-course AUC | DDD-2014B exam Δρ |
| --- | --- | --- | --- | --- | --- |
| V0 | v2 learned half-life (reference) | direct (v2 features) | 0.1613 | 0.898 | −0.097 |
| V1 | learned half-life | shared, H = 0.4 | 0.1616 | 0.898 | −0.041 |
| V2 | ACT-R, fixed d (= 2.0, at bound) | direct (content only) | **0.1578** | **0.903** | −0.014 |
| **V3** | **ACT-R, spacing (a = 0.60, c = 0.217)** | **shared, H = 0.4** | 0.1580 | 0.902 | **−0.008** |
| V4 | DAS3H windows on attributed practice | shared, H = 0.4 | 0.1623 | 0.885 | −0.168 |

- No variant reached Δρ >= 0. The preregistered fallback (highest Δρ within 0.002 of the best log-loss) selected **V3**.
- Validation exam check for V3 vs v2: improvement +0.089 [+0.048, +0.134].
- Validation recommender NDCG@5 (combined): v3 A5 0.802 vs v2 A5 0.778 (+0.024, p_Holm < 1e-4); v3 A8 0.825; v3 A9 0.828.

## Preregistered criteria

| Criterion | Result | Met? |
| --- | --- | --- |
| (a) Holdout within-course AUC not worse than the v2 learned half-life by more than 0.002 | v3 **0.848** [0.838, 0.859] vs v2 0.841 [0.831, 0.851]; difference **+0.007 [+0.003, +0.011]** | **met** |
| (b) CCC exam Δρ above v2's −0.096, with the paired CI of the improvement excluding 0 | v3 Δρ **−0.026** vs v2 −0.096 (recomputed −0.0955); improvement **+0.070 [+0.052, +0.088]** | **met** (exploratory) |
| (c) Holdout NDCG@5 of the v3 forgetting-aware configuration >= its v2 counterpart | v3 A5 **0.654** [0.646, 0.661] vs v2 A5 0.718 [0.712, 0.723] | **not met** |

## Holdout results (primary)

**Prediction** (`holdout/model_metrics_holdout*.csv`), 24,122 topic outcomes:

| | Within-course AUC | Log-loss | AAA | BBB | CCC | EEE | GGG |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2 learned half-life | 0.841 | 0.2030 | 0.850 | 0.835 | 0.807 | 0.927 | 0.889 |
| v3 (V3) | **0.848** | **0.2018** | **0.875** | 0.830 | **0.826** | 0.919 | **0.894** |

v3 is better in 3 of 5 modules and worse in BBB and EEE. The pooled gain is +0.007 [+0.003, +0.011].

**Ranking** (`holdout/recommender_holdout.csv`, `recommender_wilcoxon_holdout.csv`), NDCG@5 pooled over 4,526 students:

| Config | Combined | Rel-A | Rel-B |
| --- | --- | --- | --- |
| v2 A3 = v3 A3 | 0.708 [0.703, 0.713] | 0.704 | 0.720 |
| v2 A5 | **0.718** [0.712, 0.723] | 0.714 | 0.717 |
| v2 A9 | **0.742** [0.737, 0.746] | 0.738 | 0.734 |
| v3 A5 | 0.654 [0.646, 0.661] | 0.656 | 0.486 |
| v3 A6 | 0.617 [0.610, 0.625] | 0.619 | 0.470 |
| v3 A8 | 0.680 [0.673, 0.687] | 0.680 | 0.558 |
| v3 A9 | 0.652 [0.645, 0.659] | 0.651 | 0.548 |

Wilcoxon, Holm-corrected, combined NDCG@5:
- v3 A5 vs v2 A5: −0.064, p < 1e-4;
- v3 A5 vs v3 A3: −0.055, p < 1e-4;
- v3 A9 vs v3 A8: −0.028, p < 1e-4.

**The validation gain in ranking did not transfer.** The validation-tuned weights put all of the score on
difficulty fit D (A5/A6) or on D, F, G and popularity (A8/A9). On new modules this ranks the forgetting-flagged
topics in an order that matches neither what passing students engaged with nor what failing students needed
(Rel-B drops from 0.72 to 0.49).

## Exploratory results (each scored once; post-hoc)

| | v2 | v3 | Difference [95% CI] |
| --- | --- | --- | --- |
| CCC exam Δρ (holdout) | −0.096 | −0.026 | +0.070 [+0.052, +0.088] |
| DDD 2014J exam Δρ | −0.088 | −0.012 | +0.076 [+0.043, +0.111] |
| DDD/FFF 2014J within-course AUC | 0.906 | 0.911 | +0.005 [0.000, +0.009] |
| DDD/FFF 2014J NDCG@5, A5 (combined) | 0.784 | 0.730 | −0.054, p_Holm < 1e-4 |
| DDD/FFF 2014J NDCG@5, A9 (combined) | 0.797 | 0.767 | −0.030 |

In every exam set the decayed knowledge K is still slightly *less* correlated with the exam than mastery M is
(Δρ < 0), but the gap shrinks from about −0.09 to about −0.01 to −0.03.

## Planner evaluation

- **Exam check** (K at the exam, as the planner uses it): see the table above. v3 beats v2 on DDD validation
  (+0.089), DDD 2014J (+0.076) and CCC (+0.070).
- **Observational check — not causal.** Exam score was regressed on mean M, log1p(clicks in the last 28 days), and the
  share of attributed revision that went to the topics the planner flagged 4 weeks before the exam:

| Set | n | Coefficient of flagged share [95% CI] |
| --- | --- | --- |
| DDD validation 2014B | 472 | −21.5 [−37.5, −7.1] |
| CCC 2014J (exploratory) | 823 | −22.5 [−28.4, −16.6] |
| DDD 2014J (exploratory) | 865 | +6.5 [−7.8, +20.4] |

  The association is negative in two sets and null in the third. Students who spend their revision on the
  topics the planner flags do **not** score higher, given their mastery and activity. Selection is the obvious
  explanation (weaker students revise what they found hard), so this says nothing about the effect of following
  the plan. 49, 342 and 77 exam-takers had no attributed revision practice and were excluded.
- The planner always fills its 3 slots, because every assessed topic loses some predicted knowledge by the exam.

## Paste-ready paragraph for the report

> A third, preregistered revision of the forgetting component targeted the diagnosis that students' revision was
> invisible to the model. Navigation pages were separated from content, practice on shared and exam-preparation
> resources was credited to the topics it plausibly revised, and the elapsed-time half-life was replaced by an
> ACT-R activation over the full, spacing-sensitive practice history. On five modules never used for design, this
> knowledge model predicted topic outcomes slightly better than the v2 half-life model (within-course AUC 0.848 vs
> 0.841; difference +0.007, 95% CI +0.003 to +0.011). In the only holdout module with exam scores, the gap between
> decayed knowledge and plain mastery as predictors of the final exam narrowed from −0.096 to −0.026 (improvement
> +0.070, CI +0.052 to +0.088), and a similar change appeared in DDD's test cohort. Decayed knowledge still did not
> beat mastery outright, though. The recommender built on the new knowledge state did not transfer: although it
> ranked better on the validation cohort, on the holdout modules its forgetting-aware configuration scored 0.654
> NDCG@5 against 0.718 for the v2 equivalent, failing the third preregistered criterion. A pre-exam revision planner
> produced plausible weekly plans, but in observational data students who revised the flagged topics did not score
> higher. In short, making revision visible improved the memory model as a predictor; it has not yet produced
> better recommendations, and only a controlled study could tell whether the planner helps.

## What did not work or was changed along the way

- **Outer tuning, first attempt.** The finite-difference L-BFGS-B stayed at its starting point because the
  validation loss is flat at step 1e-3. It was replaced by grid + refinement before preregistration (`DECISIONS.md`
  v3-9).
- **Learned K is not on a probability scale.** The recommender and planner therefore use K_score = M·R
  (validation choice, v3-13). The exam check and the outcome model use the learned K as specified.
- **Edge values.** V2's d hit its upper bound (2.0), and V3's a = 0.6 is the top of its grid.
- **"Training cohort" for attribution weights** means the site's own presentation, because OULAD site ids are
  presentation-specific (v3-3).
- **Interest variants (A6, A9) are worse than their non-interest counterparts** on the holdout, as in the
  validation/2014J Rel-B results.

## Reproducibility and tests

- **Tests.** `python -m pytest -q`: **42 passed**. That is the 30 from v1/v2 plus 12 new:
  - `test_v3_attribution.py` (7): weights >= 0, summing to 1 per site, using only pre-assessment window clicks; navigation never contributes practice;
  - `test_v3_memory.py` (5): activation falls without practice and rises after a practice event; the spacing effect; no events at/after the cut-off or exam date are used; attributed half-life features.
- **Clean-clone reproduction.** Fresh `git clone -b v3-revision`, raw CSVs copied in, then v1 01–03, v2 `fix4_holdout.py` (holdout data), and v3 `step1` → `step2` → `step3` → `step45` → `score_outside_validation --part holdout` → `--part posthoc`. It took about 40 minutes.
  - **All 26 CSV tables and 3 JSON files in `results/v3/` match exactly** (max |diff| = 0), as does `sample_revision_plans.md`. Every file was rewritten during the run.
  - The only other differences are the `frozen_at` timestamp of `config/final_config_v3.yaml` and the run time in `models/train_summary.json`.
  - This rerun scored the holdout and 2014J a second time with the same frozen configuration. It is a reproducibility check, not a new evaluation.
- **v1/v2 outputs unchanged:** `git diff v2-fixes --stat -- results/tables results/v2 config/final_config.yaml config/final_config_v2.yaml` prints nothing.
