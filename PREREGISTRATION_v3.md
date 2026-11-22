# v3 preregistration

Written and committed **before** any v3 scoring outside DDD/FFF validation (2014B). Every v3 choice below was
made on validation only (`results/v3/variant_comparison_val.csv`, `knowledge_scale_selection_val.csv`,
`recommender_val.csv`) and is frozen in `config/final_config_v3.yaml` and `results/v3/selected_variant.json`.
The scoring code is `scripts/v3/score_outside_validation.py` as committed together with this file. Each part is run
once. **The criteria below will not be changed after scoring.**

All v3 work is post-hoc: the CCC exam, DDD 2014J and the holdout modules were looked at in v1/v2.

## 1. Selected variant and frozen settings

| Setting | Frozen value |
| --- | --- |
| Resource classes | NAVIGATION = homepage + forumng/subpage/glossary with normalised weekly-click entropy >= 0.80; else CONTENT |
| Attribution | shared-soft, H = 0.4 (co-access weights from pre-assessment window clicks; spread sites, final-window and exam-preparation sites spread over topics assessed before the click day + the current window's topic) |
| Memory model | **V3**: ACT-R activation, spacing-sensitive decay d_i = c·exp(m_i) + a with **a = 0.600, c = 0.217**; submission dose **x_sub = 10.0** |
| Knowledge | K = σ(b0 + bM·logit M + bm·z(m)) (+ bI·z(I) for interest twins); outcome model uses mean K over the chain prerequisites |
| Coefficients | β, θ, b refitted on each course set's training presentations (L2 1e-5; interest twin 1e-4) |
| Recommender knowledge scale | K_score = M·R, R = K(t)/K(t_submit + 1) clipped to [0, 1]; unassessed topics: model prediction |
| Filters | frozen from v2: within-student, q_rev = 50, q_new = 30, τ_q = 0.4 (all configs); p* = 0.6 (A3), 0.9 (A5, A6, A8, A9) |
| Weights (G, Q, F, D, I, Pop) | A3 (0.7, 0, 0, 0.3, 0, 0); A5 and A6 (0, 0, 0, 1.0, 0, 0); A8 and A9 (0.2, 0, 0.2, 0.2, 0, 0.4) |
| Planner | decision points exam − 28/21/14/7 days; top 3 assessed topics by M − K_score(exam), priority > 0 |
| v2 reference | v2 learned half-life with v1 hyper-parameters, refitted per module (as in v2 Fix 4); v2 recommender configs from `final_config_v2.yaml` |

Selection rule used (applied on validation): lowest validation within-course log-loss among variants with
DDD-validation exam Δρ >= 0. If none reaches >= 0, choose the highest exam Δρ among variants within 0.002 of the
best validation log-loss. No variant reached Δρ >= 0, so the second branch selected V3 (Δρ −0.008, log-loss 0.1580)
over V2 (Δρ −0.014, log-loss 0.1578).

## 2. Comparisons to run

**Primary: five holdout modules (AAA, BBB, CCC, EEE, GGG; train = earlier presentations, test = 2014J).**
1. Knowledge-model prediction on 2014J topic outcomes: within-course AUC (row-weighted mean of the five per-module
   AUCs) and log-loss, v3 vs v2 learned half-life. 95% CI from 1,000 paired bootstrap resamples of students.
2. Recommender: NDCG@5 (combined, Rel-A, Rel-B), plus NDCG@3, P@K and R@K for K ∈ {3, 5}, for v2 A3/A5/A6/A8/A9
   and v3 A3/A5/A6/A8/A9, pooled over the five modules. Paired Wilcoxon with Holm correction on per-student
   NDCG@5 for (i) v3 A5 vs v2 A5, (ii) v3 A5 vs v3 A3, (iii) v3 A9 vs v3 A8. The primary family uses combined
   relevance; Rel-A and Rel-B families are supplementary.

**Exploratory (each scored once, labelled post-hoc):**
3. CCC exam check (the only holdout module with exam scores): Δρ = ρ(mean K) − ρ(mean M) for v3 and the v2
   reference; paired bootstrap CI (1,000) of the improvement. Planner on CCC with the observational regression.
4. DDD/FFF 2014J: model metrics (v3 vs v2 learned), recommender (same comparisons as 2), DDD 2014J exam check,
   planner and observational regression.

## 3. Success criteria (fixed in advance)

| | Criterion | Operational definition | Met if |
| --- | --- | --- | --- |
| (a) | Holdout within-course AUC not worse than the v2 learned half-life by more than 0.002 | point estimate of (v3 − v2) within-course AUC, comparison 1 | v3 − v2 >= −0.002 |
| (b) | CCC exam Δρ higher than v2's −0.096, with the paired bootstrap CI of the improvement excluding 0 | comparison 3 | Δρ_v3 > −0.096 **and** the lower 95% bound of (Δρ_v3 − Δρ_v2) > 0 |
| (c) | Holdout NDCG@5 of the v3 forgetting-aware configuration >= its v2 counterpart | pooled combined NDCG@5 point estimates, v3 A5 vs v2 A5 | v3 A5 >= v2 A5 |

The v2 reference values are recomputed in the same run with the same code; they are expected to reproduce v2's
published numbers (CCC Δρ −0.096, v2 A5 NDCG@5 0.718). Each criterion is reported as met or not met, with
numbers and CIs, in `results/v3/preregistered_criteria.csv` and `results/v3/SUMMARY_v3.md`.
