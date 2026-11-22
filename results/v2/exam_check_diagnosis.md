# Exam retention check: diagnosis (v2 Fix 1, post-hoc, validation only)

Produced by `scripts/v2/fix1_exam_diagnosis.py`; tables in `results/v2/exam_check/`. Everything here uses the
**validation** split (2014B). No test data were read for this diagnosis.

## a) Does validation show the same sign as test?

Yes. v1 `results/tables/exam_retention_check_val.csv`, Spearman ρ(mean K, exam) − ρ(mean M, exam):

| Decay | Validation Δρ [95% CI] | Test Δρ (v1, for reference) |
| --- | --- | --- |
| fixed S = 120 d | −0.139 [−0.191, −0.091] | −0.152 |
| learned half-life | −0.097 [−0.142, −0.054] | −0.088 |
| learned + interest | −0.110 [−0.158, −0.068] | −0.096 |

**Scope v1 did not state.** OULAD has exam scores only for modules CCC and DDD. FFF has none, so the v1 exam
check is effectively a **DDD-only** check (524 validation and 950 test students). On the holdout modules (Fix 4),
only CCC can be checked.

## b) Audit of Δ at the exam date — no bug

- Δ was recomputed independently, with a naive loop over `vle_daily` implementing Methodology 5.1 literally: last meaningful practice = max(submission day, any later day before the exam with >= 3 clicks on the topic's resources).
- This was done for all 2,930 assessed (student, topic) pairs at the validation exam dates.
- **Result:** the pipeline matches on every pair (max |difference| = 0, no pair missing on either side).
- Revision-period clicks (between the last assessment and the exam) are counted and do reset Δ.
- Days on or after the exam date are not used. There are 38 such click rows in the data, and none affects Δ.
- New unit tests (`tests/test_v2_exam_delta.py`, 5 tests) cover:
  - a reset by a revision-period day with >= 3 clicks;
  - no reset with fewer than 3 clicks, or on/after the exam day;
  - the day before the exam is counted;
  - clicks before the submission do not count.

Verdict: **no BUG-FIX**. The code does what Methodology 5.1 specifies, so (d) the validation exam check is unchanged (Δρ = −0.097 for the learned decay).

## c) Diagnostics (validation, DDD 2014B, learned half-life)

Per topic, ρ with the exam score (`per_topic_exam_check_val.csv`, `delta_and_revision_at_exam_val.csv`):

| Topic | ρ(M_j) | ρ(K_j) | Δρ | Median Δ at exam (days) | Mean R | Revision-period clicks (any / reset ≥3) | ρ(R_j, exam) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.412 | 0.458 | **+0.046** | 13 | 0.92 | 92% / 85% | 0.28 |
| 2 | 0.531 | 0.334 | **−0.196** | 188 | 0.46 | 48% / 23% | 0.22 |
| 3 | 0.570 | 0.292 | **−0.277** | 159 | 0.60 | 60% / 34% | 0.11 |
| 4 | 0.626 | 0.503 | −0.123 | 118 | 0.64 | 53% / 31% | 0.21 |
| 5 | 0.463 | 0.358 | −0.105 | 83 | 0.60 | 46% / 20% | 0.19 |
| 6 | 0.497 | 0.471 | −0.027 | 13 | 0.90 | 92% / 90% | 0.29 |

Leave-one-topic-out (`leave_one_topic_out_val.csv`): dropping topic 3 moves the overall Δρ from −0.097 to −0.065,
and dropping topic 2 moves it to −0.083. Dropping topics 1 or 6 makes it *more* negative (−0.125, −0.136).

**What drives the negative sign:**

1. **The middle topics (2–5) drive it.** Most students never come back to their resources with >= 3 clicks a
   day after submitting: only 20–34% reset Δ in the revision period. So Δ at the exam is the time since submission
   (median 83–188 days) and R falls to about 0.5–0.6. Topics 1 and 6 have Δ ≈ 13 days and R ≈ 0.9, so K ≈ M there,
   and on its own each of those topics even shows K slightly ahead of M.
2. **R carries a weak exam signal compared with the score.** For the middle topics, ρ(R_j, exam) is 0.11–0.22 while
   ρ(M_j, exam) is 0.46–0.63. Multiplying M by an R whose spread comes mostly from "did the student re-open this
   topic's pages" mixes a strong signal with a much weaker one. ρ(K) ends up between the two.
3. **Revision probably happens through resources that are not mapped to the old topics.** 92% of exam-takers click
   topic-1 and topic-6 resources in the revision period. Topic 1 holds course-wide resources (v1 `DECISIONS.md` #14),
   and topic 6 holds the material of the last window, which in DDD includes exam preparation. Revising the content of
   topics 2–5 through those pages does not reset their Δ. The assessment-window topic definition (v1 limitation)
   therefore makes many students look as if they "forgot" topics they in fact revised.
4. **The learned half-life does not repair this.** Its engagement inputs (φ_c < 0, φ_d > 0, collinear) leave R
   only weakly related to how much a student used the topic: for topics 2–5, ρ(R, c_inf) = −0.07 to +0.20. R is
   driven mainly by Δ, which (point 3) misses revision done through other topics' resources.

**Conclusion for the report.** The negative exam result is not a code error. It reflects the decay as modelled:
Δ is measured from clicks on a topic's own resources, and with assessment-window topics those clicks miss most
revision. The check is also DDD-only. As instructed, no further decay variants were tried.
