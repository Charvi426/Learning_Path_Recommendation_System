# Methodology: Forgetting- and Interest-Aware Learning Path Recommendation on OULAD

Nov 19, 2026 · @Charvi

## 1. Overview

The system predicts, for every student and topic, the probability of passing that topic's assessment, and lets that prediction decay with time since the topic was last practised. It then ranks new and revision topics with an interpretable score. Everything runs as scikit-learn and pandas code on a laptop; one optional deep baseline (DKT) is the only part that benefits from a GPU.

The pipeline has six stages, each writing a file the next stage reads:

1. **Preprocess**: clean the seven OULAD tables, define topics as assessment windows, build a student × topic interaction table with binary outcomes.
2. **Compute factors**: engagement in time windows, prior performance, behavioural interest, prerequisite graph, topic difficulty.
3. **Fit the knowledge model**: one logistic model (PFA → DAS3H-style) that gives mastery, effective knowledge and retention for any student, topic and date.
4. **Generate candidates**: new topics whose prerequisites are ready, and learned topics whose effective knowledge has dropped.
5. **Rank and build the path**: hybrid score, prerequisite-ordered path of K = 5 items, one explanation per item.
6. **Evaluate**: prediction metrics for the model, ranking metrics and ablations for the recommender, plus an exam-based check of the forgetting component.

&#91;embedded content: pipeline · 6 stages\]

The knowledge model (stage 3) is the core: candidates, ranking and evaluation all read its M, K and R values rather than refitting anything.

Design principles: every number the recommender uses comes from a fitted, inspectable coefficient; every feature for a prediction at time t uses only data before t; and every component can be switched off for the ablation study.

## 2. Project file structure

One Python package (`src/`) holds the logic; thin scripts in `scripts/` run each stage from the config, so every result can be regenerated with five commands.

```
fds-learning-path/
├── README.md                  # setup + how to reproduce every table
├── requirements.txt           # pandas, numpy, scikit-learn, scipy, pyarrow, pyyaml, matplotlib (pykt-toolkit optional)
├── config/
│   └── config.yaml            # paths, modules, pass mark, time windows, thresholds, weight grid, seeds
├── data/
│   ├── raw/                   # the 7 OULAD CSVs, never edited
│   ├── interim/               # cleaned tables (parquet)
│   └── processed/             # topic_map, prereq_edges, interactions, features_{train,val,test}
├── src/
│   ├── data/
│   │   ├── load.py            # read raw CSVs with dtypes
│   │   ├── clean.py           # banked rows, missing dates/scores, withdrawals, click caps
│   │   ├── topics.py          # assessment windows, id_site -> topic mapping
│   │   └── split.py           # presentation-based temporal split
│   ├── features/
│   │   ├── engagement.py      # windowed click / success / failure counts
│   │   ├── interest.py        # behavioural interest proxy I(s,k)
│   │   ├── prerequisites.py   # chain graph + optional data-derived edges
│   │   ├── difficulty.py      # cohort pass rate per topic
│   │   └── build.py           # assembles one feature row per (student, topic, time)
│   ├── models/
│   │   ├── baselines.py       # majority, topic-difficulty (IRT-like)
│   │   ├── pfa.py             # PFA logistic model
│   │   ├── das3h.py           # PFA + time windows + interest terms
│   │   ├── decay.py           # fixed-S exponential and HLR half-life variants
│   │   ├── dkt_pykt.py        # optional deep baseline via pyKT
│   │   └── knowledge_state.py # M, K, R for any (student, topic, date)
│   ├── recommend/
│   │   ├── candidates.py      # new + revision candidate sets
│   │   ├── scoring.py         # G, Q, F, D, I and the hybrid score
│   │   ├── path.py            # prerequisite-ordered path of K items
│   │   └── explain.py         # template explanations
│   ├── evaluation/
│   │   ├── model_metrics.py   # AUC, log-loss, RMSE, Brier, calibration
│   │   ├── relevance.py       # ground-truth relevant topics per decision point
│   │   ├── ranking_metrics.py # Precision@K, Recall@K, NDCG@K
│   │   ├── ablation.py        # runs every configuration in section 7
│   │   └── stats.py           # bootstrap CIs, Wilcoxon tests
│   └── utils/                 # io, logging, seeding
├── scripts/
│   ├── 01_preprocess.py
│   ├── 02_build_features.py
│   ├── 03_train_models.py
│   ├── 04_tune_recommender.py
│   └── 05_evaluate.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_topic_windows.ipynb
│   ├── 03_factor_checks.ipynb  # interest proxy validity, decay curves
│   └── 04_results.ipynb
├── models/                    # fitted models (.joblib) + coefficients (.csv)
├── results/
│   ├── tables/                # metrics CSVs that go into the report
│   ├── figures/
│   └── sample_paths/          # example paths + explanations for the report
├── tests/
│   ├── test_topics.py
│   ├── test_no_leakage.py     # asserts every feature timestamp < prediction time
│   └── test_metrics.py
└── report/                    # final report source
```

Key processed tables:

| File | One row per | Main columns |
| --- | --- | --- |
| `topic_map.parquet` | id\_site | module, presentation, topic, activity\_type, voluntary flag |
| `prereq_edges.csv` | prerequisite edge | module, presentation, from\_topic, to\_topic, source (chain / data) |
| `interactions.parquet` | student × topic | t\_open, t\_due, t\_submit, score, y (pass), is\_submitted |
| `vle_daily.parquet` | student × topic × day | clicks, voluntary clicks |
| `features_{split}.parquet` | student × topic × prediction time | all factors in section 4 + y |
| `knowledge_state.parquet` | student × topic × decision time | M, K, R, I, Q, D |

## 3. Dataset and preprocessing

OULAD has no topic labels and no per-question answers, so preprocessing must create both: a topic is the period between two consecutive non-exam assessments, and the outcome is whether the student passed that assessment. All dates in OULAD are days relative to the module start, so times are plain integers.

### 3.1 Tables used

| Table | Used for | Key columns |
| --- | --- | --- |
| `courses` | presentation length (fills missing exam dates) | code\_module, code\_presentation, module\_presentation\_length |
| `assessments` | topic boundaries, assessment type and weight | id\_assessment, assessment\_type (TMA / CMA / Exam), date, weight |
| `studentAssessment` | outcomes | id\_assessment, id\_student, date\_submitted, is\_banked, score |
| `vle` | resource metadata | id\_site, activity\_type, week\_from, week\_to |
| `studentVle` | engagement and timing | id\_student, id\_site, date, sum\_click |
| `studentRegistration` | active period of each student | date\_registration, date\_unregistration |
| `studentInfo` | prior attempts; demographics for fairness checks only | num\_of\_prev\_attempts, highest\_education, final\_result |

### 3.2 Cleaning steps (`clean.py`)

1. **Select modules.** Count non-exam assessments per module presentation (`assessments.groupby([...]).size()`) and keep modules with at least 5. Fewer windows give too few topics for a path.
2. **Drop banked results** (`is_banked == 1`): they were earned in an earlier presentation and carry no timing information.
3. **Missing assessment dates.** Exams with no date get `module_presentation_length`. Exams are kept out of the topic set and used only for the retention check in section 7.4.
4. **Missing or absent scores.** A submission with no score is dropped. A non-submission is kept with `is_submitted = 0`. It is excluded from model training targets by default, and a sensitivity run treats it as a fail.
5. **Withdrawals.** For a student who unregistered, keep only topics whose due date falls before `date_unregistration`.
6. **Clicks.** Aggregate `studentVle` to student × id\_site × day. Cap `sum_click` at the 99.5th percentile per activity type, and use log(1 + clicks) everywhere. Clicks before day 0 are assigned to the first topic.

### 3.3 Topic definition (`topics.py`)

Order the non-exam assessments of a presentation by date: a₁, …, a\_n. Topic k is the window (t\_{k−1}, t\_k\], with t₀ = the first day of VLE activity and t\_k = the due date of a\_k. Two assessments due on the same day are merged into one topic, and their weighted mean score is used.

Each VLE resource (`id_site`) gets one topic:

- if `week_from` is present, the topic whose window contains day 7 × week\_from;
- otherwise, the topic window in which the whole cohort clicked it most.

Document the share of resources mapped each way, and plot clicks per topic over time in `02_topic_windows.ipynb` as a sanity check.

### 3.4 Outcome

```latex
y_{s,k} = \mathbb{1}[\text{score}_{s,k} \ge 40], \qquad \tilde{y}_{s,k} = \text{score}_{s,k}/100
```

The binary y uses the Open University pass mark of 40 and drives the classification models. The normalised score is kept for RMSE/MAE reporting. A sensitivity run uses a threshold of 70 ("good mastery") to check the conclusions don't depend on 40.

### 3.5 Train / validation / test split (`split.py`)

Split by presentation, because it is the cleanest temporal split OULAD allows: no future cohort leaks into training.

| Split | Presentations | Purpose |
| --- | --- | --- |
| Train | 2013B, 2013J | fit all model coefficients and cohort statistics (difficulty, z-score means) |
| Validation | 2014B | tune thresholds, score weights, decay variant |
| Test | 2014J | report final numbers once |

Modules that did not run in 2014B split the 2014J students 50/50 by `id_student` into validation and test. Within every split, predictions for a topic use only events before its prediction time (section 4.1). `tests/test_no_leakage.py` asserts this for every feature column.

## 4. Factor computation

Every factor is computed for a student s, a topic k and a time t, from events strictly before t. The model in section 5 turns these into mastery, retention and effective knowledge; the recommender in section 6 adds readiness, difficulty and interest.

### 4.1 Prediction and decision times

- **Training rows:** one per (s, k) with t = t\_k, the due date of topic k's assessment. The label is y\_{s,k}.
- **Decision points:** one per (s, k) with t = the day after the student submits assessment k−1. This is when the system recommends what to do next. The knowledge state for every topic is recomputed at that t.

### 4.2 Engagement in time windows (`engagement.py`)

Following DAS3H, practice is counted in nested windows before t, so the model can learn that recent practice counts more than old practice. Windows W = {7, 30, ∞} days (tune {7, 14, 30, ∞} on validation).

| Feature | Definition (before t, within window w) |
| --- | --- |
| c\_{s,k,w} | log(1 + clicks on topic k resources) |
| d\_{s,k,w} | number of distinct active days on topic k |
| succ\_{s,P(k),w} | number of passed prerequisite topics with an assessment in w |
| fail\_{s,P(k),w} | number of failed prerequisite topics with an assessment in w |
| Δt\_{s,k} | days since the last click on topic k resources (or its submission, if later) |

P(k) = the prerequisites of k (section 4.5). With the chain graph, P(k) = {k−1}, so the success and failure counts capture how well and how recently the previous topic was learned.

### 4.3 Prior performance

- **ability\_s(t):** the student's mean y over all topics assessed before t in this presentation, shrunk towards the cohort mean with weight 2 (so one early result doesn't dominate). New students start at the cohort mean.
- **prev\_attempts\_s:** `num_of_prev_attempts`, capped at 2.

Demographic columns are **not** model inputs by default. They are used only in a fairness check (section 7.5), so the recommender doesn't steer students by gender, region or IMD band.

### 4.4 Behavioural interest proxy I\_{s,k} (`interest.py`)

Interest is measured as voluntary, topic-relative engagement: activity a student chose beyond what their grade required, compared with their own other topics.

1. **Voluntary resource types.** V = {forumng, ouwiki, oucollaborate, glossary, url, resource, ouelluminate}. Check the actual `activity_type` values first, and justify each inclusion in the report.
2. **Three raw signals**, all counted before t:
   - vol\_{s,k}: log(1 + clicks on V-type resources of topic k);
   - early\_{s,k}: log(1 + clicks on topic k resources before its window opens);
   - post\_{s,k}: log(1 + clicks on topic k resources after its assessment was submitted). This exists only for topics already assessed.
3. **Combine:** raw = vol + λ₁·early + λ₂·post, with λ₁ = λ₂ = 1 by default.
4. **Within-student share:** share\_{s,k} = raw\_{s,k} / Σ\_j raw\_{s,j}, over the topics the student has opened so far.
5. **Cohort z-score** per (module, presentation, topic), using means and standard deviations from the training split.
6. **Topic-relative interest:** subtract the student's own mean z, then squash.

```latex
I_{s,k} = \sigma\!\left( z_{s,k} - \frac{1}{|J_s|}\sum_{j \in J_s} z_{s,j} \right) \in (0,1)
```

I = 0.5 when the student has no voluntary activity yet (neutral). Subtracting the student's mean removes general diligence: a student who clicks everything a lot gets I ≈ 0.5 everywhere.

**Validity checks** (in `03_factor_checks.ipynb`, reported in the paper):

- the three signals should correlate positively with each other (they measure the same thing);
- I should correlate weakly with total clicks (target |r| < 0.3), confirming it isn't just activity;
- split-half stability: I computed from odd vs. even days should correlate.

In the report, call I a behavioural interest proxy, never measured interest.

### 4.5 Prerequisite graph (`prerequisites.py`)

- **Default: chain.** k−1 → k within each presentation. OULAD course order is the only stated structure.
- **Optional: data-derived edges** u → v for u < v when, in training data, P(y\_v = 1 | y\_u = 1) − P(y\_v = 1 | y\_u = 0) > 0.15 with at least 200 students in each group. Report the resulting edges and compare chain vs. chain + data edges in the ablation.

### 4.6 Topic difficulty

```latex
\text{diff}_k = 1 - \frac{\#\{s : y_{s,k}=1\}}{\#\{s : \text{submitted } k\}} \quad \text{(training split only)}
```

This feeds the topic intercept's prior and the difficulty-suitability term D in section 6.

## 5. Models

The main model is a forgetting-aware PFA: a logistic model of passing topic k, where the knowledge carried in from prerequisite topics decays with a learned half-life. It keeps the report's equations (2)–(4), but learns the weights and the memory strength S from data instead of setting them by hand. Everything else in this section is either a baseline it must beat or an alternative it is checked against.

### 5.1 Main model: forgetting-aware PFA (`das3h.py`, `decay.py`)

The knowledge state of a topic j that has already been assessed:

```latex
M_{s,j} = \tilde{y}_{s,j}, \qquad R_{s,j}(t) = 2^{-\Delta_{s,j}(t)/h_{s,j}}, \qquad K_{s,j}(t) = M_{s,j}\,R_{s,j}(t)
```

- M is mastery, taken as the normalised score at the time j was assessed.
- Δ is the number of days since the last meaningful practice of j: its submission, or a later day with at least 3 clicks on j's resources, so revision resets the clock.
- h is the half-life in days. This is the report's exp(−t/S) with S = h / ln 2.

The half-life is learned, HLR-style, from the student's engagement with j and their interest in it:

```latex
\log_2 h_{s,j} = \phi_0 + \phi_c\, c_{s,j,\infty} + \phi_d\, d_{s,j,\infty} + \phi_I\, I_{s,j}
```

The outcome of the next topic is the "recall test" for its prerequisites:

```latex
\operatorname{logit} P(y_{s,k}=1) = \beta_k + \theta_a\,\text{ability}_s + \theta_K\,\bar{K}_{s,P(k)}(t_k) + \theta_c\, c_{s,k,\infty} + \theta_I\, I_{s,k} + \theta_{Ic}\, I_{s,k}\, c_{s,k,\infty}
```

Here K̄ is the mean K over k's prerequisites at k's due date. Each term maps to one hypothesis:

| Coefficient | Expected sign | What it tests |
| --- | --- | --- |
| θ\_K | + | knowledge carried in from prerequisites helps |
| φ\_c, φ\_d | + | more practice of a topic means slower forgetting |
| φ\_I | + | interest means longer retention ("retains longer") |
| θ\_Ic | + | each click on an interesting topic is worth more ("learns faster") |
| θ\_I | + or 0 | direct effect of interest on passing |

**Fitting.** Minimise L2-regularised log-loss over (β, θ, φ) with `scipy.optimize.minimize` (L-BFGS-B). Bound φ so that h stays within \[1, 365\] days, standardise inputs, and initialise θ from the plain PFA fit. Tune the L2 strength on validation log-loss. Report every coefficient with a 95% bootstrap interval: 200 resamples of students, refitting each time.

**Identification check (do this first).** h is only learnable if Δ varies across students. Plot the distribution of Δ at prerequisite due dates. If it is narrow (most students submit right at the deadline), add the data-derived non-adjacent edges from section 4.5, which give longer and more varied gaps.

### 5.2 Forgetting variants (for the ablation)

| Variant | R\_{s,j}(t) | Fitted parameters |
| --- | --- | --- |
| No forgetting | 1 | none |
| Fixed exponential (report eq. 3) | exp(−Δ/S), one S for everyone | S chosen from {7, 14, 30, 60, 120} days on validation |
| Learned half-life (main) | 2^(−Δ/h), h per student and topic | φ₀, φ\_c, φ\_d |
| Learned half-life + interest | as above, with the φ\_I term | φ₀, φ\_c, φ\_d, φ\_I |

### 5.3 Benchmark: DAS3H-style windows (`das3h.py`)

As a check that the parametric decay isn't costing accuracy, fit a plain `sklearn.linear_model.LogisticRegression`. Its inputs are topic one-hot, ability, and for w ∈ {7, 30, ∞}: c\_{s,k,w}, succ\_{s,P(k),w}, fail\_{s,P(k),w}, plus I and I × c. It has no explicit curve; recency is captured by window coefficients. If it beats the main model by more than 0.01 AUC, report it, and consider using its predictions for K.

### 5.4 Baselines

| Model | Inputs | Role |
| --- | --- | --- |
| Majority | none | floor |
| Topic difficulty (IRT-like) | β\_k + θ\_a · ability | is any learning signal useful? |
| PFA | + succ / fail counts and clicks, no time | the main model without forgetting or interest |
| DKT (optional, `dkt_pykt.py`) | topic + y sequence via pyKT | deep comparison |

For DKT, convert interactions to pyKT's format (one sequence per student and presentation, concept = topic id) and run its default DKT configuration. Sequences are short (one per topic, usually under a dozen steps), so expect DKT to match PFA at best. Gervet et al. (2020) report this pattern on moderate-sized data, and you can cite it when explaining the result.

### 5.5 Knowledge state used by the recommender (`knowledge_state.py`)

At each decision time t, for every topic j of the student's presentation:

- **assessed topics:** M = ỹ\_{s,j}, R = 2^(−Δ/h), K = M · R;
- **topics not yet assessed:** M = K = the model's predicted P(y\_{s,j} = 1) if attempted now, with R undefined ("–" in the report's Table II).

Store these in `knowledge_state.parquet` with I\_{s,j}, so sections 6 and 7 never refit anything.

## 6. Recommendation

At each decision point the system filters topics into new and revision candidates, scores them with five interpretable terms, and outputs a path of K = 5 items ordered so that no topic comes before its prerequisites. The candidate filter plays the role of CSEAL's cognitive navigation (Liu et al., 2019). The scoring replaces its reinforcement-learning policy with a transparent weighted sum.

### 6.1 Candidate generation (`candidates.py`)

```latex
C^s_{\text{new}} = \{k \text{ not yet assessed} : M_{s,k} < \tau_m,\; Q_{s,k} \ge \tau_q,\; k \le k_{\text{now}} + 2\}
```

```latex
C^s_{\text{rev}} = \{j \text{ assessed} : K_{s,j}(t) < \tau_r\}, \qquad C^s = C^s_{\text{new}} \cup C^s_{\text{rev}}
```

The horizon k\_now + 2 keeps new topics close to where the course actually is. Starting thresholds are τ\_m = 0.8, τ\_q = 0.5 and τ\_r = 0.6, tuned on validation over {0.4, 0.5, 0.6, 0.7, 0.8}.

### 6.2 Score terms (`scoring.py`)

All terms lie in \[0, 1\], so the weights are comparable.

| Term | New topic k | Revision topic j | Meaning |
| --- | --- | --- | --- |
| G (knowledge gap) | 1 − M\_{s,k} | 1 − M\_{s,j} | how much is still to learn |
| Q (readiness) | min over j ∈ P(k) of K\_{s,j}(t); 1 if no prerequisites | 1 | prerequisites still retained |
| F (forgetting) | 0 | M\_{s,j} − K\_{s,j}(t) = M(1 − R) | knowledge that was learned and then lost |
| D (difficulty fit) | 1 − \|p̂ − p\*\| / max(p\*, 1 − p\*) | same, with p̂ = K\_{s,j}(t) | closeness to a target success rate p\* |
| I (interest) | I\_{s,k} | I\_{s,j} | behavioural interest proxy |

p̂ is the predicted pass probability, and p\* = 0.7 is the target (tune over {0.6, 0.7, 0.8}). Using M − K for F separates "forgotten" from "never learned", which the report's 1 − K (eq. 5) lumps together. G covers the never-learned part.

```latex
S_{s,k} = \alpha G_{s,k} + \beta Q_{s,k} + \gamma F_{s,k} + \delta D_{s,k} + \varepsilon I_{s,k}, \qquad \alpha+\beta+\gamma+\delta+\varepsilon = 1,\; \varepsilon \le 0.2
```

The cap on ε keeps interest a tie-breaker. Because I only re-ranks topics that already passed the prerequisite filter, a low-interest prerequisite is never dropped.

### 6.3 Weight tuning (`04_tune_recommender.py`)

- Grid over (α, β, γ, δ, ε) in steps of 0.1, summing to 1, with ε ≤ 0.2: 671 combinations, each a vectorised pass over the validation decision points.
- Objective: NDCG@5 on validation (section 7.2); ties are broken by Recall@5.
- Tune the thresholds from 6.1 first with equal weights, then the weights, then repeat the thresholds once.
- Freeze everything before touching the test split.

### 6.4 Path construction (`path.py`)

1. Rank C^s by S, descending.
2. Take the top 5, with at most 3 revision items so the student still moves forward.
3. Before each new topic, insert any of its prerequisites that are in the revision list.
4. Never place a new topic before an earlier new topic that is its prerequisite.
5. Label each item: **Revise** (revision candidate), **Practice** (new topic already opened, with clicks > 0), or **Learn** (new topic not yet opened).

The report's Table II example then comes out as: Revise Functions → Learn Recursion → …

### 6.5 Explanations (`explain.py`)

Each item's explanation comes from its largest weighted term (w · term). A second reason is added if that one contributes more than 25% of S.

| Top term | Template |
| --- | --- |
| F | "Revise {j}: you scored {M}%, but estimated retention has dropped to {R}% after {Δ} days without practice." |
| G | "Learn {k}: your predicted mastery is still low ({M}%)." |
| Q | "{k} is ready: its prerequisite {j} is still {K}% retained." |
| D | "{k} is at the right level for you now (predicted success {p̂}%)." |
| I | "You've been exploring {k} beyond the required material." |

Save 10 sample paths with explanations (strong, average and struggling students) to `results/sample_paths/` for the report.

## 7. Evaluation

Evaluation answers three questions in order: does the model predict outcomes well, does forgetting-awareness improve the recommended paths, and does interest add anything beyond that? Every number is computed once on the 2014J test split, with 95% bootstrap intervals.

### 7.1 Knowledge-model metrics (`model_metrics.py`)

| Metric | On | Why |
| --- | --- | --- |
| AUC (primary) | P(y\_{s,k} = 1) vs. y | ranking quality, independent of threshold |
| Log-loss, Brier score | same | probability quality |
| RMSE, MAE | predicted p vs. y; and ỹ for the score variant | matches the report's eqs. (11)–(12) |
| Calibration plot (10 bins) | same | K and M are used as probabilities, so they must be calibrated |

Compare: Majority, Topic difficulty, PFA, the four forgetting variants (5.2), DAS3H windows, and optionally DKT. Also report the fitted coefficients from 5.1 with their intervals. A positive φ\_I and θ\_Ic whose intervals exclude 0 is the direct evidence for the interest hypothesis.

### 7.2 Recommendation metrics (`relevance.py`, `ranking_metrics.py`)

Offline, the true "best next topics" are unknown, so relevance is defined from what happened in the following window (from the decision time to t\_k). Two definitions are reported separately:

| Relevance set | Students | A topic j is relevant if… |
| --- | --- | --- |
| Rel-A: helpful engagement | those who passed assessment k | they engaged with j (≥ 10 clicks on j's resources) during the window |
| Rel-B: needed topics | those who scored < 40 on assessment k | j = k, or j is a prerequisite of k (up to two levels back in the chain) |

Rel-A rewards recommending what successful students actually did. Rel-B rewards spotting what struggling students needed. For NDCG@K, use graded relevance where a topic in both sets counts 2.

Metrics: Precision@K, Recall@K and NDCG@K (report eqs. 9–10) for K ∈ {3, 5}, averaged over decision points and then over students.

### 7.3 Simulated learning gain

Apply each configuration's path to each test decision point, then predict the student's pass probability on assessment k as if the path were followed. A Revise item resets Δ for j to 0; a Learn or Practice item adds the median per-topic clicks. Report the mean predicted gain over doing nothing.

Use the **DAS3H window model** (5.3) as the simulator, not the main model. A recommender judged by its own model would grade itself. This is the same simulate-with-a-KT-model idea used in RL path recommendation, applied as an evaluation only. Clearly label it model-based.

### 7.4 Exam-based check of the forgetting component

The final exam is cumulative, so it is a natural test of retention. For each student, compute the mean M and the mean K over all topics at the exam date. If forgetting is modelled correctly, mean K should predict the exam score better than mean M does.

- Spearman correlation of each with the exam score, with a bootstrap interval on the difference;
- AUC of each for exam pass (≥ 40).

This is the strongest single piece of evidence that the decay is real and not an artefact of the recommender.

### 7.5 Ablation study (`ablation.py`)

Each configuration gets its own weights tuned on validation, so differences come from components, not from tuning luck.

| ID | Configuration | Active terms | Forgetting | Interest |
| --- | --- | --- | --- | --- |
| A0 | Random | none | – | – |
| A1 | Popularity | cohort clicks in the window | – | – |
| A2 | Mastery only | G | none | – |
| A3 | Mastery + prerequisites | G, Q, D | none | – |
| A4 | + fixed forgetting | G, Q, D, F | fixed S | – |
| A5 | + learned forgetting (proposed) | G, Q, D, F | learned h | – |
| A6 | + interest | G, Q, D, F, I | learned h with φ\_I | θ\_I, θ\_Ic, ε |
| A7 | A6 + data-derived prerequisites | same | same | same |

The main comparisons are A5 vs. A3 (forgetting), A5 vs. A4 (learned vs. fixed decay) and A6 vs. A5 (interest).

### 7.6 Statistics, sensitivity and fairness (`stats.py`)

- **Intervals:** 1,000 bootstrap resamples of students for every metric.
- **Tests:** paired Wilcoxon signed-rank on per-student NDCG@5 for the three main comparisons, with Holm correction (α = 0.05).
- **Sensitivity:** pass threshold 70 instead of 40; non-submission counted as a fail; windows {7, 14, 30, ∞}; chain vs. data-derived prerequisites. Report whether the A5 > A3 conclusion holds in each.
- **Fairness:** NDCG@5 and AUC broken down by gender, IMD band and disability. Report the largest gap between groups for A3 and A6, so it's visible whether the interest proxy widens it.

## 8. Work plan, risks and references

### 8.1 Work plan (six weeks; shift to your deadline)

1. **Week 1:** load and clean the data, choose the modules, build topic windows and the topic map, run the EDA notebook.
2. **Week 2:** engagement, prior-performance and difficulty features; prerequisite graph; leakage tests passing.
3. **Week 3:** baselines and PFA; the identification check on Δ; the main forgetting-aware model and its variants.
4. **Week 4:** interest proxy with validity checks; refit with interest terms; DAS3H benchmark; optional DKT.
5. **Week 5:** candidates, scoring, path and explanations; tuning on validation; freeze the configuration.
6. **Week 6:** test-split evaluation, ablations, exam check, statistics, fairness; figures and report.

### 8.2 Risks and fallbacks

| Risk | Sign | Fallback |
| --- | --- | --- |
| Δ barely varies, so h can't be learned | narrow Δ histogram; wide intervals on φ | add data-derived non-adjacent edges; else report fixed-S decay (A4) as the main forgetting model |
| Too few topics per module | fewer than 5 windows | split long windows at the midpoint between assessments; keep only the richest modules |
| Interest proxy just measures activity | \|r\| with total clicks ≥ 0.3 | drop the vol signal and keep early + post; report the proxy as inconclusive |
| Relevance definitions favour popularity | A1 close to A5 on Rel-A | lead with Rel-B, the simulated gain and the exam check |
| DKT setup takes too long | not running by week 4 | drop it and cite pyKT and Gervet et al. for the expected result |

### 8.3 Limitations to state in the report

- Topics are assessment windows, not true concepts, and prerequisites are mostly course order.
- Offline relevance is a proxy: the real benefit of a path can only be shown by a live study, which is out of scope.
- Interest is inferred from behaviour and can't be separated fully from motivation or deadlines.

### 8.4 References

1. Kuzilek, Hlosta and Zdrahal (2017). Open University Learning Analytics Dataset. *Scientific Data* 4, 170171. Also available from the UCI Machine Learning Repository.
2. Pavlik, Cen and Koedinger (2009). [Performance Factors Analysis: A New Alternative to Knowledge Tracing](https://researchr.org/publication/PavlikCK09). AIED 2009.
3. Vie and Kashima (2019). [Knowledge Tracing Machines: Factorization Machines for Knowledge Tracing](https://ojs.aaai.org/index.php/AAAI/article/view/3853). AAAI 2019.
4. Choffin, Popineau, Bourda and Vie (2019). [DAS3H: Modeling Student Learning and Forgetting for Optimally Scheduling Distributed Practice of Skills](https://arxiv.org/abs/1905.06873). EDM 2019.
5. Settles and Meeder (2016). [A Trainable Spaced Repetition Model for Language Learning](https://preview.aclanthology.org/naacl24-info/P16-1174). ACL 2016.
6. Piech et al. (2015). [Deep Knowledge Tracing](https://papers.neurips.cc/paper/2015/hash/bac9162b47c56fc8a4d2a519803d51b3-Abstract.html). NeurIPS 2015.
7. Gervet, Koedinger, Schneider and Mitchell (2020). [When is Deep Learning the Best Approach to Knowledge Tracing?](https://jedm.educationaldatamining.org/index.php/JEDM/article/view/451) *JEDM* 12(3).
8. Liu et al. (2022). [pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models](https://proceedings.neurips.cc/paper_files/paper/2022/hash/75ca2b23d9794f02a92449af65a57556-Abstract.html). NeurIPS 2022 Datasets and Benchmarks.
9. Liu et al. (2019). [Exploiting Cognitive Structure for Adaptive Learning](https://arxiv.org/abs/1905.12470) (CSEAL). KDD 2019.
10. Ghosh, Heffernan and Lan (2020). [Context-Aware Attentive Knowledge Tracing](https://arxiv.org/abs/2007.12324). KDD 2020.
11. Shen et al. (2021). [Learning Process-consistent Knowledge Tracing](https://dm.ustc.edu.cn/paper_pdf/2021/Shuanghong-Shen-KDD.pdf). KDD 2021.
12. Hidi and Renninger (2006). [The Four-Phase Model of Interest Development](https://works.swarthmore.edu/fac-education/12). *Educational Psychologist* 41(2).
13. Gruber, Gelman and Ranganath (2014). [States of Curiosity Modulate Hippocampus-Dependent Learning via the Dopaminergic Circuit](https://pmc.ncbi.nlm.nih.gov/articles/PMC4252494). *Neuron* 84(2).
