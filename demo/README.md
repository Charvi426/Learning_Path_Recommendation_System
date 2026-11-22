# Demo

The app has two parts:

- **My learning path**, the working prototype. A student gives their past scores and VLE clicks, chooses topics, and
  gets an ordered, explained path computed on the spot. It has its own model and none of the bugs listed below.
- **Past analysis (has bugs)**: a replay of the recorded v2/v3 research for 2014J test students (300 in each of the
  seven modules).

> **Known defects of the past analysis.** The recorded results contain a label leak, a wrong exam day and a v3 scoring run that did not
> follow the preregistration. The demo says so on every page. It shows how the system works. It is not evidence
> that the system works well. See `DECISIONS.md`, "Demo and report".

## Install

Run everything from the repository root, with Python 3.12.

```bash
python -m venv .venv-demo
# Windows PowerShell:   .venv-demo\Scripts\Activate.ps1
# macOS / Linux:        source .venv-demo/bin/activate
pip install -r requirements-demo.txt
```

## Build the data

Put the seven OULAD CSVs in `data/raw/`, then:

```bash
python demo/build_prototype.py     # the working prototype: about 4 minutes
python demo/build_demo_data.py     # the past-analysis pages: 1-5 minutes (optional)
```

For the second command:

- About 1-2 minutes when the intermediate data exists, a few minutes more from raw CSVs only.
- Reproduces the frozen fits, tunes nothing, and writes nothing under `results/` or `config/`.
- Stops if the recorded coefficients or the recorded A5 scores cannot be reproduced.

## Run

```bash
streamlit run demo/app.py
```

The app opens at http://localhost:8501. It reads only `demo/data/` and `demo/prototype/model.json`, and it never
fits a model.

## Using the prototype

1. In the sidebar, say where the data is:
   - **OULAD student ID**: any 2014J student. Only what was known before "today" is loaded, and you can edit it.
   - **Type in my data**: a score and submission day per topic, and rows of day, topic, clicks.
   - **Upload my files**: a VLE CSV (`id_site, date, sum_click` or `topic, day, clicks`), a score CSV
     (`id_assessment, date_submitted, score` or `topic, day, score`), or both. To get example files, choose a student
     ID and open *Download this history as files*.
2. Set the module and "Today is day". Days count from the start of the course.
3. Choose topics. Earlier topics they depend on are added unless you untick the box.
4. Read or download the path. It updates with every edit.

Nothing entered is written to disk.

### How a path is made

| Part | Kind | Detail |
| --- | --- | --- |
| Chance to submit and pass | learned | logistic regression per module on scores, clicks and deadlines before today |
| Still known | **assumed** | score halved every 60 days (adjustable) since submission or the last revision |
| Order | rules | due within 7 days and what it needs, then revision from the weakest, then course order |
| Suggested resources | rules | by the course plan, or by what the cohort used before today |

- The model is trained on presentations before 2014J. Validation AUC is 0.78-0.86 by module, against 0.52-0.65 for
  the calendar alone (`demo/prototype/model.json`). It has **not** been tested on 2014J, and the forgetting curve and
  the order have **not** been validated at all.
- It is not the v1-v3 model and not a correction of those results. Do not put its AUC in their tables.
- It only knows the seven OULAD modules and their 2014J calendars. A new course needs its own history to train on.

Screenshots for the report (optional): `python -m playwright install chromium`, then `python demo/screenshots.py`.

## Two-minute walkthrough for a live demo

Practise once. The app opens on *My learning path* with an example DDD student on day 100.

**0:00 - 0:15, the idea.**
"A course gives everyone the same order. This prototype takes a student's past scores and VLE clicks and builds a
path for the topics they choose."

**0:15 - 0:45, the path.**
Point at *Where you stand*, then at the steps. "Secure topics are skipped. Fading ones are revised, weakest first,
and upcoming ones show a predicted chance of passing, with resources to start from."

**0:45 - 1:05, choose topics.**
Clear the topics and select only topic 5. "It adds what topic 5 builds on." Untick the box to show the difference.

**1:05 - 1:35, it is live.**
Choose *Type in my data*. Enter score 88 on day 18 for topic 1: it is Fading and the predictions rise. Then add
activity day 98, topic 1, 12 clicks: topic 1 becomes Secure and leaves the path.

**1:35 - 2:00, honesty.**
Open *How this works, and how far to trust it*. "The forecast is trained on earlier cohorts and validated without
leakage. The forgetting rate is an assumption, and we have not tested whether following the path improves results.
Our earlier research evaluation had bugs: those pages are kept separate and labelled."

**Likely questions.**

- *Is this the model in the report?* No. The report's v1-v3 models have a label leak, so the prototype has its own.
- *Does it predict?* Yes, live: the chance of submitting and passing each topic not yet assessed.
- *Does interest count?* No. The analysis never showed that it helps.
- *Does it work for another course?* Not without retraining on that course's history.
- *Does the path improve marks?* Unknown. That needs a study with students.
