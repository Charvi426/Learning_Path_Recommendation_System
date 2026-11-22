"""Vector PDF figures for one IEEE column (3.5 in wide, fonts >= 8 pt).

  python report/make_figures.py

Reads frozen results and, for the example forgetting curve, demo/data/ (python demo/build_demo_data.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "demo"))
import results_catalog as RC  # noqa: E402

FIG = HERE / "figures"
W = 3.5
BLUE, ORANGE, GREEN, RED, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#777777"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"], "pdf.fonttype": 42,
                     "mathtext.fontset": "stix", "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def pipeline():
    fig, ax = plt.subplots(figsize=(W, 2.3))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")

    def box(x, y, w, text, colour="#EAF1F8"):
        ax.add_patch(FancyBboxPatch((x, y), w, 16, boxstyle="round,pad=0.4", fc=colour, ec=GREY, lw=0.7))
        ax.text(x + w / 2, y + 8, text, ha="center", va="center")

    def arrow(a, b):
        ax.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.8))

    box(2, 80, 44, "OULAD\nscores and clicks")
    box(54, 80, 44, "Topics\n(assessment windows)")
    box(54, 54, 44, "Factors: engagement,\nability, interest, prereqs.")
    box(2, 54, 44, "Knowledge model\n$M$, $R$, $K$")
    box(2, 28, 44, "Candidates\nnew + fading topics")
    box(54, 28, 44, "Weighted score")
    box(54, 2, 44, "Ordered path\nwith explanations", "#FBEEDC")
    box(2, 2, 44, "Pre-exam planner (v3)", "#FBEEDC")
    arrow((46.5, 88), (53.5, 88))
    arrow((76, 79.5), (76, 70.5))
    arrow((53.5, 62), (46.5, 62))
    arrow((24, 53.5), (24, 44.5))
    arrow((46.5, 36), (53.5, 36))
    arrow((76, 27.5), (76, 18.5))
    ax.plot([2, 0.3, 0.3], [62, 62, 10], color=GREY, lw=0.8)
    arrow((0.3, 10), (1.9, 10))
    save(fig, "pipeline")


def forgetting():
    data = ROOT / "demo" / "data"
    if not (data / "meta.parquet").exists():
        raise SystemExit("demo/data is missing: run  python demo/build_demo_data.py  first")
    sid = pd.read_parquet(data / "meta.parquet")["default_student"].iloc[0]
    topic = 2
    c = pd.read_parquet(data / f"curves_{sid[:3]}.parquet")
    c = c[(c["student"] == sid) & (c["topic"] == topic)].sort_values("day")
    e = pd.read_parquet(data / f"events_{sid[:3]}.parquet")
    e = e[(e["student"] == sid) & (e["topic"] == topic) & (e["kind"] == "Revision")]
    p = pd.read_parquet(data / "plans.parquet")
    sat = p.loc[p["student"] == sid, "day_sat"].iloc[0]
    fig, ax = plt.subplots(2, 1, figsize=(W, 2.55), sharex=True, sharey=True, constrained_layout=True)
    for a, m, colour, title in ((ax[0], "v2", BLUE, "v2: half-life, own pages reset the clock"),
                                (ax[1], "v3", RED, "v3: activation over attributed practice")):
        a.plot(c["day"], c[m], color=colour, lw=1.1)
        r = e[e["model"] == m].assign(day=lambda x: x["day"] + 1).merge(c[["day", m]], on="day")
        a.scatter(r["day"], r[m], marker="^", s=16, color="black", zorder=3, label="revision")
        a.axvline(sat, color=GREY, ls="--", lw=0.8, label="exam sat")
        a.set_title(title, loc="left")
        a.set_ylim(0, 1)
        a.set_ylabel("$K = MR$")
    ax[0].legend(loc="lower left", frameon=False, ncol=2)
    ax[1].set_xlabel("day of the course")
    save(fig, "forgetting")


def exam_diagnosis():
    d = pd.read_csv(ROOT / "results/v2/exam_check/delta_and_revision_at_exam_val.csv")
    fig, ax = plt.subplots(2, 1, figsize=(W, 2.45), sharex=True, constrained_layout=True)
    ax[0].bar(d["topic"], d["delta_median"], color=BLUE, width=0.6)
    ax[0].vlines(d["topic"], d["delta_p10"], d["delta_p90"], color="black", lw=0.8)
    ax[0].set_ylabel(r"$\Delta$ (days)")
    ax[0].set_title("Days since last practice (median, 10th to 90th pct.)", loc="left")
    ax[1].bar(d["topic"], 100 * d["share_reset_in_revision_period"], color=ORANGE, width=0.6)
    ax[1].set_ylim(0, 100)
    ax[1].set_ylabel("students (%)")
    ax[1].set_title("Clock reset in the revision period", loc="left")
    ax[1].set_xlabel("DDD topic")
    ax[1].set_xticks(d["topic"])
    save(fig, "exam_diagnosis")


def ndcg_holdout():
    t = RC.tables()["rank_holdout"]
    rows = t.rows[::-1]
    fig, ax = plt.subplots(figsize=(W, 2.65), constrained_layout=True)
    colour = {"v1": GREY, "v2": BLUE, "v3": RED}
    for y, (name, v, _, _) in enumerate(rows):
        e, lo, hi = RC.read(v.est), RC.read(v.lo), RC.read(v.hi)
        ax.errorbar(e, y, xerr=[[e - lo], [hi - e]], fmt="o", ms=3.5, capsize=2, color=colour[name[:2]],
                    mfc="white" if "raw K" in name else colour[name[:2]])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("holdout NDCG@5, combined relevance (95% CI)")
    save(fig, "ndcg_holdout")


def criteria():
    c = RC.criteria()
    fig, ax = plt.subplots(3, 1, figsize=(W, 3.05), constrained_layout=True)

    def get(v):
        return [RC.read(x) for x in (v.est, v.lo, v.hi) if x is not None]

    def status(a, crit, text):
        met = bool(RC.read(crit["met"]))
        a.set_title(f"({crit['id']}) {text}", loc="left")
        a.text(1, 1.02, "met*" if met else "not met*", transform=a.transAxes, ha="right", va="bottom",
               color=GREEN if met else RED, fontweight="bold")

    for a, crit, limit, text in ((ax[0], c[0], -0.002, "AUC, v3 minus v2"), (ax[1], c[1], 0.0, "CCC exam gap, improvement")):
        e, lo, hi = get(crit["effect"])
        a.errorbar(e, 0, xerr=[[e - lo], [hi - e]], fmt="o", ms=4, capsize=3, color="black")
        a.axvline(limit, color=RED, ls="--", lw=0.8)
        a.set_yticks([])
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        a.set_xlim(min(limit, lo) - 0.12 * (hi - limit), hi + 0.12 * (hi - limit))
        status(a, crit, text)
    for y, (key, colour) in enumerate((("v3", RED), ("v2", BLUE))):
        e, lo, hi = get(c[2][key])
        ax[2].errorbar(e, y, xerr=[[e - lo], [hi - e]], fmt="o", ms=4, capsize=3, color=colour)
    ax[2].set_yticks([0, 1], ["v3 (raw K)", "v2"])
    ax[2].set_ylim(-0.6, 1.6)
    status(ax[2], c[2], "NDCG@5, A5")
    fig.supxlabel("* as recorded; none is a reliable test (see text)", fontsize=8)
    save(fig, "criteria")


def demo_crop():
    """The readable part of the student-view screenshot (no sidebar, banner kept)."""
    src = FIG / "demo_student.png"
    if not src.exists():
        print("no demo screenshot: run python demo/screenshots.py")
        return
    im = plt.imread(src)
    h, w = im.shape[:2]
    plt.imsave(FIG / "demo_crop.png", im[int(0.15 * h):int(0.70 * h), int(0.29 * w):int(0.95 * w)].copy(order="C"))


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (pipeline, forgetting, exam_diagnosis, ndcg_holdout, criteria, demo_crop):
        f()
    print("5 vector figures and the screenshot crop -> report/figures/")
