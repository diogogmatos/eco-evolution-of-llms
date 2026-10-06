# Shared code of the analysis notebooks: paths, loading of the analysis tables and chart style.
# The tables are built by 00-dataset.ipynb. The chart style and the colours follow
# model-selection/model-selection.ipynb, so the figures of the thesis look the same.

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "analysis/data"
CHART_DIR = ROOT_DIR / "analysis/charts"

# Results of each execution of the experiment
RESULTS_DIRS = {
    "main": ROOT_DIR / "results/main",
    # 16-bit runs of the Ministral 3 instruct models, which ran with 8-bit weights in the main execution
    "mistral-bf16": ROOT_DIR / "results/mistral-bf16",
}

# Families left out of the analysis until the mistral-bf16 execution is finished.
# Empty this list to include them again.
PENDING_FAMILIES = ["Mistral"]

# Number of questions of the benchmark sample and its chance level (mean of 1 / number of options)
QUESTIONS = 372

CATEGORY_ORDER = ["Tiny/Mobile", "Small/Edge", "Standard/Desktop"]
CATEGORY_COLORS = {
    "Tiny/Mobile": "#3B82F6",
    "Small/Edge": "#14B8A6",
    "Standard/Desktop": "#F59E0B",
}
MODE_COLORS = {"Instruct": "#003f5c", "Reasoning": "#bb4e99"}
ARCHITECTURE_COLORS = {"Dense": "#003f5c", "Sparsely activated": "#bb4e99"}

# Colour is taken by the size category, so families are told apart by the shape of the marker
FAMILY_ORDER = ["Llama", "Qwen", "Gemma", "DeepSeek", "Phi", "Mistral"]
FAMILY_MARKERS = {
    "Llama": "^",
    "Qwen": "o",
    "Gemma": "s",
    "DeepSeek": "D",
    "Phi": "v",
    "Mistral": "P",
}

# Same scatter marks as the model-selection notebook
SCATTER_STYLE = dict(s=60, alpha=0.85, linewidth=0.4)

# Figures are sized for the width of a page of the thesis, so that their text stays readable there
plt.rcParams.update({
    "figure.figsize": (9, 5.5),
    "font.size": 11,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "0.9",
    "grid.linewidth": 0.6,
    "svg.hashsalt": "analysis",
})


def load_model_runs(include_pending=False, include_superseded=False):
    """One row per model run. Runs that are not part of the study (the 8-bit Ministral 3 runs)
    and families that are pending are left out unless asked for."""
    model_runs = pd.read_csv(DATA_DIR / "model_runs.csv", parse_dates=["release_date"])
    if not include_superseded:
        model_runs = model_runs[model_runs["in_study"]]
    if not include_pending:
        model_runs = model_runs[~model_runs["family"].isin(PENDING_FAMILIES)]
    return model_runs.reset_index(drop=True)


def load_repetitions(model_runs=None):
    """One row per repetition of a model run."""
    repetitions = pd.read_csv(DATA_DIR / "repetitions.csv")
    return _keep(repetitions, model_runs)


def load_questions(model_runs=None):
    """One row per model run and question, summarizing its 10 repetitions."""
    questions = pd.read_csv(DATA_DIR / "questions.csv")
    return _keep(questions, model_runs)


def load_inferences(model_runs=None):
    """One row per inference."""
    inferences = pd.read_csv(DATA_DIR / "inferences.csv")
    return _keep(inferences, model_runs)


def _keep(table, model_runs):
    if model_runs is None:
        model_runs = load_model_runs()
    return table[table["model_run"].isin(model_runs["model_run"])].reset_index(drop=True)


def short_name(model_run):
    """Name of a model run without its organization, and with the operating mode abbreviated."""
    name = model_run.split("/")[-1]
    return name.replace(" (Thinking Mode)", " (R)").replace(" (Instruct Mode)", " (I)")


def scatter_model_runs(ax, model_runs, x, y, color=None, **kwargs):
    """Scatter of model runs: colour for the size category, marker shape for the family,
    filled markers for instruct runs and hollow ones for reasoning runs."""
    style = {**SCATTER_STYLE, **kwargs}
    for (family, mode), group in model_runs.groupby(["family", "mode"]):
        colors = group["category"].map(CATEGORY_COLORS) if color is None else color
        if mode == "Instruct":
            ax.scatter(group[x], group[y], c=colors, marker=FAMILY_MARKERS[family], edgecolors="w", **style)
        else:
            hollow = {**style, "linewidth": 1.5}
            ax.scatter(group[x], group[y], facecolors="none", edgecolors=colors, marker=FAMILY_MARKERS[family], **hollow)


def legend_handles(categories=True, families=None, modes=True):
    """Legend entries for the encodings of scatter_model_runs."""
    from matplotlib.lines import Line2D

    handles = []
    if categories:
        for category in CATEGORY_ORDER:
            handles.append(Line2D([], [], marker="o", linestyle="", markersize=8, color=CATEGORY_COLORS[category], label=category))
    for family in families if families is not None else []:
        handles.append(Line2D([], [], marker=FAMILY_MARKERS[family], linestyle="", markersize=8, color="0.35", label=family))
    if modes:
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=8, color="0.35", label="Instruct (filled)"))
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor="none", markeredgewidth=1.5, color="0.35", label="Reasoning (hollow)"))
    return handles


def format_date_axis(ax):
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def save_chart(fig, name):
    CHART_DIR.mkdir(exist_ok=True)
    fig.savefig(CHART_DIR / f"{name}.svg", format="svg", bbox_inches="tight", metadata={"Date": None})


def pareto_front(model_runs, cost, benefit):
    """Model runs for which no other run has both a lower cost and a higher benefit, by increasing cost."""
    front, best = [], -np.inf
    for index, row in model_runs.sort_values([cost, benefit], ascending=[True, False]).iterrows():
        if row[benefit] > best:
            front.append(index)
            best = row[benefit]
    return model_runs.loc[front]


def chance_level(questions):
    """Expected accuracy of answering at random: the mean of 1 / number of options of the questions."""
    return (1 / questions.drop_duplicates("question_id")["options"]).mean()


def bootstrap_questions(questions, columns, n_resamples=2000, seed=12345):
    """Bootstrap over the questions of one model run, stratified by subject. Returns, for each
    column, the mean of that column in every resample. The questions are sorted first, so two
    model runs resampled with the same seed get the same questions in each resample."""
    questions = questions.sort_values("question_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    draws = []
    for _, group in questions.groupby("subject", sort=True):
        positions = group.index.to_numpy()
        draws.append(positions[rng.integers(0, len(positions), size=(n_resamples, len(positions)))])
    sample = np.concatenate(draws, axis=1)
    return {column: questions[column].to_numpy()[sample].mean(axis=1) for column in columns}
