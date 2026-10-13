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
# Measurements of the idle server
IDLE_DIR = ROOT_DIR / "results/idle_consumption"

# Number of repetitions of each model run used by the analysis, per execution. The mistral-bf16
# execution was stopped before its 10 repetitions, and its results keep 5 repetitions of each model run.
REPETITIONS = {"main": 10, "mistral-bf16": 5}

# Number of questions of the benchmark sample and its chance level (mean of 1 / number of options)
QUESTIONS = 372

# Energy budgets (kJ of GPU energy per question) and accuracy targets used to compare models at a fixed level
BUDGETS_KJ = [0.5, 1, 2, 5]
ACCURACY_TARGETS = [0.4, 0.5, 0.6, 0.7]
# One shade per level, from the lowest to the highest: orange for the energy budgets, blue for the accuracy targets
BUDGET_COLORS = ["#f2b44a", "#e08e00", "#b06c00", "#784800"]
TARGET_COLORS = ["#9cc3d5", "#5b9bb8", "#27708f", "#003f5c"]

CATEGORY_ORDER = ["Tiny/Mobile", "Small/Edge", "Standard/Desktop"]
CATEGORY_COLORS = {
    "Tiny/Mobile": "#3B82F6",
    "Small/Edge": "#14B8A6",
    "Standard/Desktop": "#F59E0B",
}
MODE_COLORS = {"Instruct": "#003f5c", "Reasoning": "#bb4e99"}
ARCHITECTURE_COLORS = {"Dense": "#003f5c", "Sparsely activated": "#bb4e99"}
# The three factors of the intelligence efficiency: accuracy / (response length x energy per token)
FACTOR_COLORS = {"Accuracy": "#003f5c", "Response length": "#bb4e99", "Energy per token": "#ffa600"}

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


def load_run_table(execution):
    """Run table of an execution of the experiment, with the runs that were executed."""
    path = RESULTS_DIRS[execution] / "run_table.csv"
    done = pd.read_csv(path, usecols=["__done"])["__done"] == "DONE"
    return pd.read_csv(path, skiprows=lambda row: row > 0 and not done[row - 1])


def load_model_runs(include_superseded=False):
    """One row per model run. Runs that are not part of the study (the 8-bit Ministral 3 runs)
    are left out unless asked for."""
    model_runs = pd.read_csv(DATA_DIR / "model_runs.csv", parse_dates=["release_date"])
    if not include_superseded:
        model_runs = model_runs[model_runs["in_study"]]
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


def format_year_axis(ax):
    """Date axis with one tick per year, for narrow panels."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def plain_log_axis(axis, subs=(1, 2, 5)):
    """Log axis with ticks at 1, 2 and 5 of each decade (or the given multiples), written as plain numbers."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    axis.set_major_locator(LogLocator(base=10, subs=subs))
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}" if value >= 1 else f"{value:g}"))
    axis.set_minor_formatter(NullFormatter())


def percent_axis(axis):
    """Axis of an accuracy, written as a percentage."""
    from matplotlib.ticker import PercentFormatter

    axis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))


def category_panels(nrows=1, figsize=(10, 4), sharex=True, sharey="row"):
    """A grid with one column of panels per size category."""
    fig, axes = plt.subplots(nrows, len(CATEGORY_ORDER), figsize=figsize, sharex=sharex, sharey=sharey, squeeze=False)
    for ax, category in zip(axes[0], CATEGORY_ORDER):
        ax.set_title(category)
    return fig, axes


def lineages(model_runs):
    """Groups of instruct runs of the same family, size category and architecture, ordered by release date.
    Yields (family, category, architecture, runs) for the groups with more than one release."""
    instruct = model_runs[model_runs["mode"] == "Instruct"]
    for (category, family, architecture), runs in instruct.groupby(["category", "family", "architecture"]):
        if len(runs) > 1:
            yield family, category, architecture, runs.sort_values("release_date")


# Ticks of the charts that show a change as a factor, on a log scale
LOG_TICKS = [1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8]
LOG_TICK_LABELS = ["÷8", "÷4", "÷2", "=", "×2", "×4", "×8"]


def with_log_factors(model_runs):
    """Adds to each model run the mean, over its repetitions, of the logarithm of the intelligence efficiency and
    of its three factors: accuracy, generated tokens per question and GPU energy per generated token.
    With these means, the contributions of the three factors add up exactly to the change of efficiency."""
    repetitions = load_repetitions(model_runs)
    logs = pd.DataFrame({
        "log_accuracy": np.log(repetitions["avg_accuracy"]),
        "log_tokens": np.log(repetitions["avg_generated_tokens_per_inference"]),
        "log_energy_per_token": np.log(repetitions["gpu_joules_per_generated_token"]),
        "log_efficiency": np.log(repetitions["accuracy_per_gpu_joule"]),
    }).groupby(repetitions["model_run"]).mean()
    return model_runs.merge(logs, left_on="model_run", right_index=True)


def contributions(before, after):
    """Contribution of each factor to the change of intelligence efficiency between two model runs, on a log scale.
    before and after can also be groups of model runs, for which geometric means are used."""
    return {
        "Accuracy": np.mean(after["log_accuracy"]) - np.mean(before["log_accuracy"]),
        "Response length": -(np.mean(after["log_tokens"]) - np.mean(before["log_tokens"])),
        "Energy per token": -(np.mean(after["log_energy_per_token"]) - np.mean(before["log_energy_per_token"])),
        "Net": np.mean(after["log_efficiency"]) - np.mean(before["log_efficiency"]),
    }


def stacked_contributions(ax, table, horizontal=False):
    """Stacked bars of the three contributions of each row of the table, with a diamond at their sum.
    Positive contributions are stacked on one side of zero and negative ones on the other."""
    positions = np.arange(len(table))
    positive, negative = np.zeros(len(table)), np.zeros(len(table))
    for factor, color in FACTOR_COLORS.items():
        values = table[factor].to_numpy()
        start = np.where(values >= 0, positive, negative)
        if horizontal:
            ax.barh(positions, values, left=start, height=0.6, color=color, edgecolor="w", linewidth=1)
        else:
            ax.bar(positions, values, bottom=start, width=0.6, color=color, edgecolor="w", linewidth=1)
        positive += np.where(values >= 0, values, 0)
        negative += np.where(values < 0, values, 0)
    if horizontal:
        ax.scatter(table["Net"], positions, marker="D", s=26, color="0.1", zorder=3)
        ax.axvline(0, color="0.3", linewidth=1)
        ax.set_xticks(np.log(LOG_TICKS), labels=LOG_TICK_LABELS)
    else:
        ax.scatter(positions, table["Net"], marker="D", s=26, color="0.1", zorder=3)
        ax.axhline(0, color="0.3", linewidth=1)
        ax.set_yticks(np.log(LOG_TICKS), labels=LOG_TICK_LABELS)
    return positions


def factor_handles():
    """Legend entries for the charts of stacked_contributions."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=color, label=factor) for factor, color in FACTOR_COLORS.items()]
    handles.append(Line2D([], [], marker="D", linestyle="", markersize=5, color="0.1", label="Change of intelligence efficiency"))
    return handles


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
