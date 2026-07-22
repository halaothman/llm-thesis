"""
Generate publication figures for IJATEE paper.
Output: docs/figures/fig1_*.png, fig2_*.png, fig3_*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
OUTPUTS = ROOT / "outputs"

HUMAN_METRICS_AR = [
    "الوضوح اللغوي",
    "الصياغة المنطقية",
    "الملاءمة",
    "جودة الخيارات",
    "الدقة",
]
HUMAN_METRICS_EN = [
    "Linguistic clarity",
    "Logical formulation",
    "Relevance",
    "Option quality",
    "Accuracy",
]
METRIC_COLUMNS = ["f1_score", "bert_score", "bleu", "precision", "recall"]
METRIC_LABELS = ["F1", "BERTScore", "BLEU", "Precision", "Recall"]

MODEL_DISPLAY = {"llama": "LLaMA", "qwen": "Qwen"}
METHOD_DISPLAY = {"vanilla": "Vanilla", "rag": "RAG"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def save_fig(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stem}.png"
    fig.savefig(path, facecolor="white")
    print(f"Saved {path}")


def fig1_system_architecture():
    """Fig. 1 — مخطط الميثودولوجي للأطروحة (Thesis Methodology Diagram)."""
    from generate_methodology_figure import build_methodology_figure, save_thesis_methodology_figure

    fig = build_methodology_figure()
    save_thesis_methodology_figure(fig)
    plt.close(fig)


def _human_averages():
    files = {
        "LLaMA Vanilla": ROOT / "llama_vanilla_human_evaluation.csv",
        "LLaMA RAG": ROOT / "llama_rag_human_evaluation.csv",
        "Qwen Vanilla": ROOT / "qwen_vanilla_human_evaluation.csv",
        "Qwen RAG": ROOT / "qwen_rag_human_evaluation.csv",
    }
    rows = []
    for group, path in files.items():
        df = pd.read_csv(path, encoding="utf-8-sig")
        avg = df[df["#"].astype(str) == "المتوسط"].iloc[0]
        for ar, en in zip(HUMAN_METRICS_AR, HUMAN_METRICS_EN):
            rows.append(
                {
                    "group": group,
                    "metric": en,
                    "score": float(avg[ar]),
                }
            )
    return pd.DataFrame(rows)


def fig2_human_evaluation():
    """Fig. 2 — human ratings Vanilla vs RAG."""
    df = _human_averages()
    models = ["LLaMA", "Qwen"]
    metrics = HUMAN_METRICS_EN
    x = np.arange(len(metrics))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    colors_v = "#4C78A8"
    colors_r = "#F58518"

    for ax, model in zip(axes, models):
        v_scores = [
            df[(df["group"] == f"{model} Vanilla") & (df["metric"] == m)]["score"].iloc[0]
            for m in metrics
        ]
        r_scores = [
            df[(df["group"] == f"{model} RAG") & (df["metric"] == m)]["score"].iloc[0]
            for m in metrics
        ]
        ax.bar(x - width / 2, v_scores, width, label="Vanilla", color=colors_v, edgecolor="white")
        ax.bar(x + width / 2, r_scores, width, label="RAG", color=colors_r, edgecolor="white")
        ax.set_title(model, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=25, ha="right")
        ax.set_ylim(1, 5.2)
        ax.set_ylabel("Mean rating (1–5)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "Fig. 2. Human evaluation: Vanilla versus RAG (mean scores)",
        fontsize=11,
        fontweight="bold",
        y=1.03,
    )
    save_fig(fig, "fig2_human_evaluation")
    plt.close(fig)


def parse_question_file(path: Path):
    stem = path.stem
    version = "after" if "_new" in stem else "before"
    stem_clean = stem.replace("_new", "")
    parts = stem_clean.split("_")
    model, method = "unknown", "unknown"
    if len(parts) >= 4 and parts[0] == "questions":
        model = parts[1]
        method = parts[2]
    return MODEL_DISPLAY.get(model, model), METHOD_DISPLAY.get(method, method), version


def load_automatic_metrics():
    rows = []
    if not OUTPUTS.exists():
        return pd.DataFrame()
    for path in OUTPUTS.rglob("questions_*.json"):
        if "UPDATED" in path.stem.upper():
            continue
        model, method, version = parse_question_file(path)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        for q in data.get("questions", []):
            m = q.get("metrics") or {}
            ppl = m.get("perplexity")
            if ppl in (0.0, 1000.0, None):
                continue
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "version": version,
                    "f1_score": m.get("f1_score"),
                    "bert_score": m.get("bert_score"),
                    "bleu": m.get("bleu"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "perplexity": ppl,
                }
            )
    return pd.DataFrame(rows)


def fig3_automatic_metrics():
    """Fig. 3 — automatic metrics distributions (BERTScore and F1)."""
    df = load_automatic_metrics()
    if df.empty:
        print("No automatic metrics data; skipping fig3")
        return

    plot_metrics = ["bert_score", "f1_score"]
    plot_labels = ["BERTScore", "F1 score"]
    groups = []
    for _, row in df.iterrows():
        for col, label in zip(plot_metrics, plot_labels):
            val = row[col]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            groups.append(
                {
                    "value": float(val),
                    "metric": label,
                    "cell": f"{row['model']}\n{row['method']}",
                }
            )
    plot_df = pd.DataFrame(groups)
    if plot_df.empty:
        print("No valid metric values; skipping fig3")
        return

    order = [
        "LLaMA\nVanilla",
        "LLaMA\nRAG",
        "Qwen\nVanilla",
        "Qwen\nRAG",
    ]
    colors = {"BERTScore": "#72B7B2", "F1 score": "#B279A2"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)
    for ax, metric in zip(axes, plot_labels):
        sub = plot_df[plot_df["metric"] == metric]
        data = [sub[sub["cell"] == g]["value"].values for g in order]
        bp = ax.boxplot(
            data,
            labels=order,
            patch_artist=True,
            widths=0.55,
            showfliers=True,
            medianprops=dict(color="#222222", linewidth=1.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[metric])
            patch.set_alpha(0.75)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(
        "Fig. 3. Distribution of automatic quality metrics by model and method",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    save_fig(fig, "fig3_automatic_metrics")
    plt.close(fig)


def fig4_perplexity():
    """Fig. 4 — perplexity distributions (lower is better)."""
    df = load_automatic_metrics()
    if df.empty or "perplexity" not in df.columns:
        print("No perplexity data; skipping fig4")
        return

    order = [
        "LLaMA\nVanilla",
        "LLaMA\nRAG",
        "Qwen\nVanilla",
        "Qwen\nRAG",
    ]
    plot_df = df.copy()
    plot_df["cell"] = plot_df["model"] + "\n" + plot_df["method"]
    plot_df = plot_df[plot_df["cell"].isin(order)]
    plot_df = plot_df.dropna(subset=["perplexity"])
    if plot_df.empty:
        print("No valid perplexity values; skipping fig4")
        return

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    data = [plot_df[plot_df["cell"] == g]["perplexity"].values for g in order]
    bp = ax.boxplot(
        data,
        labels=order,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        medianprops=dict(color="#222222", linewidth=1.5),
    )
    colors = ["#4C78A8", "#F58518", "#72B7B2", "#E45756"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.72)

    ax.set_title("Perplexity (lower is better)", fontweight="bold")
    ax.set_ylabel("Perplexity")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=8)
    ax.text(
        0.5,
        -0.22,
        "Invalid values (0.0 and 1000.0) were excluded before plotting.",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )

    fig.suptitle(
        "Fig. 4. Distribution of perplexity by model and method",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    save_fig(fig, "fig4_perplexity")
    plt.close(fig)


def main():
    try:
        fig1_system_architecture()
        fig2_human_evaluation()
        fig3_automatic_metrics()
        fig4_perplexity()
        print(f"\nAll figures written to {OUT}")
    except Exception as e:
        raise SystemExit(f"Failed: {e}") from e


if __name__ == "__main__":
    main()
