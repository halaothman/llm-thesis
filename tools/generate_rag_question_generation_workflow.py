"""
RAG question generation workflow — thesis figure (indexing + inference).

Outputs:
  docs/figures/rag_question_generation_workflow.svg
  docs/figures/rag_question_generation_workflow.png
  docs/figures/rag_question_generation_workflow.svg
  docs/figures/rag_question_generation_workflow.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
PRES_OUT = OUT

BORDER = "#1B4F72"
TEXT = "#000000"
ARROW = "#1B4F72"
FILL = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
    }
)

INDEX_STEPS = [
    "Input Document",
    "Extract Text",
    "Text Cleaning",
    "Text Chunking",
    "Generate Embeddings\n(AraBERT Embedding Model)",
    "Store Embeddings in FAISS\nVector Database",
]

INFERENCE_STEPS = [
    "User Uploads a Document",
    "Generate Query Embedding",
    "Similarity Search in FAISS",
    "Retrieve Top-k Chunks",
    "Cross-Encoder Re-ranking",
    "Select Most Relevant Context",
    "Construct Prompt\n(Retrieved Context + Question\nGeneration Instructions)",
    "Large Language Model\n(Qwen2.5-7B or Llama3.2-3B)",
    "Generate Questions",
    "Save Generated Questions as JSON",
]


def _box(ax, x, y, w, h):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.05",
            linewidth=0.9,
            edgecolor=BORDER,
            facecolor=FILL,
            zorder=2,
        )
    )


def _v_arrow(ax, x, y1, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x, y1),
            (x, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.85,
            color=ARROW,
            shrinkA=3,
            shrinkB=3,
            zorder=1,
        )
    )


def _draw_step(ax, x, y, w, h, label, font_size):
    cx = x + w / 2
    _box(ax, x, y, w, h)
    ax.text(
        cx,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=font_size,
        color=TEXT,
        zorder=3,
        linespacing=1.15,
    )
    return y, y + h


def build_rag_workflow_figure(*, presentation: bool = False) -> plt.Figure:
    if presentation:
        fig_w, fig_h = 10.5, 18.0
        box_w, box_h = 7.8, 0.66
        v_gap, phase_gap = 0.24, 0.50
        title_size, phase_size, step_size = 15, 11, 10.0
        x0 = (fig_w - box_w) / 2
        y = fig_h - 0.75
    else:
        fig_w, fig_h = 7.2, 15.6
        box_w, box_h = 5.6, 0.54
        v_gap, phase_gap = 0.20, 0.40
        title_size, phase_size, step_size = 12.5, 9.5, 8.4
        x0 = (fig_w - box_w) / 2
        y = fig_h - 0.55

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.set_facecolor("white")

    cx = fig_w / 2
    ax.text(
        cx,
        y + 0.18,
        "RAG Question Generation Workflow",
        ha="center",
        va="bottom",
        fontsize=title_size,
        fontweight="bold",
        color=TEXT,
    )
    y -= 0.55

    def draw_phase(title, steps, *, continue_from=None, extra_gap_before=0.0):
        nonlocal y
        y -= extra_gap_before
        ax.text(
            cx,
            y,
            title,
            ha="center",
            va="bottom",
            fontsize=phase_size,
            fontweight="bold",
            color=TEXT,
        )
        y -= 0.28
        prev_bottom = continue_from
        for label in steps:
            y -= box_h
            bottom, top = _draw_step(ax, x0, y, box_w, box_h, label, step_size)
            if prev_bottom is not None:
                _v_arrow(ax, cx, prev_bottom, top)
            prev_bottom = bottom
            y -= v_gap
        return prev_bottom

    last = draw_phase("Knowledge Base Indexing", INDEX_STEPS)
    draw_phase("RAG Inference Pipeline", INFERENCE_STEPS, continue_from=last, extra_gap_before=phase_gap)

    return fig


def save_figure(fig: plt.Figure, stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, facecolor="white", edgecolor="none", format=ext)
        print(f"Saved {path}")


def main() -> None:
    fig = build_rag_workflow_figure(presentation=False)
    save_figure(fig, "rag_question_generation_workflow", OUT)
    plt.close(fig)

    fig_pres = build_rag_workflow_figure(presentation=True)
    save_figure(fig_pres, "rag_question_generation_workflow", PRES_OUT)
    plt.close(fig_pres)


if __name__ == "__main__":
    main()
