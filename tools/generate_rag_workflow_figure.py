"""
Generate RAG workflow diagram — academic layout (matplotlib only).

Outputs:
  docs/figures/rag_workflow_funnel.png
  docs/figures/rag_workflow_funnel.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
PRES_OUT = OUT

AC = {
    "border": "#2C3E50",
    "arrow": "#566573",
    "title": "#1A1A1A",
    "text": "#333333",
    "muted": "#5D6D7E",
    "band": "#F8F9FA",
    "band_edge": "#E5E8E8",
    "input": "#E8F4FC",
    "search": "#EBF5FB",
    "retrieval": "#E8F6F3",
    "generation": "#F4F6F7",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)

STEPS = [
    (
        "Query embedding",
        "Encode uploaded passage\nas a dense vector.",
        "E5-large-v2 or AraBERT",
        AC["input"],
    ),
    (
        "Similarity search",
        "Compare query vector against\nchunk embeddings in FAISS.",
        "Top-K retrieval; threshold filter",
        AC["search"],
    ),
    (
        "Context retrieval",
        "Return ranked passages\nwith source metadata.",
        "Improved: rerank 10 → select 5",
        AC["retrieval"],
    ),
    (
        "Question generation",
        "LLM produces structured\nMCQ and true/false items.",
        "LLaMA 3.2 3B or Qwen 2.5 7B",
        AC["generation"],
    ),
]


def _box(ax, x, y, w, h, fill):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.05",
            linewidth=0.9,
            edgecolor=AC["border"],
            facecolor=fill,
            zorder=2,
        )
    )


def _arrow(ax, x1, y1, x2, y2, style="-|>", connection="arc3,rad=0"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=10,
            linewidth=0.9,
            color=AC["arrow"],
            connectionstyle=connection,
            zorder=1,
        )
    )


def _draw_step(ax, idx, x, y, w, h, title, desc, note, fill):
    cx = x + w / 2
    _box(ax, x, y, w, h, fill)
    ax.text(cx, y + h - 0.28, f"{idx}. {title}", ha="center", va="top", fontsize=9, fontweight="bold", color=AC["text"], zorder=3)
    ax.text(cx, y + h / 2 - 0.05, desc, ha="center", va="center", fontsize=7.4, color=AC["muted"], zorder=3, linespacing=1.15)
    ax.text(cx, y + 0.28, note, ha="center", va="bottom", fontsize=6.8, color=AC["muted"], style="italic", zorder=3)
    return cx, y + h / 2


def build_rag_workflow_figure() -> plt.Figure:
    fig_w, fig_h = 10.5, 6.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    ax.text(
        fig_w / 2,
        6.35,
        "RAG workflow for Arabic question generation",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=AC["title"],
    )

    box_w = 3.85
    box_h = 1.55
    gap_x = 0.55
    gap_y = 0.75
    x_left = (fig_w - (2 * box_w + gap_x)) / 2
    x_right = x_left + box_w + gap_x
    y_top = 3.85
    y_bottom = 1.55

    ax.add_patch(
        Rectangle((0.35, y_bottom - 0.18), fig_w - 0.7, 2 * box_h + gap_y + 0.36, facecolor=AC["band"], edgecolor=AC["band_edge"], linewidth=0.6, zorder=0)
    )

    positions = {}
    positions[0] = _draw_step(ax, 1, x_left, y_top, box_w, box_h, *STEPS[0])
    positions[1] = _draw_step(ax, 2, x_right, y_top, box_w, box_h, *STEPS[1])
    positions[2] = _draw_step(ax, 3, x_left, y_bottom, box_w, box_h, *STEPS[2])
    positions[3] = _draw_step(ax, 4, x_right, y_bottom, box_w, box_h, *STEPS[3])

    cx0, cy0 = positions[0]
    cx1, cy1 = positions[1]
    cx2, cy2 = positions[2]
    cx3, cy3 = positions[3]

    _arrow(ax, x_left + box_w + 0.03, cy0, x_right - 0.03, cy1)
    _arrow(ax, cx1, y_top, cx1, y_top - 0.25, style="-")
    _arrow(ax, cx1, y_top - 0.25, cx2, y_top - 0.25, style="-")
    _arrow(ax, cx2, y_top - 0.25, cx2, y_bottom + box_h + 0.02)
    _arrow(ax, x_left + box_w + 0.03, cy2, x_right - 0.03, cy3)

    ax.text(
        fig_w / 2,
        0.55,
        "Vanilla path skips retrieval and uses the uploaded passage only.",
        ha="center",
        va="center",
        fontsize=7.5,
        color=AC["muted"],
        style="italic",
    )

    return fig


def save_figure(fig: plt.Figure, stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    try:
        fig.savefig(path, facecolor="white", edgecolor="none")
        print(f"Saved {path}")
    except PermissionError:
        print(f"Skipped (file locked): {path}")


def main() -> None:
    fig = build_rag_workflow_figure()
    save_figure(fig, "rag_workflow_funnel", OUT)
    save_figure(fig, "rag_workflow_funnel", PRES_OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
