"""
Generate system flow diagram — academic two-row layout (matplotlib only).

Outputs:
  docs/figures/system_flow_question_generation.png
  docs/figures/system_flow_question_generation.png
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
    "fills": [
        "#E8F4FC",
        "#EBF5FB",
        "#F4F6F7",
        "#F8F9FA",
        "#E8F6F3",
        "#F5F5F5",
        "#F0F3F4",
    ],
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
    ("Text processing", "Extract content from\nPDF, DOCX, TXT."),
    ("Text chunking", "500-char segments;\n100-char overlap."),
    ("Embedding", "E5-large-v2\nvectorization."),
    ("Indexing", "FAISS index\nand metadata."),
    ("Question generation", "LLaMA / Qwen;\nVanilla or RAG."),
    ("User interaction", "Quiz UI with\nauto-correction."),
    ("Evaluation", "Automatic metrics\nand human ratings."),
]

ROW1_IDX = (0, 1, 2, 3)
ROW2_IDX = (4, 5, 6)


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


def _draw_step(ax, idx, x, y, w, h, fill):
    title, desc = STEPS[idx]
    cx = x + w / 2
    _box(ax, x, y, w, h, fill)
    ax.text(
        cx,
        y + h - 0.24,
        f"{idx + 1}. {title}",
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=AC["text"],
        zorder=3,
    )
    ax.text(
        cx,
        y + 0.32,
        desc,
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=AC["muted"],
        zorder=3,
        linespacing=1.2,
    )
    return x, y, w, h, cx, y + h / 2


def build_system_flow_figure() -> plt.Figure:
    fig_w, fig_h = 11, 6.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    ax.text(
        fig_w / 2,
        5.85,
        "Question generation and assessment flow",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=AC["title"],
    )

    box_w = 2.35
    box_h = 1.45
    gap = 0.38

    row1_y = 3.55
    row2_y = 1.35
    row1_w = len(ROW1_IDX) * box_w + (len(ROW1_IDX) - 1) * gap
    row2_w = len(ROW2_IDX) * box_w + (len(ROW2_IDX) - 1) * gap
    row1_x0 = (fig_w - row1_w) / 2
    row2_x0 = (fig_w - row2_w) / 2

    # Row bands
    ax.add_patch(
        Rectangle((0.35, row1_y - 0.18), fig_w - 0.7, box_h + 0.36, facecolor=AC["band"], edgecolor=AC["band_edge"], linewidth=0.6, zorder=0)
    )
    ax.add_patch(
        Rectangle((0.35, row2_y - 0.18), fig_w - 0.7, box_h + 0.36, facecolor=AC["band"], edgecolor=AC["band_edge"], linewidth=0.6, zorder=0)
    )
    ax.text(0.55, row1_y + box_h / 2, "Data\npipeline", ha="left", va="center", fontsize=7.5, color=AC["muted"], fontweight="bold", linespacing=1.15)
    ax.text(0.55, row2_y + box_h / 2, "Generation\n& assessment", ha="left", va="center", fontsize=7.5, color=AC["muted"], fontweight="bold", linespacing=1.15)

    positions = {}

    for j, idx in enumerate(ROW1_IDX):
        x = row1_x0 + j * (box_w + gap)
        positions[idx] = _draw_step(ax, idx, x, row1_y, box_w, box_h, AC["fills"][idx])
        if j < len(ROW1_IDX) - 1:
            x1 = x + box_w + 0.03
            x2 = x + box_w + gap - 0.03
            _arrow(ax, x1, row1_y + box_h / 2, x2, row1_y + box_h / 2)

    for j, idx in enumerate(ROW2_IDX):
        x = row2_x0 + j * (box_w + gap)
        positions[idx] = _draw_step(ax, idx, x, row2_y, box_w, box_h, AC["fills"][idx])
        if j < len(ROW2_IDX) - 1:
            x1 = x + box_w + 0.03
            x2 = x + box_w + gap - 0.03
            _arrow(ax, x1, row2_y + box_h / 2, x2, row2_y + box_h / 2)

    # Row 1 (step 4) -> Row 2 (step 5)
    _, _, _, _, cx4, _ = positions[3]
    _, _, _, _, cx5, _ = positions[4]
    mid_y = (row1_y + row2_y + box_h) / 2
    _arrow(ax, cx4, row1_y, cx4, mid_y + 0.08, style="-")
    _arrow(ax, cx4, mid_y + 0.08, cx5, mid_y + 0.08, style="-")
    _arrow(ax, cx5, mid_y + 0.08, cx5, row2_y + box_h + 0.02)

    ax.text(
        fig_w / 2,
        0.42,
        "Implemented in Python (Streamlit); retrieval via FAISS with E5 or AraBERT embeddings.",
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
    fig = build_system_flow_figure()
    save_figure(fig, "system_flow_question_generation", OUT)
    save_figure(fig, "system_flow_question_generation", PRES_OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
