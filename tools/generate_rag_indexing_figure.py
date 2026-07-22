"""
Generate RAG indexing flow diagram — academic two-row layout (matplotlib only).

Outputs:
  docs/figures/rag_indexing_process.png
  docs/figures/rag_indexing_process.png
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
        "#EAECEE",
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
    ("Preparation", "Input folder\n→ file list."),
    ("Text extraction", "PDF / DOCX / TXT\n→ raw text."),
    ("Text cleaning", "Remove noise and\nnormalize text."),
    ("Stemming", "Arabic stemming\n→ normalized tokens."),
    ("Chunking", "500-char segments;\n100-char overlap."),
    ("Embedding", "E5-large-v2 or AraBERT\n→ vector representations."),
    ("Indexing", "FAISS similarity search\n→ index + metadata."),
    ("Storage", "Batch persistence\n→ complete index store."),
]

ROW1 = (0, 1, 2, 3)
ROW2 = (4, 5, 6, 7)


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
        fontsize=8.0,
        fontweight="bold",
        color=AC["text"],
        zorder=3,
    )
    ax.text(
        cx,
        y + 0.30,
        desc,
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=AC["muted"],
        zorder=3,
        linespacing=1.15,
    )
    return cx, y + h / 2


def _draw_row(ax, indices, y, box_w, box_h, gap, x0):
    positions = {}
    for j, idx in enumerate(indices):
        x = x0 + j * (box_w + gap)
        cx, cy = _draw_step(ax, idx, x, y, box_w, box_h, AC["fills"][idx])
        positions[idx] = (x, y, box_w, box_h, cx, cy)
        if j < len(indices) - 1:
            _arrow(ax, x + box_w + 0.03, y + box_h / 2, x + box_w + gap - 0.03, y + box_h / 2)
    return positions


def build_rag_indexing_figure() -> plt.Figure:
    fig_w, fig_h = 12, 6.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    ax.text(
        fig_w / 2,
        5.95,
        "RAG indexing process for Arabic texts",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=AC["title"],
    )

    box_w = 2.45
    box_h = 1.42
    gap = 0.34
    row1_y = 3.55
    row2_y = 1.35
    row_w = 4 * box_w + 3 * gap
    row_x0 = (fig_w - row_w) / 2

    ax.add_patch(
        Rectangle((0.35, row1_y - 0.16), fig_w - 0.7, box_h + 0.32, facecolor=AC["band"], edgecolor=AC["band_edge"], linewidth=0.6, zorder=0)
    )
    ax.add_patch(
        Rectangle((0.35, row2_y - 0.16), fig_w - 0.7, box_h + 0.32, facecolor=AC["band"], edgecolor=AC["band_edge"], linewidth=0.6, zorder=0)
    )
    ax.text(0.55, row1_y + box_h / 2, "Text\npreparation", ha="left", va="center", fontsize=7.5, color=AC["muted"], fontweight="bold", linespacing=1.1)
    ax.text(0.55, row2_y + box_h / 2, "Vector index\nconstruction", ha="left", va="center", fontsize=7.5, color=AC["muted"], fontweight="bold", linespacing=1.1)

    pos1 = _draw_row(ax, ROW1, row1_y, box_w, box_h, gap, row_x0)
    pos2 = _draw_row(ax, ROW2, row2_y, box_w, box_h, gap, row_x0)

    _, _, _, _, cx4, _ = pos1[3]
    _, _, _, _, cx5, _ = pos2[4]
    mid_y = (row1_y + row2_y + box_h) / 2
    _arrow(ax, cx4, row1_y, cx4, mid_y + 0.06, style="-")
    _arrow(ax, cx4, mid_y + 0.06, cx5, mid_y + 0.06, style="-")
    _arrow(ax, cx5, mid_y + 0.06, cx5, row2_y + box_h + 0.02)

    ax.text(
        fig_w / 2,
        0.42,
        "External references indexed with FAISS-CPU; metadata stored per chunk (source file, passage).",
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
    fig = build_rag_indexing_figure()
    save_figure(fig, "rag_indexing_process", OUT)
    save_figure(fig, "rag_indexing_process", PRES_OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
