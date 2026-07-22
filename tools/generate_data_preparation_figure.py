"""
Publication-quality data preparation pipeline (six stages).

Outputs (PNG only):
  docs/figures/data_preparation_pipeline.png          — 2×3 grid (thesis / paper)
  docs/figures/data_preparation_pipeline.png     — horizontal (slides)
  docs/figures/presentation_detail_data_preparation.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
PRES_OUT = OUT

# Dark, projector-readable text (no light gray)
TITLE_COLOR = "#1F2937"
BODY_COLOR = "#374151"
BORDER_COLOR = "#4B5563"
ARROW_COLOR = "#4B5563"

# Data prep = blue family · embedding = green · FAISS output = purple
STAGE_FILLS = [
    "#DBEAFE",  # documents
    "#D6EAF8",  # extraction
    "#EBF5FB",  # preprocessing
    "#E8F4FC",  # chunking
    "#D5F5E3",  # embedding
    "#EDE9FE",  # FAISS knowledge base
]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Tahoma", "Arial"],
        "font.weight": "bold",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.10,
    }
)

STAGES = [
    {
        "title": "Arabic Educational\nDocuments",
        "lines": ["PDF · DOCX · TXT"],
        "bullets": False,
    },
    {
        "title": "Text Extraction",
        "lines": ["Parse documents", "Preserve structure"],
        "bullets": True,
    },
    {
        "title": "Arabic Text\nPreprocessing",
        "lines": ["Unicode normalize", "Whitespace trim", "Text cleaning"],
        "bullets": True,
    },
    {
        "title": "Chunking Strategy",
        "lines": ["500-char chunks", "100-char overlap", "Semantic continuity"],
        "bullets": True,
    },
    {
        "title": "Embedding\nGeneration",
        "lines": ["AraBERT embeddings", "Vector representation"],
        "bullets": True,
    },
    {
        "title": "FAISS Knowledge\nBase",
        "lines": ["Vector index", "Metadata storage", "Source mapping"],
        "bullets": True,
    },
]

LAYOUT = {
    "grid": {
        "fig_w": 10.2,
        "fig_h": 5.6,
        "cols": 3,
        "rows": 2,
        "box_w": 2.85,
        "box_h": 2.05,
        "gap_x": 0.38,
        "gap_y": 0.48,
        "title_size": 10.8,
        "line_size": 9.0,
        "title_leading": 0.18,
        "body_leading": 0.155,
        "block_gap": 0.10,
    },
    "presentation": {
        "fig_w": 16.25,
        "fig_h": 9.0,
        "box_w": 2.58,
        "box_h": 2.18,
        "gap": 0.12,
        "title_size": 13.0,
        "line_size": 10.2,
        "title_leading": 0.20,
        "body_leading": 0.17,
        "block_gap": 0.11,
    },
}


def _box(ax, x, y, w, h, fill):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.05",
            linewidth=1.1,
            edgecolor=BORDER_COLOR,
            facecolor=fill,
            zorder=2,
        )
    )


def _arrow_h(ax, x1, x2, y):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y),
            (x2, y),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=ARROW_COLOR,
            shrinkA=2,
            shrinkB=2,
            zorder=1,
        )
    )


def _arrow_l(ax, x1, y1, x2, y2, y_mid):
    """L-shaped connector (vertical → horizontal → vertical) with arrow at end."""
    kw = dict(
        arrowstyle="-",
        mutation_scale=10,
        linewidth=1.0,
        color=ARROW_COLOR,
        shrinkA=0,
        shrinkB=0,
        zorder=1,
    )
    ax.add_patch(FancyArrowPatch((x1, y1), (x1, y_mid), **kw))
    ax.add_patch(FancyArrowPatch((x1, y_mid), (x2, y_mid), **kw))
    ax.add_patch(
        FancyArrowPatch(
            (x2, y_mid),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=ARROW_COLOR,
            shrinkA=0,
            shrinkB=2,
            zorder=1,
        )
    )


def _stage_pos(cfg, index: int) -> tuple[float, float]:
    """Map stage index to (x, y) for a 3×2 grid: top row 0→2, bottom row 3→5."""
    cols, rows = cfg["cols"], cfg["rows"]
    box_w, box_h = cfg["box_w"], cfg["box_h"]
    gap_x, gap_y = cfg["gap_x"], cfg["gap_y"]
    total_w = cols * box_w + (cols - 1) * gap_x
    total_h = rows * box_h + (rows - 1) * gap_y
    x0 = (cfg["fig_w"] - total_w) / 2
    y0 = (cfg["fig_h"] - total_h) / 2

    col = index % cols
    row = index // cols  # 0 = top row, 1 = bottom row
    x = x0 + col * (box_w + gap_x)
    y = y0 + (rows - 1 - row) * (box_h + gap_y)
    return x, y


def _draw_stage(ax, x, y, w, h, stage, fill, cfg):
    _box(ax, x, y, w, h, fill)
    cx = x + w / 2
    cy = y + h / 2

    title_size = cfg["title_size"]
    line_size = cfg["line_size"]
    title_lines = stage["title"].count("\n") + 1
    body_lines = len(stage["lines"])
    title_block = title_lines * cfg["title_leading"]
    body_block = body_lines * cfg["body_leading"]
    block_gap = cfg["block_gap"]
    total_h = title_block + block_gap + body_block
    y_top = cy + total_h / 2

    ax.text(
        cx,
        y_top,
        stage["title"],
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=TITLE_COLOR,
        zorder=3,
        linespacing=1.08,
        clip_on=True,
    )

    bullet = "• " if stage["bullets"] else ""
    body = "\n".join(f"{bullet}{line}" for line in stage["lines"])
    ax.text(
        cx,
        y_top - title_block - block_gap,
        body,
        ha="center",
        va="top",
        fontsize=line_size,
        fontweight="bold",
        color=BODY_COLOR,
        zorder=3,
        linespacing=1.18,
        clip_on=True,
    )


def build_data_preparation_figure(*, presentation: bool = False) -> plt.Figure:
    """Horizontal pipeline — six connected stages (presentation slides)."""
    cfg = LAYOUT["presentation"]
    box_w, box_h, gap = cfg["box_w"], cfg["box_h"], cfg["gap"]
    n = len(STAGES)
    total_w = n * box_w + (n - 1) * gap
    fig_w, fig_h = cfg["fig_w"], cfg["fig_h"]
    x0 = (fig_w - total_w) / 2
    y_box = (fig_h - box_h) / 2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.set_facecolor("white")

    cy = y_box + box_h / 2
    for i, stage in enumerate(STAGES):
        x = x0 + i * (box_w + gap)
        _draw_stage(ax, x, y_box, box_w, box_h, stage, STAGE_FILLS[i], cfg)
        if i < n - 1:
            _arrow_h(ax, x + box_w + 0.03, x + box_w + gap - 0.03, cy)

    return fig


def build_data_preparation_grid_figure() -> plt.Figure:
    """Balanced 2×3 grid — compact thesis figure (~10×5.6 in, neither too wide nor tall)."""
    cfg = LAYOUT["grid"]
    box_w, box_h = cfg["box_w"], cfg["box_h"]
    gap_x, gap_y = cfg["gap_x"], cfg["gap_y"]
    fig_w, fig_h = cfg["fig_w"], cfg["fig_h"]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.set_facecolor("white")

    positions = []
    for i, stage in enumerate(STAGES):
        x, y = _stage_pos(cfg, i)
        positions.append((x, y))
        _draw_stage(ax, x, y, box_w, box_h, stage, STAGE_FILLS[i], cfg)

    # Top row: 1 → 2 → 3
    for i in range(2):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        cy = y1 + box_h / 2
        _arrow_h(ax, x1 + box_w + 0.03, x2 - 0.03, cy)

    # Bottom row: 4 → 5 → 6
    for i in range(3, 5):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        cy = y1 + box_h / 2
        _arrow_h(ax, x1 + box_w + 0.03, x2 - 0.03, cy)

    # Wrap: stage 3 (top-right) ↓ stage 4 (bottom-left)
    x3, y3 = positions[2]
    x4, y4 = positions[3]
    y_mid = y4 + box_h + gap_y / 2
    _arrow_l(
        ax,
        x3 + box_w / 2,
        y3 - 0.03,
        x4 + box_w / 2,
        y4 + box_h + 0.03,
        y_mid,
    )

    return fig


def save_figure(fig: plt.Figure, stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, facecolor="white", edgecolor="none")
    print(f"Saved {path}")


def main() -> None:
    fig_grid = build_data_preparation_grid_figure()
    save_figure(fig_grid, "data_preparation_pipeline", OUT)
    plt.close(fig_grid)

    fig_pres = build_data_preparation_figure(presentation=True)
    save_figure(fig_pres, "data_preparation_pipeline", PRES_OUT)
    save_figure(fig_pres, "presentation_detail_data_preparation", PRES_OUT)
    plt.close(fig_pres)


if __name__ == "__main__":
    main()
