"""
Taxonomy of LLM tasks in NLP — symmetric grid layout (publication quality).

Outputs:
  docs/figures/llm_tasks_taxonomy.png
  docs/figures/llm_tasks_taxonomy.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
PRES_OUT = OUT

BORDER = "#4B5563"
TEXT = "#1F2937"
MUTED = "#374151"
CONNECTOR = "#4B5563"
CENTER_FILL = "#F3F4F6"

CATEGORY_COLORS = {
    "understanding": "#DBEAFE",
    "specialized": "#FEF3C7",
    "generation": "#FCE7F3",
    "longform": "#CCFBF1",
    "transformation": "#EDE9FE",
    "reasoning": "#E5E7EB",
    "interaction": "#E9D5FF",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.10,
    }
)

# Fixed symmetric layout on 16×9 canvas (center at 8, 4.05)
CX, CY = 8.0, 4.05
CAT_W, CAT_H = 3.85, 2.20
CENTER_W, CENTER_H = 4.10, 1.65

CATEGORIES = {
    "understanding": {
        "title": "Understanding",
        "items": [
            "Text classification",
            "Sentiment analysis",
            "Named entity recognition (NER)",
            "Question answering (QA)",
        ],
        "cx": 8.0,
        "cy": 6.95,
        "color": "understanding",
    },
    "specialized": {
        "title": "Specialized domains",
        "items": [
            "Medical / healthcare",
            "Legal",
            "Educational",
            "Media / journalism",
        ],
        "cx": 3.05,
        "cy": 6.95,
        "color": "specialized",
    },
    "generation": {
        "title": "Generation",
        "items": [
            "Text completion",
            "Summarization",
            "Paraphrasing / rewriting",
            "Style transfer",
            "Creative writing",
        ],
        "cx": 12.95,
        "cy": 6.95,
        "color": "generation",
    },
    "longform": {
        "title": "Long-form processing",
        "items": [
            "Document analysis",
            "Retrieval-augmented generation (RAG)",
            "Hierarchical summarization",
        ],
        "cx": 2.75,
        "cy": CY,
        "color": "longform",
    },
    "transformation": {
        "title": "Transformation",
        "items": [
            "Machine translation",
            "Cross-lingual summarization",
            "Cross-lingual QA",
        ],
        "cx": 13.25,
        "cy": CY,
        "color": "transformation",
    },
    "reasoning": {
        "title": "Reasoning & inference",
        "items": [
            "Logical reasoning",
            "Mathematical word problems",
            "Knowledge-based QA",
        ],
        "cx": 3.05,
        "cy": 1.15,
        "color": "reasoning",
    },
    "interaction": {
        "title": "Interaction",
        "items": [
            "Dialogue / conversational AI",
            "Question generation (QG)",
            "Personalization",
        ],
        "cx": 12.95,
        "cy": 1.15,
        "color": "interaction",
    },
}


def _box_rect(cx, cy, w, h):
    return cx - w / 2, cy - h / 2, w, h


def _rounded_box(ax, cx, cy, w, h, fill, lw=1.4):
    x, y, _, _ = _box_rect(cx, cy, w, h)
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.014,rounding_size=0.14",
            linewidth=lw,
            edgecolor=BORDER,
            facecolor=fill,
            zorder=2,
        )
    )


def _polyline(ax, points, lw=2.2):
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        ax.plot([x1, x2], [y1, y2], color=CONNECTOR, linewidth=lw, solid_capstyle="round", zorder=1)


def _category_box(ax, cx, cy, title, items, color_key, *, highlight=False):
    fill = CATEGORY_COLORS[color_key]
    lw = 2.0 if highlight else 1.4
    _rounded_box(ax, cx, cy, CAT_W, CAT_H, fill, lw=lw)
    x, y, w, h = _box_rect(cx, cy, CAT_W, CAT_H)
    pad_x = 0.22
    ax.text(
        cx,
        y + h - 0.28,
        title,
        ha="center",
        va="top",
        fontsize=13.5,
        fontweight="bold",
        color=TEXT,
        zorder=3,
    )
    line_step = 0.31 if len(items) >= 5 else 0.34
    line_y = y + h - 0.74
    body_size = 9.8 if len(items) >= 5 else 10.2
    for item in items:
        ax.text(
            x + pad_x,
            line_y,
            f"•  {item}",
            ha="left",
            va="top",
            fontsize=body_size,
            color=MUTED,
            zorder=3,
            linespacing=1.18,
        )
        line_y -= line_step


def _center_box(ax):
    _rounded_box(ax, CX, CY, CENTER_W, CENTER_H, CENTER_FILL, lw=1.6)
    ax.text(
        CX,
        CY + 0.12,
        "Natural Language",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=TEXT,
        zorder=3,
    )
    ax.text(
        CX,
        CY - 0.22,
        "Processing (NLP)",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=TEXT,
        zorder=3,
    )


def _draw_connectors(ax):
    """Orthogonal hub connectors — top bus, side arms, bottom bus."""
    c_top = CY + CENTER_H / 2
    c_bottom = CY - CENTER_H / 2
    c_left = CX - CENTER_W / 2
    c_right = CX + CENTER_W / 2

    top_bus_y = 6.05
    bot_bus_y = 1.95

    # Center → top bus → three top categories
    _polyline(ax, [(CX, c_top), (CX, top_bus_y)])
    _polyline(ax, [(3.05, top_bus_y), (12.95, top_bus_y)])
    for cx in (3.05, 8.0, 12.95):
        cat_bottom = 6.95 - CAT_H / 2
        _polyline(ax, [(cx, top_bus_y), (cx, cat_bottom)])

    # Center → left / right middle categories
    longform_right = 2.75 + CAT_W / 2
    trans_left = 13.25 - CAT_W / 2
    _polyline(ax, [(c_left, CY), (longform_right, CY)])
    _polyline(ax, [(c_right, CY), (trans_left, CY)])

    # Center → bottom bus → two bottom categories
    _polyline(ax, [(CX, c_bottom), (CX, bot_bus_y)])
    _polyline(ax, [(3.05, bot_bus_y), (12.95, bot_bus_y)])
    for cx in (3.05, 12.95):
        cat_top = 1.15 + CAT_H / 2
        _polyline(ax, [(cx, bot_bus_y), (cx, cat_top)])


def build_llm_tasks_taxonomy_figure() -> plt.Figure:
    fig_w, fig_h = 16.0, 9.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.set_facecolor("white")

    ax.text(
        fig_w / 2,
        8.35,
        "Taxonomy of LLM tasks in natural language processing",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=TEXT,
    )

    _draw_connectors(ax)
    _center_box(ax)

    draw_order = [
        "specialized",
        "understanding",
        "generation",
        "longform",
        "transformation",
        "reasoning",
        "interaction",
    ]
    for key in draw_order:
        cat = CATEGORIES[key]
        _category_box(
            ax,
            cat["cx"],
            cat["cy"],
            cat["title"],
            cat["items"],
            cat["color"],
            highlight=key in ("longform", "interaction"),
        )

    return fig


def save_figure(fig: plt.Figure, stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, facecolor="white", edgecolor="none")
    print(f"Saved {path}")


def main() -> None:
    fig = build_llm_tasks_taxonomy_figure()
    save_figure(fig, "llm_tasks_taxonomy", OUT)
    save_figure(fig, "llm_tasks_taxonomy", PRES_OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
