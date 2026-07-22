"""
Generate methodology figures for the thesis and related manuscripts.

Official thesis figure:
  • مخطط الميثودولوجي للأطروحة  (thesis_methodology_diagram)
  • Companion: fig2_experimental_design (2×2 factorial, separate figure)

Outputs (PNG only — no PDF):
  docs/figures/thesis_methodology_diagram.png
  docs/figures/fig1_system_architecture.png
  docs/figures/fig2_experimental_design.png
  docs/figures/presentation_methodology_overview.png
  docs/figures/presentation_methodology_overview.png
  docs/figures/presentation_detail_data_preparation.png
  docs/figures/presentation_detail_rag.png
  docs/figures/presentation_detail_generation.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
PRES_OUT = OUT

# Official thesis figure naming (file name only — no Arabic text on figures)
THESIS_METHODOLOGY_AR = "مخطط الميثودولوجي للأطروحة"  # documentation / caption in Word
THESIS_METHODOLOGY_EN = "Thesis Methodology Diagram"
THESIS_METHODOLOGY_STEM = "thesis_methodology_diagram"
THESIS_METHODOLOGY_ALIAS = "fig1_system_architecture"

PRES_METHODOLOGY_OVERVIEW_EN = "Research Methodology"

C = {
    "phase": "#F5F5F5",
    "phase_edge": "#CCCCCC",
    "design": "#D4E6F1",
    "input": "#D6EAF8",
    "process": "#EBF5FB",
    "retrieval": "#FCF3CF",
    "retrieval_improved": "#F9E79F",
    "generation": "#D5F5E3",
    "dataset": "#E8F8F5",
    "evaluation": "#E8DAEF",
    "stats": "#FADBD8",
    "border": "#1F2937",
    "arrow": "#374151",
    "arrow_light": "#6B7280",
    "text": "#1F2937",
    "muted": "#374151",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Tahoma", "DejaVu Sans", "Arial"],
        "font.size": 9.0,
        "font.weight": "bold",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)

P = {
    "bg": "#FFFFFF",
    "title": "#1B2631",
    "step1": "#3498DB",
    "step2": "#F39C12",
    "step3": "#27AE60",
    "step4": "#8E44AD",
    "step5": "#E74C3C",
    "box_bg": "#FAFBFC",
    "border": "#2C3E50",
    "arrow": "#566573",
    "muted": "#5D6D7E",
}


def _box(ax, x, y, w, h, text, facecolor, fontsize=8.5, bold=True, edgecolor=None):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.06",
        linewidth=1.25,
        edgecolor=edgecolor or C["border"],
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color=C["text"],
        zorder=3,
        linespacing=1.22,
    )
    return patch


def _arrow(ax, x1, y1, x2, y2, style="->", color=None, connection="arc3,rad=0", lw=1.1):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style,
            color=color or C["arrow"],
            linewidth=lw,
            mutation_scale=11,
            connectionstyle=connection,
            zorder=1,
        )
    )


def _phase_band(ax, y, h, label):
    ax.add_patch(
        Rectangle((0.05, y), 0.14, h, facecolor=C["phase"], edgecolor=C["phase_edge"], linewidth=0.8, zorder=0)
    )
    ax.text(
        0.12, y + h / 2, label,
        ha="center", va="center",
        fontsize=8.2, fontweight="bold", color=C["text"],
        rotation=90, zorder=1,
    )


def build_experimental_design_figure() -> plt.Figure:
    """Separate figure: 2×2 factorial experimental design."""
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(
        4.25, 5.75,
        "Experimental Design (2 × 2 Factorial)",
        ha="center", fontsize=12, fontweight="bold", color=C["text"],
    )
    ax.text(
        4.25, 5.42,
        "Identical prompts · Temperature = 0.7 · shared JSON schema",
        ha="center", fontsize=8, color=C["muted"], style="italic",
    )

    ax.text(4.25, 4.95, "Factor B: Method", ha="center", fontsize=9, fontweight="bold", color=C["text"])
    ax.text(
        1.05, 3.35,
        "Factor A:\nModel\n\n• LLaMA 3.2 3B\n• Qwen 2.5 7B",
        ha="center", va="center", fontsize=8.5, fontweight="bold", color=C["text"],
    )
    ax.text(6.55, 3.35, "×", ha="center", va="center", fontsize=18, color=C["muted"])

    cell_w, cell_h = 2.8, 1.35
    x0, x1 = 2.0, 5.05
    y0, y1 = 1.05, 2.75

    cells = [
        (x0, y1, "LLaMA 3.2 3B\nVanilla"),
        (x1, y1, "LLaMA 3.2 3B\nRAG"),
        (x0, y0, "Qwen 2.5 7B\nVanilla"),
        (x1, y0, "Qwen 2.5 7B\nRAG"),
    ]
    for x, y, label in cells:
        _box(ax, x, y, cell_w, cell_h, label, C["design"], fontsize=9.5, bold=True)

    ax.text(x0 + cell_w / 2, 4.55, "Vanilla", ha="center", fontsize=8.5, color=C["muted"])
    ax.text(x1 + cell_w / 2, 4.55, "RAG", ha="center", fontsize=8.5, color=C["muted"])

    ax.add_patch(
        Rectangle((1.75, 0.85), 6.35, 3.45, fill=False, edgecolor=C["phase_edge"], linewidth=1.2, linestyle="--", zorder=0)
    )

    _arrow(ax, 4.25, 0.85, 4.25, 0.67)
    _box(ax, 2.55, 0.15, 3.4, 0.52, "4 Experimental Groups", C["design"], fontsize=9.5, bold=True)

    return fig


def build_methodology_figure() -> plt.Figure:
    """مخطط الميثودولوجي للأطروحة — pipeline first, then config branches; Vanilla | RAG fork."""
    fig_w, fig_h = 10.5, 13.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 13.6)
    ax.axis("off")

    # No on-figure title — name is file-only: thesis_methodology_diagram
    # Arabic caption goes in Word: مخطط الميثودولوجي للأطروحة

    # ── Phase I: Data ──
    _phase_band(ax, 11.2, 1.05, "Phase I\nData")
    _box(ax, 0.55, 12.15, 2.0, 0.72, "Arabic educational\ndocuments", C["input"], fontsize=8.8)
    _box(ax, 2.75, 12.15, 1.75, 0.72, "Extract &\nnormalize", C["process"], fontsize=8.8)
    _box(ax, 4.7, 12.15, 1.65, 0.72, "Text\nchunking", C["process"], fontsize=8.8)
    _box(ax, 6.55, 12.15, 3.4, 0.72, "FAISS index · embeddings · metadata", C["process"], fontsize=8.5)
    for x0, x1 in [(2.55, 2.75), (4.5, 4.7), (6.35, 6.55)]:
        _arrow(ax, x0, 12.51, x1, 12.51)

    # ── Phase II: Retrieval (pipeline → configuration branches) ──
    _phase_band(ax, 8.55, 2.15, "Phase II\nRetrieval")
    _box(
        ax, 0.55, 10.75, 9.4, 0.72,
        "Retrieval pipeline:  query embedding  →  FAISS search  →  Top-K retrieval  →  re-ranking  →  retrieved context",
        C["retrieval"], fontsize=8.3,
    )
    _arrow(ax, 8.25, 12.15, 5.25, 11.47, connection="arc3,rad=-0.08")
    _arrow(ax, 5.25, 10.75, 2.725, 10.18)
    _arrow(ax, 5.25, 10.75, 7.75, 10.18)
    _box(
        ax, 0.55, 9.25, 4.35, 0.88,
        "Baseline retrieval configuration (comparison)\nE5 · top-5 · no re-ranking · θ ≥ 0.82",
        C["retrieval"], fontsize=8.1, edgecolor="#B7950B",
    )
    _box(
        ax, 5.55, 9.25, 4.4, 0.88,
        "Improved retrieval configuration (adopted)\nAraBERT · top-10 → rerank · θ ≥ 0.65",
        C["retrieval_improved"], fontsize=8.1, edgecolor="#27AE60",
    )

    # ── Phase III: Generation (two parallel columns, no overlap) ──
    _phase_band(ax, 4.35, 3.65, "Phase III\nGeneration")
    _box(ax, 3.55, 8.35, 3.4, 0.52, "Input document", C["input"], fontsize=9.0)

    # Left — Vanilla
    _box(ax, 0.55, 7.15, 3.2, 0.72, "Vanilla prompt", C["generation"], fontsize=8.8)
    _arrow(ax, 4.0, 8.35, 2.15, 7.87)

    # Right — RAG (stacked column; gap ≥0.48 before Ollama)
    _box(ax, 6.75, 7.65, 3.2, 0.62, "Context retrieval", C["generation"], fontsize=8.5)
    _box(ax, 6.75, 6.75, 3.2, 0.62, "Retrieved context", C["generation"], fontsize=8.5)
    _box(ax, 6.75, 5.85, 3.2, 0.72, "RAG prompt", C["generation"], fontsize=8.8)
    _arrow(ax, 6.0, 8.35, 8.35, 8.27)
    _arrow(ax, 5.25, 10.75, 8.35, 8.27, color=C["arrow_light"], lw=0.9)
    _arrow(ax, 8.35, 7.65, 8.35, 7.37)
    _arrow(ax, 8.35, 6.75, 8.35, 6.47)

    # Merge → Ollama → outputs (RAG prompt bottom 5.85; Ollama top 5.0 → gap 0.85)
    _box(
        ax, 2.65, 4.0, 5.2, 1.0,
        "Ollama Inference\nLLaMA 3.2 3B\nQwen 2.5 7B\nTemperature = 0.7",
        C["generation"], fontsize=8.3,
    )
    _arrow(ax, 2.15, 7.15, 4.5, 5.0)
    _arrow(ax, 8.35, 5.85, 6.85, 5.0)
    _box(
        ax, 0.55, 3.2, 9.4, 0.58,
        "Validated JSON · MCQ · True/False · metadata (model, method, source, file id)",
        C["generation"], fontsize=8.3,
    )
    _box(
        ax, 0.55, 2.5, 9.4, 0.52,
        "Generated dataset: generated JSON outputs → cleaning → final dataset",
        C["dataset"], fontsize=8.5,
    )
    _arrow(ax, 5.25, 4.0, 5.25, 3.78)
    _arrow(ax, 5.25, 3.2, 5.25, 3.02)

    # ── Phase IV: Evaluation ──
    _phase_band(ax, 1.15, 2.35, "Phase IV\nEvaluation")
    _box(
        ax, 0.55, 1.10, 4.35, 1.15,
        "Automatic evaluation\nPrecision · Recall · F1 · BLEU · BERTScore · Perplexity",
        C["evaluation"], fontsize=8.3,
    )
    _box(
        ax, 5.55, 1.10, 4.4, 1.15,
        "Human evaluation\nSampled questions · single-blind · Likert 1–5\n"
        "Clarity · Logic · Relevance · Option quality · Accuracy",
        C["evaluation"], fontsize=8.1,
    )
    _arrow(ax, 2.72, 2.5, 2.72, 2.25)
    _arrow(ax, 7.75, 2.5, 7.75, 2.25)

    # ── Phase V: Statistics ──
    _phase_band(ax, 0.15, 1.05, "Phase V\nStatistics")
    _box(ax, 0.55, 0.35, 2.05, 0.72, "Shapiro–Wilk", C["stats"], fontsize=8.3)
    _box(ax, 2.85, 0.35, 2.15, 0.72, "Mann–Whitney U", C["stats"], fontsize=8.3)
    _box(ax, 5.25, 0.35, 2.15, 0.72, "Holm–Bonferroni", C["stats"], fontsize=8.3)
    _box(ax, 7.65, 0.35, 2.3, 0.72, "Effect size (r_rb)", C["stats"], fontsize=8.3)
    for x0, x1 in [(2.6, 2.85), (5.0, 5.25), (7.4, 7.65)]:
        _arrow(ax, x0, 0.71, x1, 0.71)
    _arrow(ax, 2.72, 1.10, 1.57, 1.07)
    _arrow(ax, 7.75, 1.10, 6.32, 1.07)

    return fig


def _step_circle(ax, cx, cy, r, num, color):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((cx, cy), r, facecolor=color, edgecolor=C["border"], linewidth=1.2, zorder=3))
    ax.text(cx, cy, str(num), ha="center", va="center", fontsize=14, fontweight="bold", color="white", zorder=4)


def _pres_box(ax, x, y, w, h, title, subtitle, color, title_size=14, sub_size=10):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            linewidth=2.0, edgecolor=color, facecolor=P["box_bg"], zorder=2,
        )
    )
    cx = x + w / 2
    if subtitle:
        ax.text(
            cx, y + h * 0.66, title,
            ha="center", va="center", fontsize=title_size, fontweight="bold",
            color=C["text"], zorder=3, linespacing=1.1,
            clip_on=True,
        )
        ax.text(
            cx, y + h * 0.24, subtitle,
            ha="center", va="center", fontsize=sub_size, fontweight="bold",
            color=C["muted"], zorder=3,
            linespacing=1.25, clip_on=True,
        )
    else:
        ax.text(
            cx, y + h / 2, title,
            ha="center", va="center", fontsize=title_size, fontweight="bold",
            color=C["text"], zorder=3, linespacing=1.1, clip_on=True,
        )


def _pres_harrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=18, linewidth=2.2, color=P["arrow"], zorder=1,
        )
    )


def build_presentation_methodology_overview() -> plt.Figure:
    """High-level 16:9 overview — minimal text; detail slides follow for steps 1–3."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor(P["bg"])

    ax.text(
        8, 8.15, PRES_METHODOLOGY_OVERVIEW_EN,
        ha="center", fontsize=24, fontweight="bold", color=P["title"],
    )
    ax.text(
        8, 7.55, "Arabic Question Generation: Vanilla vs. RAG",
        ha="center", fontsize=14, color=P["muted"], style="italic",
    )

    steps = [
        (P["step1"], "1", "Data\nPreparation", "Docs · chunking\nFAISS"),
        (P["step2"], "2", "RAG\nRetrieval", "Embedding · Retrieval\nRe-ranking"),
        (P["step3"], "3", "Question\nGeneration", "LLaMA / Qwen\nVanilla / RAG"),
        (P["step4"], "4", "Evaluation", "Automatic Metrics\nHuman Evaluation"),
        (P["step5"], "5", "Statistical\nAnalysis", "Mann–Whitney · Holm\nEffect Size"),
    ]

    box_w, box_h, gap = 2.55, 2.55, 0.45
    total_w = len(steps) * box_w + (len(steps) - 1) * gap
    x_start = (16 - total_w) / 2
    y_box, cy = 3.65, 6.55

    for i, (color, num, title, sub) in enumerate(steps):
        x = x_start + i * (box_w + gap)
        cx = x + box_w / 2
        _step_circle(ax, cx, cy, 0.38, num, color)
        _pres_box(ax, x, y_box, box_w, box_h, title, sub, color, title_size=12.5, sub_size=9)
        if i < len(steps) - 1:
            _pres_harrow(ax, x + box_w + 0.06, y_box + box_h / 2, x + box_w + gap - 0.06, y_box + box_h / 2)

    return fig


def build_presentation_detail_data_preparation() -> plt.Figure:
    """Detail slide: Phase I — data preparation pipeline."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor(P["bg"])

    ax.text(8, 8.2, "Detail (1/3): Data Preparation", ha="center", fontsize=22, fontweight="bold", color=P["title"])
    ax.text(8, 7.65, "Arabic educational documents → indexed knowledge base", ha="center", fontsize=13, color=P["muted"])

    labels = [
        ("Arabic educational\ndocuments", P["step1"]),
        ("Extract &\nnormalize", P["step1"]),
        ("Text chunking\n500 chars · overlap 100", P["step1"]),
        ("Embeddings\nAraBERT (adopted)", P["step1"]),
        ("FAISS index\n+ metadata", P["step1"]),
    ]
    bw, bh, gap = 2.55, 1.55, 0.45
    total = len(labels) * bw + (len(labels) - 1) * gap
    x0 = (16 - total) / 2
    y = 4.35

    for i, (label, color) in enumerate(labels):
        x = x0 + i * (bw + gap)
        _pres_box(ax, x, y, bw, bh, label, "", color, title_size=12, sub_size=10)
        if i < len(labels) - 1:
            _pres_harrow(ax, x + bw + 0.05, y + bh / 2, x + bw + gap - 0.05, y + bh / 2)

    ax.add_patch(Rectangle((1.5, 2.0), 13.0, 1.55, facecolor="#D6EAF8", edgecolor="#85C1E9", linewidth=1.2, zorder=0))
    ax.text(
        8, 2.95,
        "Output: searchable vector index for RAG retrieval",
        ha="center", fontsize=13, fontweight="bold", color=P["title"],
    )
    ax.text(
        8, 2.35,
        "Supports PDF · DOCX · TXT · MD  ·  chunk metadata: file, page, paragraph",
        ha="center", fontsize=11, color=P["muted"],
    )

    return fig


def build_presentation_detail_rag() -> plt.Figure:
    """Detail slide: Phase II — RAG retrieval mechanism."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor(P["bg"])

    ax.text(8, 8.2, "Detail (2/3): RAG Retrieval Mechanism", ha="center", fontsize=22, fontweight="bold", color=P["title"])
    ax.text(8, 7.65, "Shared pipeline · two retrieval configurations compared", ha="center", fontsize=13, color=P["muted"])

    _pres_box(
        ax, 1.2, 5.75, 13.6, 1.25,
        "Retrieval pipeline",
        "query embedding → FAISS search → Top-K retrieval\nre-ranking → retrieved context",
        P["step2"], title_size=14, sub_size=10,
    )
    _arrow(ax, 8, 5.75, 4.3, 5.20, lw=1.8)
    _arrow(ax, 8, 5.75, 11.7, 5.20, lw=1.8)

    _pres_box(
        ax, 1.2, 3.55, 6.2, 1.65,
        "Baseline (comparison)",
        "E5 · top-5 · no re-ranking\nθ ≥ 0.82",
        "#F39C12", title_size=13, sub_size=10,
    )
    _pres_box(
        ax, 8.6, 3.55, 6.2, 1.65,
        "Improved (adopted)",
        "AraBERT · top-10 → rerank\nθ ≥ 0.65",
        P["step3"], title_size=13, sub_size=10,
    )

    ax.add_patch(Rectangle((1.2, 1.35), 13.6, 1.65, facecolor="#FCF3CF", edgecolor="#F7DC6F", linewidth=1.2, zorder=0))
    ax.text(
        8, 2.55,
        "Retrieved context is injected into the RAG prompt (next slide)",
        ha="center", fontsize=13, fontweight="bold", color=P["title"],
    )
    ax.text(
        8, 1.85,
        "Cross-Encoder re-ranking selects the most relevant chunks for generation",
        ha="center", fontsize=11, color=P["muted"],
    )

    return fig


def build_presentation_detail_generation() -> plt.Figure:
    """Question generation — Vanilla vs RAG fork (no external captions)."""
    fig_w, fig_h = 15.0, 10.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.set_facecolor(P["bg"])

    cx = fig_w / 2
    bw_narrow, bw_wide = 3.75, 5.85
    gap_v = 0.44

    # ── Input ──
    y_input = 8.15
    h_input = 0.92
    _pres_box(
        ax, cx - bw_narrow / 2, y_input, bw_narrow, h_input,
        "Input document", "", P["step1"], title_size=15.0, sub_size=11,
    )

    # ── Vanilla branch (left) ──
    x_vanilla = 0.85
    y_vanilla = 6.05
    h_vanilla = 1.22
    _pres_box(
        ax, x_vanilla, y_vanilla, bw_narrow, h_vanilla,
        "Vanilla prompt", "document text only", P["step3"], title_size=14.0, sub_size=10.5,
    )

    # ── RAG branch (right) — clear of Ollama horizontally & vertically ──
    x_rag = fig_w - 0.70 - bw_narrow  # keep ≥0.15 gap from Ollama right edge
    h_rag = 0.90
    y_ctx = 6.55
    y_ret = y_ctx - h_rag - gap_v
    y_rag_prompt = y_ret - h_rag - gap_v
    h_rag_prompt = 1.05
    _pres_box(ax, x_rag, y_ctx, bw_narrow, h_rag, "Context retrieval", "", P["step2"], title_size=13.5, sub_size=11)
    _pres_box(ax, x_rag, y_ret, bw_narrow, h_rag, "Retrieved context", "", P["step2"], title_size=13.5, sub_size=11)
    _pres_box(
        ax, x_rag, y_rag_prompt, bw_narrow, h_rag_prompt,
        "RAG prompt", "document + context", P["step2"], title_size=13.5, sub_size=10.5,
    )

    # ── Merge → Ollama ──
    y_ollama = 1.85
    h_ollama = 1.50
    _pres_box(
        ax, cx - bw_wide / 2, y_ollama, bw_wide, h_ollama,
        "Ollama Inference",
        "LLaMA 3.2 3B · Qwen 2.5 7B\nTemperature = 0.7",
        P["step3"], title_size=15.0, sub_size=10.5,
    )

    # ── Output ──
    y_out = 0.40
    h_out = 1.18
    _pres_box(
        ax, cx - 4.75, y_out, 9.5, h_out,
        "Validated JSON output",
        "MCQ · True/False\nmetadata (model, method, source)",
        P["step4"], title_size=14.5, sub_size=10.5,
    )

    # ── Arrows ──
    vcx = x_vanilla + bw_narrow / 2
    rcx = x_rag + bw_narrow / 2

    _pres_harrow(ax, cx - 0.55, y_input, vcx, y_vanilla + h_vanilla)
    _pres_harrow(ax, cx + 0.55, y_input, rcx, y_ctx + h_rag)

    _pres_harrow(ax, rcx, y_ctx, rcx, y_ret + h_rag)
    _pres_harrow(ax, rcx, y_ret, rcx, y_rag_prompt + h_rag_prompt)

    ollama_top = y_ollama + h_ollama
    _pres_harrow(ax, vcx, y_vanilla, cx - 1.35, ollama_top)
    _pres_harrow(ax, rcx, y_rag_prompt, cx + 1.35, ollama_top)

    _pres_harrow(ax, cx, y_ollama, cx, y_out + h_out)

    return fig


def build_presentation_methodology_figure() -> plt.Figure:
    """Legacy alias — redirects to the presentation overview."""
    return build_presentation_methodology_overview()


def save_figure(fig: plt.Figure, stem: str, out_dir: Path | None = None) -> None:
    target = out_dir or OUT
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{stem}.png"
    try:
        fig.savefig(path, facecolor="white", edgecolor="none")
        print(f"Saved {path}")
    except PermissionError:
        print(f"Skipped (file locked): {path}")


def save_thesis_methodology_figure(fig: plt.Figure) -> None:
    """Save under the official thesis name and the legacy alias."""
    save_figure(fig, THESIS_METHODOLOGY_STEM)
    save_figure(fig, THESIS_METHODOLOGY_ALIAS)


def main() -> None:
    fig_design = build_experimental_design_figure()
    save_figure(fig_design, "fig2_experimental_design")
    save_figure(fig_design, "fig2_experimental_design", out_dir=PRES_OUT)
    plt.close(fig_design)

    fig_paper = build_methodology_figure()
    save_thesis_methodology_figure(fig_paper)
    plt.close(fig_paper)

    fig_pres_overview = build_presentation_methodology_overview()
    save_figure(fig_pres_overview, "presentation_methodology_overview")
    save_figure(fig_pres_overview, "presentation_methodology_overview", out_dir=PRES_OUT)
    save_figure(fig_pres_overview, "presentation_methodology")  # legacy alias
    save_figure(fig_pres_overview, "presentation_methodology", out_dir=PRES_OUT)
    plt.close(fig_pres_overview)

    pres_details = [
        (build_presentation_detail_data_preparation, "presentation_detail_data_preparation"),
        (build_presentation_detail_rag, "presentation_detail_rag"),
        (build_presentation_detail_generation, "presentation_detail_generation"),
    ]
    for builder, stem in pres_details:
        fig = builder()
        save_figure(fig, stem, out_dir=PRES_OUT)
        plt.close(fig)


if __name__ == "__main__":
    main()
