"""Export Mann-Whitney U, rank-biserial r_rb, and Holm-corrected p-values from human-eval CSVs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "outputs" / "human_evaluation_nonparametric.csv"

FILES = {
    "LLaMA": {
        "Vanilla": ROOT / "llama_vanilla_human_evaluation.csv",
        "RAG": ROOT / "llama_rag_human_evaluation.csv",
    },
    "Qwen": {
        "Vanilla": ROOT / "qwen_vanilla_human_evaluation.csv",
        "RAG": ROOT / "qwen_rag_human_evaluation.csv",
    },
}

METRICS = [
    ("الوضوح اللغوي", "linguistic_clarity"),
    ("الصياغة المنطقية", "logical_formulation"),
    ("الملاءمة", "relevance"),
    ("جودة الخيارات", "option_quality"),
    ("الدقة", "accuracy"),
]


def holm_bonferroni(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    if len(p) == 0:
        return []
    order = np.argsort(p)
    adjusted_sorted = np.empty(len(p))
    for i, idx in enumerate(order):
        adjusted_sorted[i] = (len(p) - i) * p[idx]
    for i in range(1, len(adjusted_sorted)):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)
    adjusted = np.empty(len(p))
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def effect_magnitude(r_rb: float) -> str:
    abs_rb = abs(r_rb)
    if abs_rb < 0.147:
        return "Small"
    if abs_rb < 0.33:
        return "Medium"
    if abs_rb < 0.474:
        return "Large"
    return "Very large"


def load_scores(path: Path, column: str) -> np.ndarray:
    df = pd.read_csv(path, encoding="utf-8")
    if "#" in df.columns:
        df = df[df["#"].astype(str) != "المتوسط"]
    return pd.to_numeric(df[column], errors="coerce").dropna().to_numpy()


def main() -> None:
    rows: list[dict] = []

    for model, paths in FILES.items():
        p_raw_list: list[float] = []
        pending: list[dict] = []

        for arabic_col, criterion in METRICS:
            vanilla = load_scores(paths["Vanilla"], arabic_col)
            rag = load_scores(paths["RAG"], arabic_col)

            row: dict = {
                "model": model,
                "criterion": criterion,
                "criterion_ar": arabic_col,
                "n_vanilla": int(len(vanilla)),
                "n_rag": int(len(rag)),
                "mean_vanilla": round(float(np.mean(vanilla)), 4) if len(vanilla) else None,
                "mean_rag": round(float(np.mean(rag)), 4) if len(rag) else None,
                "median_vanilla": round(float(np.median(vanilla)), 4) if len(vanilla) else None,
                "median_rag": round(float(np.median(rag)), 4) if len(rag) else None,
                "mannwhitney_u": None,
                "p_raw": None,
                "p_holm": None,
                "rank_biserial_r_rb": None,
                "effect_magnitude": None,
                "significant_after_holm": None,
            }

            if len(vanilla) < 3 or len(rag) < 3:
                row["effect_magnitude"] = "Insufficient sample"
                rows.append(row)
                continue

            u_stat, p_value = mannwhitneyu(vanilla, rag, alternative="two-sided")
            r_rb = 1 - (2 * u_stat) / (len(vanilla) * len(rag))

            row.update(
                {
                    "mannwhitney_u": round(float(u_stat), 4),
                    "p_raw": round(float(p_value), 4),
                    "rank_biserial_r_rb": round(float(r_rb), 4),
                    "effect_magnitude": effect_magnitude(r_rb),
                }
            )
            p_raw_list.append(float(p_value))
            pending.append(row)

        adjusted = holm_bonferroni(p_raw_list)
        for row, p_holm in zip(pending, adjusted):
            row["p_holm"] = round(p_holm, 4)
            row["significant_after_holm"] = "Yes" if p_holm < 0.05 else "No"
            rows.append(row)

    out_df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved {len(out_df)} rows to {OUT_PATH}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
