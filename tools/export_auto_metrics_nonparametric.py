"""Mann-Whitney U, rank-biserial, and Holm correction for automatic metrics (JSON corpus)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
OUT_PATH = OUTPUTS / "automatic_evaluation_nonparametric.csv"

METRICS = ["precision", "recall", "f1_score", "bleu", "bert_score", "perplexity"]
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1",
    "bleu": "BLEU",
    "bert_score": "BERTScore",
    "perplexity": "Perplexity",
}

COMPARISONS = [
    ("Vanilla vs RAG-improved", "Vanilla", "RAG-improved"),
    ("Vanilla vs RAG-baseline", "Vanilla", "RAG-baseline"),
    ("Vanilla vs RAG-all", "Vanilla", "RAG-all"),
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


def rag_condition(path: Path) -> str | None:
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    method = parts[2]
    if method == "vanilla":
        return "Vanilla"
    if method == "rag":
        return "RAG-improved" if path.name.endswith("_new.json") else "RAG-baseline"
    return None


def load_question_metrics() -> pd.DataFrame:
    rows = []
    for path in OUTPUTS.rglob("questions_*.json"):
        if "UPDATED" in path.stem.upper():
            continue
        cond = rag_condition(path)
        if cond is None:
            continue
        model = path.stem.split("_")[1]
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        for q in data.get("questions", []):
            m = q.get("metrics") or {}
            ppl = m.get("perplexity")
            if ppl in (0.0, 1000.0, None):
                continue
            row = {
                "model": {"llama": "LLaMA", "qwen": "Qwen"}.get(model, model),
                "condition": cond,
            }
            for key in METRICS:
                v = m.get(key)
                row[key] = float(v) if isinstance(v, (int, float)) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def add_rag_all(df: pd.DataFrame) -> pd.DataFrame:
    rag = df[df["condition"].isin(["RAG-baseline", "RAG-improved"])].copy()
    rag["condition"] = "RAG-all"
    return pd.concat([df, rag], ignore_index=True)


def compare_group(
    df: pd.DataFrame,
    model: str,
    comparison: str,
    vanilla_label: str,
    rag_label: str,
) -> list[dict]:
    sub = df[df["model"] == model]
    rows: list[dict] = []
    p_raw_list: list[float] = []
    pending: list[dict] = []

    for metric in METRICS:
        vanilla = sub.loc[sub["condition"] == vanilla_label, metric].dropna().to_numpy()
        rag = sub.loc[sub["condition"] == rag_label, metric].dropna().to_numpy()

        row = {
            "model": model,
            "comparison": comparison,
            "metric": metric,
            "metric_label": METRIC_LABELS[metric],
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

    return rows


def main() -> None:
    df = load_question_metrics()
    if df.empty:
        raise SystemExit("No automatic metric data found in outputs/.")

    df = add_rag_all(df)
    all_rows: list[dict] = []

    for model in ("LLaMA", "Qwen"):
        for comparison, vanilla_label, rag_label in COMPARISONS:
            all_rows.extend(compare_group(df, model, comparison, vanilla_label, rag_label))

    out_df = pd.DataFrame(all_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved {len(out_df)} rows to {OUT_PATH}\n")
    for comparison in [c[0] for c in COMPARISONS]:
        print(f"=== {comparison} ===")
        part = out_df[out_df["comparison"] == comparison]
        print(
            part[
                [
                    "model",
                    "metric_label",
                    "mean_vanilla",
                    "mean_rag",
                    "p_raw",
                    "p_holm",
                    "rank_biserial_r_rb",
                    "significant_after_holm",
                ]
            ].to_string(index=False)
        )
        print()


if __name__ == "__main__":
    main()
