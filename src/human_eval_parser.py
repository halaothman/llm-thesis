"""تحليل ملاحظات التقييم البشري الخام وتحويلها إلى CSV منظم للتحليل."""
import csv
import re
from pathlib import Path

RAW_PATH = Path("uploads/human_eval_notes_raw.txt")
OUT_PATH = Path("uploads/human_evaluation_raw.csv")

# Expected order per question: linguistic_clarity, logical_formulation, relevance, options_quality, accuracy, non_repetition
METRIC_ORDER = [
    "linguistic_clarity",
    "logical_formulation",
    "relevance",
    "options_quality",
    "accuracy",
    "non_repetition",
]

GROUP_HEADERS = [
    ("LLaMA 3.2:3B - Vanilla", ("Llama3.2:3b", "Vanilla")),
    ("Qwen2.5:7B - Vanilla", ("Qwen2.5:7b", "Vanilla")),
    ("LLaMA 3.2:3B - RAG", ("Llama3.2:3b", "RAG")),
    ("Qwen2.5:7B - RAG", ("Qwen2.5:7b", "RAG")),
]

num_re = re.compile(r"(?<!\d)([1-6](?:\.\d+)?)")


def extract_groups(text: str):
    groups = []
    for header, group in GROUP_HEADERS:
        idx = text.find(header)
        if idx >= 0:
            groups.append((idx, header, group))
    groups.sort(key=lambda x: x[0])

    segments = []
    for i, (start_idx, header, group) in enumerate(groups):
        end_idx = groups[i + 1][0] if i + 1 < len(groups) else len(text)
        segments.append((header, group, text[start_idx:end_idx]))
    return segments


def parse_sequential_20x6(segment_text: str):
    # Extract first 120 numbers and map to qid 1..20, 6 metrics each in METRIC_ORDER
    numbers = [float(x) for x in num_re.findall(segment_text)]
    qid_to_metrics = {}
    needed = 20 * 6
    take = numbers[:needed]
    for qid in range(1, 21):
        start = (qid - 1) * 6
        chunk = take[start:start + 6]
        metrics = {}
        for i, metric in enumerate(METRIC_ORDER):
            metrics[metric] = chunk[i] if i < len(chunk) else ""
        qid_to_metrics[qid] = metrics
    return qid_to_metrics


def parse():
    text = RAW_PATH.read_text(encoding="utf-8", errors="ignore")
    segments = extract_groups(text)

    rows = []
    for header, (model_name, method), segment in segments:
        # Use sequential extraction for all groups to be robust to free-form notes
        qid_to_metrics = parse_sequential_20x6(segment)

        for qid in sorted(qid_to_metrics.keys()):
            metrics = qid_to_metrics[qid]
            vals = [v for v in metrics.values() if isinstance(v, (int, float))]
            overall_quality = round(sum(vals) / len(vals), 3) if vals else ""
            rows.append({
                "model_name": model_name,
                "method": method,
                "question_id": qid,
                "overall_quality": overall_quality,
                "non_repetition": metrics.get("non_repetition", ""),
                "accuracy": metrics.get("accuracy", ""),
                "options_quality": metrics.get("options_quality", ""),
                "relevance": metrics.get("relevance", ""),
                "logical_formulation": metrics.get("logical_formulation", ""),
                "linguistic_clarity": metrics.get("linguistic_clarity", ""),
            })

    rows.sort(key=lambda r: (r["model_name"], r["method"], int(r["question_id"])) )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "method",
                "question_id",
                "overall_quality",
                "non_repetition",
                "accuracy",
                "options_quality",
                "relevance",
                "logical_formulation",
                "linguistic_clarity",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    parse()
