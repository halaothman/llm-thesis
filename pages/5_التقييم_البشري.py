"""صفحة عرض نتائج التقييم البشري: جداول Mann-Whitney وShapiro وHolm."""
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import mannwhitneyu, shapiro

st.set_page_config(page_title="التقييم البشري", layout="wide")

with open("style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

HUMAN_METRICS = ["الوضوح اللغوي", "الصياغة المنطقية", "الملاءمة", "جودة الخيارات", "الدقة"]


def get_clean_human_df(data: dict, key: str) -> pd.DataFrame:
    """إزالة صف المتوسط وتحويل المقاييس البشرية إلى أرقام."""
    df = data[key]["df"].copy()
    df = df[df["#"].astype(str) != "المتوسط"].copy()
    for metric in HUMAN_METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def holm_bonferroni(p_values):
    """تصحيح Holm-Bonferroni لمجموعة قيم احتمالية."""
    p_values = np.asarray(p_values, dtype=float)
    if len(p_values) == 0:
        return p_values

    order = np.argsort(p_values)
    adjusted_sorted = np.empty(len(p_values))
    for i, idx in enumerate(order):
        adjusted_sorted[i] = (len(p_values) - i) * p_values[idx]
    for i in range(1, len(adjusted_sorted)):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])

    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)
    adjusted = np.empty(len(p_values))
    adjusted[order] = adjusted_sorted
    return adjusted


def describe_effect_size(rank_biserial: float) -> str:
    abs_rb = abs(rank_biserial)
    if abs_rb < 0.147:
        return "ضعيف جدًا"
    if abs_rb < 0.33:
        return "ضعيف"
    if abs_rb < 0.474:
        return "متوسط"
    return "قوي"


def build_human_statistical_results(data: dict):
    """حساب نتائج Mann-Whitney مع Holm وShapiro-Wilk."""
    required = ["llama_vanilla", "llama_rag", "qwen_vanilla", "qwen_rag"]
    if not all(key in data for key in required):
        return None, None, None

    groups = {
        "LLaMA": {
            "Vanilla": get_clean_human_df(data, "llama_vanilla"),
            "RAG": get_clean_human_df(data, "llama_rag"),
        },
        "Qwen": {
            "Vanilla": get_clean_human_df(data, "qwen_vanilla"),
            "RAG": get_clean_human_df(data, "qwen_rag"),
        },
    }

    detail_rows = []
    shapiro_rows = []

    for model, model_groups in groups.items():
        p_values = []
        model_rows = []

        for metric in HUMAN_METRICS:
            vanilla = model_groups["Vanilla"][metric].dropna().to_numpy()
            rag = model_groups["RAG"][metric].dropna().to_numpy()

            for method_name, values in (("Vanilla", vanilla), ("RAG", rag)):
                if 3 <= len(values) <= 5000:
                    stat, p_value = shapiro(values)
                    shapiro_rows.append(
                        {
                            "النموذج": model,
                            "الطريقة": method_name,
                            "المقياس": metric,
                            "حجم العينة": len(values),
                            "W": round(float(stat), 4),
                            "p": round(float(p_value), 4),
                            "الاستنتاج": "غير طبيعي" if p_value < 0.05 else "طبيعي تقريبًا",
                        }
                    )

            if len(vanilla) < 3 or len(rag) < 3:
                model_rows.append(
                    {
                        "النموذج": model,
                        "المقياس": metric,
                        "n (Vanilla)": len(vanilla),
                        "n (RAG)": len(rag),
                        "وسيط Vanilla": None,
                        "وسيط RAG": None,
                        "U": None,
                        "p": None,
                        "p بعد Holm": None,
                        "Rank-Biserial": None,
                        "حجم الأثر": "عينة غير كافية",
                        "الاتجاه": None,
                        "القرار": "غير كافٍ",
                    }
                )
                continue

            u_stat, p_value = mannwhitneyu(vanilla, rag, alternative="two-sided")
            rank_biserial = 1 - (2 * u_stat) / (len(vanilla) * len(rag))
            vanilla_median = float(np.median(vanilla))
            rag_median = float(np.median(rag))
            model_rows.append(
                {
                    "النموذج": model,
                    "المقياس": metric,
                    "n (Vanilla)": len(vanilla),
                    "n (RAG)": len(rag),
                    "وسيط Vanilla": round(vanilla_median, 2),
                    "وسيط RAG": round(rag_median, 2),
                    "U": round(float(u_stat), 2),
                    "p": round(float(p_value), 4),
                    "p_raw": float(p_value),
                    "p بعد Holm": None,
                    "Rank-Biserial": round(float(rank_biserial), 4),
                    "حجم الأثر": describe_effect_size(rank_biserial),
                    "الاتجاه": "RAG"
                    if rag_median > vanilla_median
                    else ("Vanilla" if vanilla_median > rag_median else "تعادل في الوسيط"),
                    "القرار": None,
                }
            )
            p_values.append(p_value)

        adjusted_p = holm_bonferroni(p_values)
        adjusted_idx = 0

        for row in model_rows:
            if row.get("p_raw") is None:
                continue
            row["p بعد Holm"] = round(float(adjusted_p[adjusted_idx]), 4)
            row["القرار"] = "دال" if adjusted_p[adjusted_idx] < 0.05 else "غير دال"
            adjusted_idx += 1

        detail_rows.extend(model_rows)

    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty and "p_raw" in detail_df.columns:
        detail_df = detail_df.drop(columns=["p_raw"])

    return detail_df, pd.DataFrame(shapiro_rows)


def load_human_evaluation_data() -> Dict[str, dict]:
    """تحميل بيانات التقييم البشري من CSV."""
    files = {
        "llama_vanilla": "llama_vanilla_human_evaluation.csv",
        "llama_rag": "llama_rag_human_evaluation.csv",
        "qwen_vanilla": "qwen_vanilla_human_evaluation.csv",
        "qwen_rag": "qwen_rag_human_evaluation.csv",
    }

    data = {}
    for key, filename in files.items():
        file_path = Path(filename)
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, encoding="utf-8-sig")
                avg_row = df[df["#"].astype(str) == "المتوسط"]
                if not avg_row.empty:
                    data[key] = {"df": df, "averages": avg_row.iloc[0].to_dict()}
            except Exception as e:
                st.error(f"خطأ في قراءة {filename}: {e}")
        else:
            st.warning(f"الملف غير موجود: {filename}")

    return data


def main():
    st.title("التقييم البشري")

    data = load_human_evaluation_data()

    if not data:
        st.error("لم يتم العثور على ملفات التقييم البشري!")
        st.info(
            "**الملفات المطلوبة:**\n"
            "- `llama_vanilla_human_evaluation.csv`\n"
            "- `llama_rag_human_evaluation.csv`\n"
            "- `qwen_vanilla_human_evaluation.csv`\n"
            "- `qwen_rag_human_evaluation.csv`"
        )
        return

    if len(data) < 4:
        st.warning(f"تم تحميل {len(data)} من 4 ملفات فقط")

    stats_detail_df, shapiro_df = build_human_statistical_results(data)

    if stats_detail_df is not None and not stats_detail_df.empty:
        st.subheader("نتائج Mann-Whitney U و Rank-Biserial (Holm)")
        tabs = st.tabs(["النتائج المختصرة", "LLaMA", "Qwen", "Shapiro-Wilk"])

        with tabs[0]:
            display_df = stats_detail_df[
                [
                    "النموذج",
                    "المقياس",
                    "n (Vanilla)",
                    "n (RAG)",
                    "وسيط Vanilla",
                    "وسيط RAG",
                    "p",
                    "p بعد Holm",
                    "Rank-Biserial",
                    "حجم الأثر",
                    "القرار",
                ]
            ].copy()
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        with tabs[1]:
            st.dataframe(
                stats_detail_df[stats_detail_df["النموذج"] == "LLaMA"],
                use_container_width=True,
                hide_index=True,
            )

        with tabs[2]:
            st.dataframe(
                stats_detail_df[stats_detail_df["النموذج"] == "Qwen"],
                use_container_width=True,
                hide_index=True,
            )

        with tabs[3]:
            st.dataframe(shapiro_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
