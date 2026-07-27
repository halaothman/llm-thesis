"""صفحة المقارنة والتحليل: إحصائيات ورسوم بيانية لمقارنة Vanilla مقابل RAG."""
import json
import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from scipy.stats import mannwhitneyu, shapiro

st.set_page_config(page_title="📊 المقارنة والتحليل", layout="wide")

try:
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

OUTPUTS_DIR = Path("outputs")
PLOTS_DIR = Path("plots")
METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1_score",
    "bleu",
    "bert_score",
    "perplexity",
]
SUMMARY_COLUMNS = METRIC_COLUMNS + ["log_perplexity"]
MODEL_DISPLAY = {
    "llama": "LLaMA",
    "qwen": "Qwen",
}
METHOD_DISPLAY = {
    "vanilla": "Vanilla",
    "rag": "RAG",
}


def rtl_markdown(text: str):
    """عرض Markdown أو HTML مع RTL. يزيل المسافات البادئة المشتركة حتى لا يُفسَّر المحتوى ككتلة كود."""
    text = textwrap.dedent(text).strip()
    # لا تلفّ النص داخل <div> فقط؛ داخل HTML لا يُعالج Markdown (# العناوين تظهر كنص خام)
    st.markdown(text, unsafe_allow_html=True)


def embed_plotly(fig, height_px: int) -> None:
    """
    عرض مخطط Plotly عبر iframe مخصّص بدل st.plotly_chart الافتراضي.
    يخفّف قصّ الرسم الشائع مع وضع RTL/الأعمدة لأن الارتفاع يُضبط صراحةً ويمكن التمرير داخل الإطار.
    """
    fig.update_layout(
        height=height_px,
        autosize=False,
        margin=dict(l=52, r=44, t=76, b=56),
    )
    html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": True, "responsive": True},
    )
    # عزل اتجاه الرسم (LTR) عن صفحة التطبيق RTL
    html = html.replace("<html", '<html dir="ltr"', 1)
    components.html(html, height=height_px + 72, scrolling=True)


def clean_source_name(source: str) -> str:
    name = Path(source).stem
    return name.replace("_", " ")


def parse_filename(path: Path):
    """تحليل اسم الملف لاستخراج النموذج والطريقة والمصدر والإصدار"""
    stem = path.stem
    parts = stem.split("_")

    model = "unknown"
    method = "unknown"
    source = clean_source_name(stem)
    version = "before"  # قبل التحسين (افتراضي)

    # التحقق من وجود "_new" في اسم الملف (بعد التحسين)
    if "_new" in stem:
        version = "after"  # بعد التحسين
        # إزالة "_new" من الاسم للمعالجة
        stem_clean = stem.replace("_new", "")
        parts_clean = stem_clean.split("_")
    else:
        parts_clean = parts

    # استخراج المعلومات من أجزاء الاسم
    if len(parts_clean) >= 4 and parts_clean[0] == "questions":
        model = parts_clean[1]
        method = parts_clean[2]
        # الباقي هو اسم المصدر
        source = "_".join(parts_clean[3:])

    model_disp = MODEL_DISPLAY.get(model, model)
    method_disp = METHOD_DISPLAY.get(method, method)
    return model_disp, method_disp, source, version


def list_question_files(exclude_updated: bool = True):
    """
    قائمة بملفات الأسئلة

    Args:
        exclude_updated: إذا كان True، يتم استثناء الملفات التي تحتوي على "UPDATED" في الاسم
    """
    if not OUTPUTS_DIR.exists():
        return []
    files = []
    for path in OUTPUTS_DIR.rglob("questions_*.json"):
        if path.is_file():
            # استثناء الملفات التي تحتوي على "UPDATED" إذا كان exclude_updated=True
            if exclude_updated and "UPDATED" in path.stem.upper():
                continue
            files.append(path)
    return sorted(files)


def get_files_hash():
    """إنشاء hash من قائمة الملفات وأوقات التعديل للتحقق من التغييرات"""
    files = list_question_files()
    if not files:
        return "no_files"
    # استخدام أسماء الملفات وأوقات التعديل كـ hash
    file_info = [(str(f), f.stat().st_mtime) for f in files]
    return hash(tuple(sorted(file_info)))


@st.cache_data
def load_question_frames(_files_hash: str):
    """
    تحميل جميع ملفات الأسئلة وتحويلها إلى DataFrames
    مع إزالة التكرار واستبعاد الأسئلة ذات perplexity غير صالح

    Args:
        _files_hash: hash للملفات (يستخدم من قبل Streamlit للتحقق من التغييرات)

    Returns:
        tuple: (summary_df, question_df, cleaning_stats)
        cleaning_stats: dict يحتوي على إحصائيات التنظيف
    """
    from collections import OrderedDict

    summary_rows = []
    question_rows = []

    # إحصائيات التنظيف
    cleaning_stats = {
        "total_questions_before": 0,
        "duplicates_removed": 0,
        "invalid_perplexity_excluded": 0,
        "total_questions_after": 0,
        "duplicates_by_file": {},
        "invalid_perplexity_by_file": {}
    }

    # لتتبع التكرار عبر جميع الملفات (منفصل لكل نموذج/طريقة/فترة)
    seen_questions = OrderedDict()

    for file_path in list_question_files():
        model, method, source, version = parse_filename(file_path)
        with file_path.open(encoding="utf-8") as f:
            data = json.load(f)
        questions = data.get("questions", [])
        num_questions_before = len(questions)
        cleaning_stats["total_questions_before"] += num_questions_before

        source_metrics = {metric: [] for metric in METRIC_COLUMNS}

        # إزالة التكرار داخل الملف
        file_seen = OrderedDict()
        file_duplicates = 0

        for q in questions:
            question_text = q.get("question", "").strip()

            # تخطي الأسئلة الفارغة
            if not question_text:
                continue

            # التحقق من التكرار داخل الملف
            if question_text in file_seen:
                file_duplicates += 1
                continue

            # التحقق من التكرار عبر جميع الملفات (لا ندمج قبل/بعد التحسين)
            dedup_key = (question_text, model, method, version)
            if dedup_key in seen_questions:
                cleaning_stats["duplicates_removed"] += 1
                if file_path.name not in cleaning_stats["duplicates_by_file"]:
                    cleaning_stats["duplicates_by_file"][file_path.name] = 0
                cleaning_stats["duplicates_by_file"][file_path.name] += 1
                continue

            # التحقق من perplexity غير صالح (1000.0 فقط)
            # لا نستبعد perplexity = 0.0 هنا، سنستبعده فقط عند حساب/عرض مقياس perplexity
            metrics = q.get("metrics") or {}
            perplexity = metrics.get("perplexity")

            # استبعاد فقط إذا كان perplexity = 1000.0 (فشل واضح)
            if perplexity == 1000.0:
                cleaning_stats["invalid_perplexity_excluded"] += 1
                if file_path.name not in cleaning_stats["invalid_perplexity_by_file"]:
                    cleaning_stats["invalid_perplexity_by_file"][file_path.name] = 0
                cleaning_stats["invalid_perplexity_by_file"][file_path.name] += 1
                continue

            # إضافة السؤال إلى القوائم
            file_seen[question_text] = q
            seen_questions[dedup_key] = q

            row = {
                "file": str(file_path.relative_to(OUTPUTS_DIR)),
                "model": model,
                "method": method,
                "source": source,
                "version": version,
                "question_text": question_text,  # للحفاظ على نص السؤال للتحقق
            }
            for metric in METRIC_COLUMNS:
                value = metrics.get(metric)
                if isinstance(value, (int, float)):
                    row[metric] = float(value)
                    source_metrics[metric].append(float(value))
                else:
                    row[metric] = np.nan
            row["log_perplexity"] = (
                math.log(row["perplexity"]) if isinstance(row["perplexity"], float) and row["perplexity"] > 0 else np.nan
            )
            question_rows.append(row)

        num_questions_after = len(file_seen)
        cleaning_stats["total_questions_after"] += num_questions_after

        summary = {
            "ملف الأسئلة": str(file_path.relative_to(OUTPUTS_DIR)),
            "النموذج": model,
            "الطريقة": method,
            "المصدر": source,
            "الإصدار": version,
            "عدد الأسئلة": num_questions_after,  # بعد التنظيف
        }

        for metric in METRIC_COLUMNS:
            values = [v for v in source_metrics[metric] if isinstance(v, float)]
            summary[metric] = np.mean(values) if values else np.nan

        if summary["perplexity"] and summary["perplexity"] > 0:
            summary["log_perplexity"] = math.log(summary["perplexity"])
        else:
            summary["log_perplexity"] = np.nan

        summary_rows.append(summary)

    # إزالة عمود question_text من DataFrame النهائي (كان للتحقق فقط)
    if question_rows:
        for row in question_rows:
            row.pop("question_text", None)

    summary_df = pd.DataFrame(summary_rows)
    question_df = pd.DataFrame(question_rows)
    return summary_df, question_df, cleaning_stats


def filter_df_by_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    تصفية DataFrame حسب المقياس المحدد
    - إذا كان المقياس = perplexity أو log_perplexity: استبعاد perplexity = 0.0 أو 1000.0
    - إذا كان المقياس آخر: فقط استبعاد NaN للمقياس نفسه
    """
    if metric in ["perplexity", "log_perplexity"]:
        # استبعاد perplexity = 0.0 أو 1000.0
        filtered = df[
            (df["perplexity"].notna()) &
            (df["perplexity"] != 0.0) &
            (df["perplexity"] != 1000.0)
        ]
        # للـ log_perplexity، تأكد من أنه ليس NaN
        if metric == "log_perplexity":
            filtered = filtered[filtered["log_perplexity"].notna()]
        return filtered
    else:
        # للمقاييس الأخرى، فقط استبعاد NaN
        return df.dropna(subset=[metric])


def shapiro_section(question_df: pd.DataFrame, metric: str):
    sub_df = filter_df_by_metric(question_df, metric)

    results = []
    for (model, method), group in sub_df.groupby(["model", "method"]):
        values = group[metric].to_numpy()
        n = len(values)
        if n < 3 or n > 5000:
            results.append(
                {
                    "النموذج": model,
                    "الطريقة": method,
                    "حجم العينة": n,
                    "إحصائية شابيرو": None,
                    "القيمة الاحتمالية": None,
                    "الاستنتاج": "العينة خارج نطاق اختبار شابيرو",
                }
            )
            continue
        stat, p_value = shapiro(values)
        conclusion = "لا نرفض الفرضية الصفرية (توزيع طبيعي)" if p_value > 0.05 else "نرفض الفرضية الصفرية (توزيع غير طبيعي)"
        results.append(
            {
                "النموذج": model,
                "الطريقة": method,
                "حجم العينة": n,
                "إحصائية شابيرو": round(float(stat), 4),
                "القيمة الاحتمالية": round(float(p_value), 4),
                "الاستنتاج": conclusion,
            }
        )

    if not results:
        rtl_markdown("لا توجد بيانات كافية لتنفيذ اختبار شابيرو لهذا المقياس.")
        return

    st.dataframe(pd.DataFrame(results), use_container_width=True)
    rtl_markdown(
        "<small><strong>ملاحظة:</strong> البيانات المستخدمة في الاختبار تم تنظيفها من التكرار واستبعاد القيم غير الصالحة.</small>"
    )


def mann_whitney_section(question_df: pd.DataFrame, metric: str):
    rtl_markdown("### اختبار مان ويتني (Vanilla مقابل RAG)")
    results = []

    # حساب الأحجام الإجمالية مرة واحدة لكل نموذج (قبل أي dropna)
    # هذا يضمن أن الأحجام ثابتة بغض النظر عن المقياس المحدد
    total_counts = question_df.groupby(["model", "method"]).size().unstack(fill_value=0)

    for model, group in question_df.groupby("model"):
        # الحصول على العدد الإجمالي من total_counts (ثابت بغض النظر عن المقياس)
        try:
            vanilla_total = int(total_counts.loc[model, "Vanilla"]) if "Vanilla" in total_counts.columns else 0
        except (KeyError, IndexError):
            vanilla_total = len(group[group["method"] == "Vanilla"])

        try:
            rag_total = int(total_counts.loc[model, "RAG"]) if "RAG" in total_counts.columns else 0
        except (KeyError, IndexError):
            rag_total = len(group[group["method"] == "RAG"])

        # استخدام filter_df_by_metric فقط للقيم المستخدمة في الاختبار
        group_with_metric = filter_df_by_metric(group, metric)
        vanilla = group_with_metric[group_with_metric["method"] == "Vanilla"][metric].to_numpy()
        rag = group_with_metric[group_with_metric["method"] == "RAG"][metric].to_numpy()

        if len(vanilla) < 3 or len(rag) < 3:
            results.append(
                {
                    "النموذج": model,
                    "حجم Vanilla": vanilla_total,
                    "حجم RAG": rag_total,
                    "عدد القيم الصالحة Vanilla": len(vanilla),
                    "عدد القيم الصالحة RAG": len(rag),
                    "إحصائية U": None,
                    "القيمة الاحتمالية": None,
                    "الرتبة المتسلسلة": None,
                    "الاستنتاج": "عينة غير كافية لإجراء الاختبار",
                }
            )
            continue

        alternative = "two-sided"
        u_stat, p_value = mannwhitneyu(vanilla, rag, alternative=alternative)
        n1 = len(vanilla)
        n2 = len(rag)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

        if abs(rank_biserial) < 0.147:
            effect = "تأثير ضعيف جدًا"
        elif abs(rank_biserial) < 0.33:
            effect = "تأثير ضعيف"
        elif abs(rank_biserial) < 0.474:
            effect = "تأثير متوسط"
        else:
            effect = "تأثير قوي"

        conclusion = "يوجد فرق ذو دلالة" if p_value < 0.05 else "لا يوجد فرق ذو دلالة"

        results.append(
            {
                "النموذج": model,
                "حجم Vanilla": vanilla_total,
                "حجم RAG": rag_total,
                "عدد القيم الصالحة Vanilla": n1,
                "عدد القيم الصالحة RAG": n2,
                "وسيط Vanilla": round(float(np.median(vanilla)), 4),
                "وسيط RAG": round(float(np.median(rag)), 4),
                "إحصائية U": round(float(u_stat), 2),
                "القيمة الاحتمالية": round(float(p_value), 4),
                "الحجم التأثيري (Rank-Biserial)": round(float(rank_biserial), 4),
                "تفسير الحجم التأثيري": effect,
                "الاستنتاج": conclusion,
            }
        )

    st.dataframe(pd.DataFrame(results), use_container_width=True)
    rtl_markdown(
        "<small><strong>توجيه القراءة:</strong> إذا كانت القيمة الاحتمالية أقل من 0.05 فهذا يشير إلى فرق ذي دلالة إحصائية، "
        "بينما يوضح الحجم التأثيري قوة هذا الاختلاف واتجاهه.<br>"
        "<strong>ملاحظة:</strong> النتائج تستند إلى الأسئلة بعد إزالة التكرار واستبعاد القيم غير الصالحة.</small>"
    )


def mann_whitney_by_version_section(question_df: pd.DataFrame, metric: str):
    """مقارنة Vanilla vs RAG قبل وبعد التحسين"""
    rtl_markdown("### اختبار مان ويتني: Vanilla مقابل RAG (مقسم حسب فترة التحسين)")
    results = []

    for (model, version), group in filter_df_by_metric(question_df, metric).groupby(["model", "version"]):
        vanilla = group[group["method"] == "Vanilla"][metric].to_numpy()
        rag = group[group["method"] == "RAG"][metric].to_numpy()

        if len(vanilla) < 3 or len(rag) < 3:
            results.append(
                {
                    "النموذج": model,
                    "الفترة": "قبل التحسين" if version == "before" else "بعد التحسين",
                    "حجم Vanilla": len(vanilla),
                    "حجم RAG": len(rag),
                    "إحصائية U": None,
                    "القيمة الاحتمالية": None,
                    "الرتبة المتسلسلة": None,
                    "الاستنتاج": "عينة غير كافية لإجراء الاختبار",
                }
            )
            continue

        alternative = "two-sided"
        u_stat, p_value = mannwhitneyu(vanilla, rag, alternative=alternative)
        n1 = len(vanilla)
        n2 = len(rag)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

        if abs(rank_biserial) < 0.147:
            effect = "تأثير ضعيف جدًا"
        elif abs(rank_biserial) < 0.33:
            effect = "تأثير ضعيف"
        elif abs(rank_biserial) < 0.474:
            effect = "تأثير متوسط"
        else:
            effect = "تأثير قوي"

        conclusion = "يوجد فرق ذو دلالة" if p_value < 0.05 else "لا يوجد فرق ذو دلالة"

        # تحديد من الأفضل (للأسئلة التي تعتبر قيمة أعلى أفضل)
        if metric in ["precision", "recall", "f1_score", "bleu", "bert_score"]:
            better = "RAG" if np.median(rag) > np.median(vanilla) else "Vanilla"
        else:  # perplexity - قيمة أقل أفضل
            better = "RAG" if np.median(rag) < np.median(vanilla) else "Vanilla"

        results.append(
            {
                "النموذج": model,
                "الفترة": "قبل التحسين" if version == "before" else "بعد التحسين",
                "حجم Vanilla": n1,
                "حجم RAG": n2,
                "وسيط Vanilla": round(float(np.median(vanilla)), 4),
                "وسيط RAG": round(float(np.median(rag)), 4),
                "إحصائية U": round(float(u_stat), 2),
                "القيمة الاحتمالية": round(float(p_value), 4),
                "الحجم التأثيري (Rank-Biserial)": round(float(rank_biserial), 4),
                "تفسير الحجم التأثيري": effect,
                "الأفضل": better if p_value < 0.05 else "لا فرق ذو دلالة",
                "الاستنتاج": conclusion,
            }
        )

    if not results:
        rtl_markdown("لا توجد بيانات كافية للمقارنة.")
        return

    st.dataframe(pd.DataFrame(results), use_container_width=True)
    rtl_markdown(
        "<small>هذه المقارنة تُظهر أداء Vanilla مقابل RAG في كل فترة (قبل وبعد التحسين) بشكل منفصل.</small>"
    )


def mann_whitney_by_version_filtered(question_df: pd.DataFrame, metric: str, version_filter: str):
    """مقارنة مباشرة بين RAG و Vanilla في فترة محددة (قبل أو بعد التحسين)"""
    version_name = "بعد التحسين" if version_filter == "after" else "قبل التحسين"
    rtl_markdown(f"### اختبار مان ويتني: RAG مقابل Vanilla ({version_name})")
    results = []

    filtered_df = filter_df_by_metric(question_df[question_df["version"] == version_filter], metric)

    if filtered_df.empty:
        rtl_markdown(f"لا توجد بيانات {version_name} للمقارنة.")
        return

    for model, group in filtered_df.groupby("model"):
        vanilla = group[group["method"] == "Vanilla"][metric].to_numpy()
        rag = group[group["method"] == "RAG"][metric].to_numpy()

        if len(vanilla) < 3 or len(rag) < 3:
            results.append(
                {
                    "النموذج": MODEL_DISPLAY.get(model, model),
                    "حجم Vanilla": len(vanilla),
                    "حجم RAG": len(rag),
                    "متوسط Vanilla": None,
                    "متوسط RAG": None,
                    "Δ (RAG - Vanilla)": None,
                    "وسيط Vanilla": None,
                    "وسيط RAG": None,
                    "إحصائية U": None,
                    "القيمة الاحتمالية": None,
                    "الحجم التأثيري (Rank-Biserial)": None,
                    "تفسير الحجم التأثيري": None,
                    "الاستنتاج": "عينة غير كافية لإجراء الاختبار",
                }
            )
            continue

        vanilla_mean = np.mean(vanilla)
        rag_mean = np.mean(rag)
        delta = rag_mean - vanilla_mean

        alternative = "two-sided"
        u_stat, p_value = mannwhitneyu(vanilla, rag, alternative=alternative)
        n1 = len(vanilla)
        n2 = len(rag)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

        if abs(rank_biserial) < 0.147:
            effect = "تأثير ضعيف جدًا"
        elif abs(rank_biserial) < 0.33:
            effect = "تأثير ضعيف"
        elif abs(rank_biserial) < 0.474:
            effect = "تأثير متوسط"
        else:
            effect = "تأثير قوي"

        conclusion = "يوجد فرق ذو دلالة" if p_value < 0.05 else "لا يوجد فرق ذو دلالة"

        if metric in ["precision", "recall", "f1_score", "bleu", "bert_score"]:
            better = "RAG" if rag_mean > vanilla_mean else "Vanilla"
        else:
            better = "RAG" if rag_mean < vanilla_mean else "Vanilla"

        results.append(
            {
                "النموذج": MODEL_DISPLAY.get(model, model),
                "حجم Vanilla": n1,
                "حجم RAG": n2,
                "متوسط Vanilla": round(float(vanilla_mean), 4),
                "متوسط RAG": round(float(rag_mean), 4),
                "Δ (RAG - Vanilla)": round(float(delta), 4),
                "وسيط Vanilla": round(float(np.median(vanilla)), 4),
                "وسيط RAG": round(float(np.median(rag)), 4),
                "إحصائية U": round(float(u_stat), 2),
                "القيمة الاحتمالية": round(float(p_value), 4),
                "الحجم التأثيري (Rank-Biserial)": round(float(rank_biserial), 4),
                "تفسير الحجم التأثيري": effect,
                "الأفضل": better if p_value < 0.05 else "لا فرق ذو دلالة",
                "الاستنتاج": conclusion,
            }
        )

    if not results:
        rtl_markdown("لا توجد بيانات كافية للمقارنة.")
        return

    st.dataframe(pd.DataFrame(results), use_container_width=True)
    rtl_markdown(
        f"<small><strong>ملاحظة:</strong> هذه المقارنة تُظهر أثر RAG على توليد الأسئلة في نسخة {version_name}.</small>"
    )


def mann_whitney_before_after_section(question_df: pd.DataFrame, metric: str):
    """مقارنة قبل التحسين vs بعد التحسين لكل طريقة"""
    rtl_markdown("### اختبار مان ويتني: قبل التحسين مقابل بعد التحسين (مقسم حسب النموذج والطريقة)")
    results = []

    for (model, method), group in filter_df_by_metric(question_df, metric).groupby(["model", "method"]):
        before = group[group["version"] == "before"][metric].to_numpy()
        after = group[group["version"] == "after"][metric].to_numpy()

        if len(before) < 3 or len(after) < 3:
            results.append(
                {
                    "النموذج": model,
                    "الطريقة": method,
                    "حجم قبل التحسين": len(before),
                    "حجم بعد التحسين": len(after),
                    "إحصائية U": None,
                    "القيمة الاحتمالية": None,
                    "الرتبة المتسلسلة": None,
                    "الاستنتاج": "عينة غير كافية لإجراء الاختبار",
                }
            )
            continue

        alternative = "two-sided"
        u_stat, p_value = mannwhitneyu(before, after, alternative=alternative)
        n1 = len(before)
        n2 = len(after)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

        if abs(rank_biserial) < 0.147:
            effect = "تأثير ضعيف جدًا"
        elif abs(rank_biserial) < 0.33:
            effect = "تأثير ضعيف"
        elif abs(rank_biserial) < 0.474:
            effect = "تأثير متوسط"
        else:
            effect = "تأثير قوي"

        conclusion = "يوجد فرق ذو دلالة" if p_value < 0.05 else "لا يوجد فرق ذو دلالة"

        # تحديد ما إذا كان التحسين حسّن الأداء
        if metric in ["precision", "recall", "f1_score", "bleu", "bert_score"]:
            improved = "تحسن" if np.median(after) > np.median(before) else "تراجع"
        else:  # perplexity - قيمة أقل أفضل
            improved = "تحسن" if np.median(after) < np.median(before) else "تراجع"

        results.append(
            {
                "النموذج": model,
                "الطريقة": method,
                "حجم قبل التحسين": n1,
                "حجم بعد التحسين": n2,
                "وسيط قبل التحسين": round(float(np.median(before)), 4),
                "وسيط بعد التحسين": round(float(np.median(after)), 4),
                "إحصائية U": round(float(u_stat), 2),
                "القيمة الاحتمالية": round(float(p_value), 4),
                "الحجم التأثيري (Rank-Biserial)": round(float(rank_biserial), 4),
                "تفسير الحجم التأثيري": effect,
                "التأثير": improved if p_value < 0.05 else "لا فرق ذو دلالة",
                "الاستنتاج": conclusion,
            }
        )

    if not results:
        rtl_markdown("لا توجد بيانات كافية للمقارنة.")
        return

    st.dataframe(pd.DataFrame(results), use_container_width=True)
    rtl_markdown(
        "<small>هذه المقارنة تُظهر تأثير التحسين على أداء كل طريقة (Vanilla/RAG) لكل نموذج بشكل منفصل.</small>"
    )


def render_comparison_plots(question_df: pd.DataFrame, metric: str, use_adaptive_range: bool = True):
    """رسم مخططات لعرض أثر RAG في كلا الحالتين (قبل وبعد التحسين)"""
    plot_df = filter_df_by_metric(question_df, metric)
    if plot_df.empty:
        rtl_markdown("لا توجد بيانات كافية لرسم المخططات.")
        return

    y_range = y_axis_range(metric, plot_df[metric], use_adaptive=use_adaptive_range)

    rtl_markdown("""
    المخططات التالية تُظهر **أثر RAG على توليد الأسئلة** من خلال مقارنة مباشرة بين Vanilla و RAG في كلا الحالتين:
    - **قبل التحسين**: مقارنة Vanilla vs RAG في النسخة الأصلية
    - **بعد التحسين**: مقارنة Vanilla vs RAG في النسخة المحسّنة
    """)

    color_discrete_map_vr = {
        "Vanilla": "#1f77b4",
        "RAG": "#ff7f0e"
    }

    col1, col2 = st.columns(2)
    with col1:
        fig_vr_violin = px.violin(
            plot_df,
            x="method",
            y=metric,
            color="method",
            facet_col="model",
            facet_row="version",
            box=True,
            points="outliers",
            hover_data=["file", "source"],
            color_discrete_map=color_discrete_map_vr,
            category_orders={"version": ["before", "after"]},
        )
        fig_vr_violin.update_layout(
            title=f"مخطط Violin: أثر RAG (Vanilla vs RAG) قبل وبعد التحسين - {metric}",
            legend_title="الطريقة",
            template="plotly_white",
        )
        fig_vr_violin.update_yaxes(range=y_range, tickformat=".4f")
        # تحديث تسميات الصفوف
        fig_vr_violin.for_each_annotation(lambda a: a.update(text=a.text.replace("version=before", "قبل التحسين").replace("version=after", "بعد التحسين")))
        # تحسين عرض النقاط
        fig_vr_violin.update_traces(pointpos=0, jitter=0.3, points="outliers")
        embed_plotly(fig_vr_violin, height_px=940)

        # زر التصدير
        if st.button(f"💾 تصدير مخطط Violin", key=f"export_vr_violin_{metric}", use_container_width=True):
            try:
                path = export_plot(fig_vr_violin, f"violin_rag_effect_{metric}.png")
                st.success(f"✅ تم حفظ المخطط بنجاح في: `{path}`")
            except Exception as e:
                st.error(f"❌ {str(e)}")

    with col2:
        fig_vr_box = px.box(
            plot_df,
            x="method",
            y=metric,
            color="method",
            facet_col="model",
            facet_row="version",
            hover_data=["file", "source"],
            color_discrete_map=color_discrete_map_vr,
            category_orders={"version": ["before", "after"]},
        )
        fig_vr_box.update_layout(
            title=f"مخطط Box: أثر RAG (Vanilla vs RAG) قبل وبعد التحسين - {metric}",
            legend_title="الطريقة",
            template="plotly_white",
        )
        fig_vr_box.update_yaxes(range=y_range, tickformat=".4f")
        # تحديث تسميات الصفوف
        fig_vr_box.for_each_annotation(lambda a: a.update(text=a.text.replace("version=before", "قبل التحسين").replace("version=after", "بعد التحسين")))
        # تحسين عرض المخطط
        fig_vr_box.update_traces(boxmean='sd', boxpoints='outliers')
        embed_plotly(fig_vr_box, height_px=940)

        # زر التصدير
        if st.button(f"💾 تصدير مخطط Box", key=f"export_vr_box_{metric}", use_container_width=True):
            try:
                path = export_plot(fig_vr_box, f"box_rag_effect_{metric}.png")
                st.success(f"✅ تم حفظ المخطط بنجاح في: `{path}`")
            except Exception as e:
                st.error(f"❌ {str(e)}")

    st.markdown("<hr />", unsafe_allow_html=True)


def y_axis_range(metric: str, values: pd.Series, use_adaptive: bool = True):
    """
    حساب نطاق المحور Y للمقياس المحدد

    Args:
        metric: اسم المقياس
        values: قيم المقياس
        use_adaptive: إذا كان True، يستخدم نطاق ديناميكي بناءً على البيانات الفعلية
    """
    if metric in {"precision", "recall", "f1_score", "bleu", "bert_score"}:
        if use_adaptive and len(values) > 0:
            # حساب الإحصائيات
            min_val = float(values.min())
            max_val = float(values.max())
            q25 = float(values.quantile(0.25))
            q75 = float(values.quantile(0.75))
            iqr = q75 - q25

            # إذا كانت معظم القيم قريبة من 1.0 (أكثر من 90% عند 0.9+)
            high_values_ratio = (values >= 0.9).sum() / len(values)

            if high_values_ratio > 0.9:
                # استخدام نطاق مكبر للتركيز على المنطقة القريبة من 1.0
                # عرض المنطقة من 0.9 إلى 1.0 + هامش صغير
                lower = max(0, min_val - 0.05)
                upper = min(1.0, max_val + 0.02)
                # إذا كان النطاق صغير جداً، نستخدم نطاق أوسع قليلاً
                if upper - lower < 0.15:
                    lower = max(0, min_val - 0.1)
                    upper = min(1.0, max_val + 0.05)
                return [lower, upper]
            else:
                # إذا كانت البيانات متنوعة، نستخدم النطاق الكامل مع هامش
                lower = max(0, min_val - 0.05)
                upper = min(1.0, max_val + 0.05)
                return [lower, upper]
        else:
            return [0, 1]
    if metric == "perplexity":
        lower = max(0, float(values.min()) - 10)
        upper = float(values.quantile(0.95)) + 10
        return [lower, upper]
    if metric == "log_perplexity":
        lower = float(values.min()) - 0.5
        upper = float(values.max()) + 0.5
        return [lower, upper]
    return None


def render_plots(question_df: pd.DataFrame, metric: str, use_adaptive_range: bool = True):
    plot_df = filter_df_by_metric(question_df, metric)
    if plot_df.empty:
        rtl_markdown("لا توجد بيانات كافية لرسم المخططات.")
        return

    y_range = y_axis_range(metric, plot_df[metric], use_adaptive=use_adaptive_range)

    # عرض معلومات إحصائية
    with st.expander("📊 إحصائيات سريعة للمقياس"):
        stats_df = plot_df.groupby(["model", "method"])[metric].agg([
            ("العدد", "count"),
            ("المتوسط", "mean"),
            ("الوسيط", "median"),
            ("الحد الأدنى", "min"),
            ("الحد الأقصى", "max"),
            ("الربيع الأول", lambda x: x.quantile(0.25)),
            ("الربيع الثالث", lambda x: x.quantile(0.75))
        ]).round(4)
        st.dataframe(stats_df, use_container_width=True)

    color_discrete_map = {"Vanilla": "#1f77b4", "RAG": "#ff7f0e"}

    col1, col2 = st.columns(2)

    with col1:
        fig_violin = px.violin(
            plot_df,
            x="method",
            y=metric,
            color="method",
            facet_col="model",
            box=True,
            points="outliers",  # عرض النقاط الشاذة فقط لتقليل الازدحام
            hover_data=["file", "source"],
            color_discrete_map=color_discrete_map,
        )
        fig_violin.update_layout(
            title=f"مخطط Violin لمقياس {metric}",
            legend_title="الطريقة",
            template="plotly_white",
        )
        fig_violin.update_yaxes(range=y_range, tickformat=".4f")
        # تحسين عرض النقاط
        fig_violin.update_traces(pointpos=0, jitter=0.3)
        embed_plotly(fig_violin, height_px=560)

        # زر التصدير
        if st.button(f"💾 تصدير مخطط Violin", key=f"export_violin_{metric}", use_container_width=True):
            try:
                path = export_plot(fig_violin, f"violin_{metric}.png")
                st.success(f"✅ تم حفظ المخطط بنجاح في: `{path}`")
            except Exception as e:
                st.error(f"❌ {str(e)}")

    with col2:
        fig_box = px.box(
            plot_df,
            x="method",
            y=metric,
            color="method",
            facet_col="model",
            hover_data=["file", "source"],
            color_discrete_map=color_discrete_map,
        )
        fig_box.update_layout(
            title=f"مخطط Box لمقياس {metric}",
            legend_title="الطريقة",
            template="plotly_white",
        )
        fig_box.update_yaxes(range=y_range, tickformat=".4f")
        # تحسين عرض النقاط
        fig_box.update_traces(boxmean='sd', boxpoints='outliers')  # إظهار المتوسط والانحراف المعياري وعرض النقاط الشاذة فقط
        embed_plotly(fig_box, height_px=560)

        # زر التصدير
        if st.button(f"💾 تصدير مخطط Box", key=f"export_box_{metric}", use_container_width=True):
            try:
                path = export_plot(fig_box, f"box_{metric}.png")
                st.success(f"✅ تم حفظ المخطط بنجاح في: `{path}`")
            except Exception as e:
                st.error(f"❌ {str(e)}")

    st.markdown("<hr />", unsafe_allow_html=True)


def export_plot(fig, filename: str):
    """تصدير مخطط إلى ملف PNG في مجلد plots"""
    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = PLOTS_DIR / filename

        # محاولة التصدير
        fig.write_image(str(file_path), width=1200, height=800, scale=2)

        # التحقق من وجود الملف
        if file_path.exists():
            return file_path
        else:
            raise FileNotFoundError(f"الملف لم يُنشأ: {file_path}")
    except Exception as e:
        # إرجاع الخطأ بدلاً من الملف
        raise Exception(f"فشل تصدير المخطط: {str(e)}. تأكد من تثبيت kaleido: pip install kaleido")


def export_buttons(question_df: pd.DataFrame, metric: str):
    plot_df = filter_df_by_metric(question_df, metric)
    if plot_df.empty:
        return

    y_range = y_axis_range(metric, plot_df[metric])
    color_discrete_map = {"Vanilla": "#1f77b4", "RAG": "#ff7f0e"}

    fig_violin = px.violin(
        plot_df,
        x="method",
        y=metric,
        color="method",
        facet_col="model",
        box=True,
        points="all",
        hover_data=["file", "source"],
        color_discrete_map=color_discrete_map,
    )
    fig_violin.update_layout(template="plotly_white")
    fig_violin.update_yaxes(range=y_range, tickformat=".4f")

    fig_box = px.box(
        plot_df,
        x="method",
        y=metric,
        color="method",
        facet_col="model",
        hover_data=["file", "source"],
        color_discrete_map=color_discrete_map,
    )
    fig_box.update_layout(template="plotly_white")
    fig_box.update_yaxes(range=y_range, tickformat=".4f")
    fig_box.update_traces(boxpoints='outliers')

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 تصدير مخطط Violin", use_container_width=True):
            try:
                path = export_plot(fig_violin, f"violin_{metric}.png")
                st.success(f"✅ تم حفظ المخطط بنجاح في: `{path}`")
            except Exception as e:
                st.error(f"❌ {str(e)}")
    with col2:
        if st.button("💾 تصدير مخطط Box", use_container_width=True):
            try:
                path = export_plot(fig_box, f"box_{metric}.png")
                st.success(f"✅ تم حفظ المخطط بنجاح في: `{path}`")
            except Exception as e:
                st.error(f"❌ {str(e)}")


def main():
    rtl_markdown("# 📊 المقارنة والتحليل")

    files_hash = get_files_hash()
    if st.session_state.get("_outputs_files_hash") != files_hash:
        load_question_frames.clear()
        st.session_state["_outputs_files_hash"] = files_hash

    _, question_df, _ = load_question_frames(files_hash)

    if question_df.empty:
        rtl_markdown("لا توجد ملفات أسئلة في المجلد `outputs/`.")
        return

    metric_options = SUMMARY_COLUMNS
    selected_metric = st.selectbox(
        "اختر المقياس للتحليل التفصيلي",
        options=metric_options,
        index=0,
    )

    rtl_markdown("---")
    rtl_markdown("## 📊 التحليل الإحصائي")

    rtl_markdown("### اختبار شابيرو-ويلك")
    shapiro_section(question_df, selected_metric)

    rtl_markdown("---")
    rtl_markdown("## قبل vs بعد التحسين")
    mann_whitney_before_after_section(question_df, selected_metric)

    rtl_markdown("---")
    rtl_markdown("## أثر RAG (Vanilla vs RAG)")

    rtl_markdown("### قبل التحسين")
    mann_whitney_by_version_filtered(question_df, selected_metric, "before")

    rtl_markdown("### بعد التحسين")
    mann_whitney_by_version_filtered(question_df, selected_metric, "after")

    rtl_markdown("---")
    rtl_markdown("## المخططات التفاعلية")

    # زر لحذف جميع المخططات
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ حذف جميع المخططات المصدرة", type="secondary", use_container_width=True):
            try:
                if PLOTS_DIR.exists():
                    files = list(PLOTS_DIR.glob("*"))
                    deleted_count = 0
                    for file_path in files:
                        if file_path.is_file():
                            file_path.unlink()
                            deleted_count += 1
                    if deleted_count > 0:
                        st.success(f"✅ تم حذف {deleted_count} ملف من مجلد plots")
                    else:
                        st.info("لا توجد ملفات في مجلد plots")
                else:
                    st.info("مجلد plots غير موجود")
            except Exception as e:
                st.error(f"❌ فشل حذف الملفات: {str(e)}")

    rtl_markdown("")
    use_adaptive_range = st.checkbox(
        "🔍 تكبير المناطق المهمة في المخططات (للمقاييس المتركزة)",
        value=True,
        help="عند التفعيل، سيتم تكبير المناطق التي تحتوي على معظم البيانات لرؤية أفضل. مفيد للمقاييس المتركزة عند 1.0 مثل precision"
    )

    render_comparison_plots(question_df, selected_metric, use_adaptive_range=use_adaptive_range)


if __name__ == "__main__":
    main()

