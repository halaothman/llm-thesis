"""نقطة دخول Streamlit: الصفحة الرئيسية، شريط التنقل، وفحص حالة Ollama والفهارس."""
import streamlit as st
import logging

# إعدادات التصحيح - تم نقلها إلى config.toml
# st.set_option("logger.level", "error")  # لا يمكن تعيينها هنا
logging.getLogger().setLevel(logging.ERROR)
st.set_page_config(
    page_title="توليد الأسئلة التعليمية",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "توليد أسئلة عربية/إنجليزية — LLaMA / Qwen، Vanilla & RAG"
    }
)

# إضافة CSS شامل لجميع الصفحات
with open("style.css", "r", encoding="utf-8") as f:
    css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# إضافة CSS إضافي لمحاذاة النصوص
st.markdown("""
<style>
    /* محاذاة النصوص الرئيسية */
    .main .block-container {
        direction: rtl;
        text-align: right;
    }
    
    .stMarkdown {
        direction: rtl;
        text-align: right;
    }
    
    .stSelectbox label,
    .stTextInput label,
    .stTextArea label,
    .stFileUploader label {
        direction: rtl;
        text-align: right;
    }
    
    .stRadio > div {
        direction: rtl;
        text-align: right;
    }
    
    .stButton > button {
        direction: rtl;
    }
    
    .stMetric {
        direction: rtl;
        text-align: right;
    }
    
    .stDataFrame {
        direction: rtl;
    }
    
    .stExpander {
        direction: rtl;
    }
    
    .stAlert {
        direction: rtl;
        text-align: right;
    }
    
    .stSuccess,
    .stError,
    .stWarning,
    .stInfo {
        direction: rtl;
        text-align: right;
    }
    
    .stProgress {
        direction: rtl;
    }
    
    .stSpinner {
        direction: rtl;
    }
    
    /* محاذاة الجداول */
    .stDataFrame table {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة الأعمدة */
    .stColumns {
        direction: rtl;
    }
    
    /* محاذاة النصوص في العناصر */
    div[data-testid="stMarkdownContainer"] {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة العناوين */
    h1, h2, h3, h4, h5, h6 {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة النصوص في الـ sidebar */
    .css-1d391kg {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة خيارات الأسئلة لليمين */
    .stRadio > div {
        direction: rtl;
        text-align: right;
    }
    
    .stRadio > div > label {
        direction: rtl;
        text-align: right;
        justify-content: flex-start;
    }
    
    .stRadio > div > label > div {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة النصوص داخل الخيارات */
    .stRadio > div > label > div > div {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة الأزرار الدائرية */
    .stRadio > div > label > div > div > div {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة النصوص في الخيارات */
    .stRadio > div > label > div > div > div > div {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة إضافية لخيارات الأسئلة */
    div[data-testid="stRadio"] {
        direction: rtl;
        text-align: right;
    }
    
    div[data-testid="stRadio"] > div {
        direction: rtl;
        text-align: right;
    }
    
    div[data-testid="stRadio"] > div > label {
        direction: rtl;
        text-align: right;
        justify-content: flex-start;
    }
    
    /* محاذاة النصوص في الخيارات */
    div[data-testid="stRadio"] > div > label > div {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة الأزرار الدائرية */
    div[data-testid="stRadio"] > div > label > div > div {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة النصوص في الخيارات */
    div[data-testid="stRadio"] > div > label > div > div > div {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة النصوص في الخيارات */
    div[data-testid="stRadio"] > div > label > div > div > div > div {
        direction: rtl;
        text-align: right;
    }
    
    /* تحسين عرض خيارات الأسئلة - كل خيار على سطر منفصل */
    .stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    
    .stRadio > div > label {
        display: block;
        margin-bottom: 8px;
        padding: 8px;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        background-color: #f9f9f9;
        transition: background-color 0.2s;
    }
    
    .stRadio > div > label:hover {
        background-color: #f0f0f0;
    }
    
    .stRadio > div > label > div {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* محاذاة النصوص في الخيارات */
    .stRadio > div > label > div > div {
        direction: rtl;
        text-align: right;
        flex: 1;
    }
    
    /* تحسين الأزرار الدائرية */
    .stRadio > div > label > div > div > div {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة النصوص في الخيارات */
    .stRadio > div > label > div > div > div > div {
        direction: rtl;
        text-align: right;
    }
    
    /* تحسين إضافي لخيارات الأسئلة */
    div[data-testid="stRadio"] > div {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    
    div[data-testid="stRadio"] > div > label {
        display: block;
        margin-bottom: 8px;
        padding: 8px;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        background-color: #f9f9f9;
        transition: background-color 0.2s;
    }
    
    div[data-testid="stRadio"] > div > label:hover {
        background-color: #f0f0f0;
    }
    
    div[data-testid="stRadio"] > div > label > div {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* محاذاة النصوص في الخيارات */
    div[data-testid="stRadio"] > div > label > div > div {
        direction: rtl;
        text-align: right;
        flex: 1;
    }
    
    /* محاذاة الأزرار الدائرية */
    div[data-testid="stRadio"] > div > label > div > div > div {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة النصوص في الخيارات */
    div[data-testid="stRadio"] > div > label > div > div > div > div {
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 توليد الأسئلة التعليمية")
st.markdown("""
**Vanilla** (النص المرفوع) أو **RAG** (مصادر مفهرسة) — LLaMA 3.2:3B و Qwen 2.5:7B عبر Ollama.

- **📚 فهرسة المراجع** — فهرسة مصادر RAG
- **❓ توليد الأسئلة** — رفع ملف وتوليد الأسئلة
- **📊 المقارنة والتحليل** — مقاييس آلية وتحليل إحصائي
- **👤 التقييم البشري** — نتائج التقييم البشري
""")

# حالة النظام
with st.expander("🔧 حالة النظام"):
    try:
        import ollama
        models = ollama.list()
        if models and "models" in models:
            st.success("✅ Ollama يعمل")
            available_models = [model["name"] for model in models["models"]]
            st.caption(f"النماذج: {', '.join(available_models)}")
        else:
            st.error("❌ Ollama غير متاح")
    except Exception as e:
        st.error(f"❌ خطأ في الاتصال بـ Ollama: {e}")
    
    # فحص الملفات
    import os
    from src.config import get_index_paths, get_rag_version
    
    # فحص الفهارس الجديدة (Baseline و Improved)
    baseline_idx, baseline_meta = "indexes/baseline.external.index", "indexes/baseline.external_meta.jsonl"
    improved_idx, improved_meta = "indexes/improved.external.index", "indexes/improved.external_meta.jsonl"
    
    baseline_exists = os.path.exists(baseline_idx) and os.path.exists(baseline_meta)
    improved_exists = os.path.exists(improved_idx) and os.path.exists(improved_meta)
    
    st.write("**الفهارس:**")
    if baseline_exists:
        st.success(f"✅ Baseline ({os.path.getsize(baseline_idx) / 1024 / 1024:.1f} MB)")
    else:
        st.warning("⚠️ Baseline: غير موجود")
    
    if improved_exists:
        st.success(f"✅ Improved ({os.path.getsize(improved_idx) / 1024 / 1024:.1f} MB)")
    else:
        st.warning("⚠️ Improved: غير موجود")
