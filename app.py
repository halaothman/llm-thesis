"""نقطة دخول Streamlit: الصفحة الرئيسية وشريط التنقل."""
import logging

import streamlit as st
import streamlit_path  # noqa: F401 — جذر المشروع على sys.path

from src.ui_styles import inject_app_styles

logging.getLogger().setLevel(logging.ERROR)
st.set_page_config(
    page_title="توليد الأسئلة التعليمية",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_app_styles()

tab_edu, tab_thesis = st.tabs(["توليد الأسئلة التعليمية", "مشروع الرسالة (RAG + Ollama)"])

with tab_edu:
    from edu_question_generator.ui import render_edu_app

    render_edu_app()

with tab_thesis:
    st.title("مشروع الرسالة")
    st.markdown("**Vanilla** (النص المرفوع) أو **RAG** (مصادر مفهرسة)")
    st.markdown("LLaMA 3.2:3B و Qwen 2.5:7B عبر Ollama.")
    st.markdown("""
- **فهرسة قاعدة المعرفة** — فهرسة مصادر RAG
- **توليد الأسئلة** — رفع ملف وتوليد الأسئلة
- **المقارنة والتحليل** — مقاييس آلية وتحليل إحصائي
- **التقييم البشري** — نتائج التقييم البشري
""")
