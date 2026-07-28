"""صفحة توليد الأسئلة: Vanilla و RAG، مع حفظ JSON وعرض الاختبار."""
import streamlit as st
import os
import time
import sys
import numpy as np
from pathlib import Path

# إضافة المسار الجذر للمشروع إلى Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.loaders import load_text
    from src.chunking import chunk_text
    from src.arabic_text import clean_ar
    from src.faiss_store import build_or_update, search, load_meta
    from src.rag import retrieve
    from src.generator import detect_lang, build_prompt_vanilla, build_prompt_rag, call_llama, safe_json, generate_questions_with_retry
    from src.storage import save_group
    from src.embeddings import embed_texts, get_model_name, get_model_info
    from src.config import get_rag_version, set_rag_version, get_index_paths
    # إضافة الوظائف الجديدة لحل الأسئلة وتصحيح الإجابات
    from src.quiz_ui import render_quiz, grade, display_quiz_results, validate_questions_format, add_missing_correct_answers
except ImportError as e:
    st.error(f"خطأ في استيراد الوحدات: {e}")
    st.stop()

st.set_page_config(page_title="توليد الأسئلة", layout="wide")

# إضافة CSS شامل لجميع الصفحات
try:
    with open("style.css", "r", encoding="utf-8") as f:
        css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    # CSS احتياطي إذا لم يوجد الملف
    st.markdown("""
    <style>
        .stApp { direction: rtl; }
        .stApp > div:first-child { direction: rtl; flex-direction: row-reverse; }
        .stSidebar { direction: rtl; text-align: right; order: 2; }
        .main .block-container { direction: rtl; text-align: right; order: 1; }
        .stSidebar * { direction: rtl; text-align: right; }
    </style>
    """, unsafe_allow_html=True)

# إضافة CSS إضافي لمحاذاة أفضل
st.markdown("""
<style>
    /* محاذاة خاصة بصفحة توليد الأسئلة */
    .stFileUploader {
        direction: rtl;
    }
    
    .stFileUploader > div {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة أزرار الراديو */
    .stRadio > div > label {
        direction: rtl;
        text-align: right;
    }
    
    .stRadio > div > div {
        direction: rtl;
    }
    
    /* محاذاة أزرار الاختيار */
    .stCheckbox > div > label {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة النصوص في الـ quiz */
    .stMarkdown h3 {
        direction: rtl;
        text-align: right;
    }
    
    .stMarkdown h4 {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة الـ spinner */
    .stSpinner > div {
        direction: rtl;
    }
    
    /* محاذاة الـ expander للمصادر */
    .streamlit-expanderHeader {
        direction: rtl;
        text-align: right;
    }
    
    .streamlit-expanderContent {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة النصوص في المصادر */
    .stMarkdown blockquote {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة الـ code blocks */
    .stCode {
        direction: ltr;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

st.title("توليد الأسئلة (Vanilla & RAG)")

# اختيار النموذج في بداية الصفحة
st.subheader("اختر النموذج")
col1, col2 = st.columns(2)

with col1:
    if st.button("LLaMA 3.2:3B", use_container_width=True, type="primary"):
        st.session_state.selected_model = "llama3.2:3b"

with col2:
    if st.button("Qwen 2.5:7B", use_container_width=True, type="primary"):
        st.session_state.selected_model = "qwen2.5:7b"

if 'selected_model' in st.session_state:
    st.caption(f"النموذج: **{st.session_state.selected_model}**")
else:
    st.warning("اختر نموذجاً أولاً")

st.markdown("---")

# إعدادات المسارات
UPLOADS = "uploads"
os.makedirs(UPLOADS, exist_ok=True)

# استيراد دالة الحصول على مسارات الفهرس
from src.config import get_index_paths

# الحصول على مسارات الفهرس بناءً على النسخة المحددة
IDX_PATH, META_PATH = get_index_paths()

# إعدادات النظام (سيتم تحديثها بناءً على النسخة المختارة)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TEMPERATURE = 0.7
MAX_RETRIES = 3

# الحصول على النسخة الحالية
def get_current_settings():
    """الحصول على الإعدادات الحالية بناءً على النسخة المختارة"""
    version = get_rag_version()
    if version == "baseline":
        return {
            "TOP_K": 5,
            "SIMILARITY_THRESHOLD": 0.82,
            "EMBEDDING_MODEL": "intfloat/e5-large-v2"
        }
    else:  # improved
        return {
            "TOP_K": 10,
            "SIMILARITY_THRESHOLD": 0.65,
            "EMBEDDING_MODEL": "aubmindlab/bert-base-arabertv2"
        }

# دالة للحصول على الإعدادات الحالية (يتم استدعاؤها عند الحاجة)
def get_settings():
    """الحصول على الإعدادات الحالية بناءً على النسخة المختارة"""
    return get_current_settings()

# تهيئة الإعدادات الأولية
settings = get_settings()
TOP_K = settings["TOP_K"]
SIMILARITY_THRESHOLD = settings["SIMILARITY_THRESHOLD"]
EMBEDDING_MODEL = settings["EMBEDDING_MODEL"]

# الشريط الجانبي - إعدادات النظام
with st.sidebar:
    st.header("إعدادات النظام")
    
    # حالة Ollama
    st.subheader("حالة Ollama")
    try:
        import ollama
        models = ollama.list()
        
        # التحقق من صحة الاستجابة
        if models and hasattr(models, 'models'):
            installed_models = []
            if hasattr(models.models, '__iter__'):
                for model in models.models:
                    if hasattr(model, 'model'):
                        installed_models.append(model.model)
                    else:
                        st.warning(f"نموذج غير صالح: {model}")
        elif models and isinstance(models, list):
            # في حالة كانت الاستجابة قائمة مباشرة
            installed_models = []
            for model in models:
                if hasattr(model, 'model'):
                    installed_models.append(model.model)
                elif isinstance(model, dict) and "model" in model:
                    installed_models.append(model["model"])
                else:
                    st.warning(f"نموذج غير صالح: {model}")
            
            if installed_models:
                st.success("Ollama يعمل")
                st.write(f"النماذج المثبتة: {len(installed_models)}")
                
                # عرض النماذج المثبتة
                for model in installed_models:
                    if "llama3.2:3b" in model:
                        st.write(f"• {model} ")
                    elif "qwen" in model.lower():
                        st.write(f"• {model} ")
                    else:
                        st.write(f"• {model}")
                
                # عرض النماذج المدعومة غير المثبتة
                supported_models = ["llama3.2:3b", "qwen2.5:7b"]
                missing_models = [model for model in supported_models if not any(model in installed for installed in installed_models)]
                if missing_models:
                    st.write("النماذج المدعومة غير المثبتة:")
                    for model in missing_models:
                        st.write(f"• {model} ")
                    
                    # زر تثبيت النماذج
                    if st.button("تثبيت النماذج المفقودة", key="install_models"):
                        st.info("لتثبيت النماذج، استخدم الأوامر التالية في Terminal:")
                        for model in missing_models:
                            st.code(f"ollama pull {model}", language="bash")
            else:
                st.warning("لا توجد نماذج مثبتة")
                st.info("لتثبيت نماذج، استخدم: `ollama pull llama3.2:3b`")
        else:
            st.error("استجابة غير صحيحة من Ollama")
            st.write(f"الاستجابة: {models}")
    except Exception as e:
        st.error(f"خطأ في الاتصال بـ Ollama: {e}")
        st.write("تأكد من أن Ollama يعمل: `ollama serve`")
    
    st.markdown("---")
    
    # اختيار النسخة (Baseline vs Improved)
    st.subheader("نسخة RAG")
    current_version = get_rag_version()
    version_options = {
        "baseline": " Baseline (الأصلية)",
        "improved": " Improved (المحسّنة)"
    }
    selected_version = st.radio(
        "اختر النسخة:",
        options=["baseline", "improved"],
        index=0 if current_version == "baseline" else 1,
        format_func=lambda x: version_options[x],
        key="rag_version_selector"
    )
    
    if selected_version != current_version:
        set_rag_version(selected_version)
        # تحديث الإعدادات والمسارات
        new_settings = get_current_settings()
        IDX_PATH, META_PATH = get_index_paths()  # تحديث مسارات الفهرس
        st.session_state['top_k'] = new_settings["TOP_K"]
        st.session_state['similarity_threshold'] = new_settings["SIMILARITY_THRESHOLD"]
        st.rerun()
    
    # تحديث المسارات بناءً على النسخة الحالية
    IDX_PATH, META_PATH = get_index_paths()
    
    # تحديث الإعدادات من session state إذا كانت موجودة
    if 'top_k' in st.session_state:
        TOP_K = st.session_state['top_k']
    if 'similarity_threshold' in st.session_state:
        SIMILARITY_THRESHOLD = st.session_state['similarity_threshold']
    
    # تحديث الإعدادات بناءً على النسخة الحالية
    current_settings = get_current_settings()
    TOP_K = current_settings["TOP_K"]
    SIMILARITY_THRESHOLD = current_settings["SIMILARITY_THRESHOLD"]
    EMBEDDING_MODEL = current_settings["EMBEDDING_MODEL"]
    
    # عرض معلومات النسخة
    version_info = {
        "baseline": {
            "name": "Baseline (الأصلية)",
            "model": "intfloat/e5-large-v2",
            "top_k": 5,
            "threshold": 0.82,
            "description": "النسخة الأصلية قبل التحسين"
        },
        "improved": {
            "name": "Improved (المحسّنة)",
            "model": "aubmindlab/bert-base-arabertv2",
            "top_k": 10,
            "threshold": 0.65,
            "description": "النسخة المحسّنة بعد التحسين"
        }
    }
    
    info = version_info[current_version]
    st.caption(f"{info['name']} — Top-K {info['top_k']}, عتبة {info['threshold']}")
    
    st.markdown("---")
    
    # إعدادات التضمين (تحديث تلقائي بناءً على النسخة)
    st.subheader("نظام التضمين")
    model_name = get_model_name()
    model_info = get_model_info()
    st.metric("النموذج", model_name)
    st.metric("النسخة", info['name'])
    st.metric("حجم الشنك", f"{CHUNK_SIZE} حرف")
    st.metric("التداخل", f"{CHUNK_OVERLAP} حرف")
    st.metric("Top-K", info['top_k'])
    st.metric("عتبة التشابه", f"{info['threshold']}")
    st.metric("مسار الفهرس", IDX_PATH)
    st.metric("مسار الميتاداتا", META_PATH)
    
    st.markdown("---")
    
    # إعدادات التوليد
    st.subheader("إعدادات التوليد")
    st.metric("درجة الحرارة", TEMPERATURE)
    st.metric("عدد المحاولات", MAX_RETRIES)
    st.metric("عدد الأسئلة", "10 (5 MCQ + 5 TF)")
    
    # النموذج المختار
    if 'selected_model' in st.session_state:
        model_name = st.session_state.selected_model
        st.metric("النموذج المختار", model_name)
        if "llama3.2:3b" in model_name:
            st.success("LLaMA 3.2:3B")
        elif "qwen" in model_name.lower():
            st.success("Qwen 2.5:7B")
        else:
            st.info(f"{model_name}")
    else:
        st.info("لم يُختر نموذج بعد")

# رفع الملف
st.subheader("رفع الملف")

# التحقق من اختيار النموذج (مطلوب لـ Vanilla و RAG)
if 'selected_model' not in st.session_state:
    st.warning("اختر نموذجاً من الأعلى (Vanilla و RAG).")

up = st.file_uploader("ارفع PDF / DOCX / TXT / MD", type=["pdf","docx","doc","txt","md"])

if not up:
    st.info("ارفع ملفاً لإظهار أزرار التوليد: Vanilla / RAG")
    st.stop()

# حفظ الملف مؤقتاً
path = os.path.join(UPLOADS, up.name)
with open(path, "wb") as f:
    f.write(up.read())

# تحليل الملف
try:
    raw_text = load_text(path)
    if not raw_text.strip():
        st.error("الملف فارغ أو لا يحتوي على نص")
        st.stop()
    
    lang = detect_lang(raw_text)
    text_length = len(raw_text)
    
    # استخدام النموذج المختار (إن وُجد — للـ Vanilla/RAG)
    model_name = st.session_state.get("selected_model")
    
    # التحقق من حالة النموذج المختار
    try:
        import ollama
        models = ollama.list()
        installed_models = []
        
        if models and hasattr(models, 'models'):
            if hasattr(models.models, '__iter__'):
                for model in models.models:
                    if hasattr(model, 'model'):
                        installed_models.append(model.model)
        
        if model_name in installed_models:
            st.success(f"النموذج {model_name} مثبت ومتاح")
        elif model_name:
            st.warning(f"النموذج {model_name} غير مثبت. سيتم محاولة تحميله عند الاستخدام.")
            st.info(f"يمكنك تثبيت النموذج باستخدام: `ollama pull {model_name}`")
        else:
            st.info("لم يُختر نموذج — اختر LLaMA أو Qwen من الأعلى لتوليد Vanilla/RAG")
    except Exception as e:
        st.error(f"خطأ في التحقق من حالة النماذج: {e}")
    
    # عرض معلومات الملف
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("اسم الملف", up.name)
    with col2:
        st.metric("اللغة المكتشفة", "العربية" if lang == "ar" else "English")
    with col3:
        st.metric("طول النص", f"{text_length:,} حرف")
    with col4:
        st.metric("النموذج المختار", model_name or "—")

except Exception as e:
    st.error(f"خطأ في قراءة الملف: {e}")
    st.stop()

# أزرار التوليد
st.subheader("اختر طريقة التوليد")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### توليد Vanilla (بدون RAG)")
    st.caption("توليد الأسئلة من النص المرفوع فقط")
    
    if st.button("توليد أسئلة Vanilla", type="primary", use_container_width=True):
        if not model_name:
            st.error("اختر نموذج LLaMA أو Qwen من الأعلى أولاً")
        else:
            try:
                import ollama
                models = ollama.list()
                installed_models = []
                if models and hasattr(models, 'models') and hasattr(models.models, '__iter__'):
                    for model in models.models:
                        if hasattr(model, 'model'):
                            installed_models.append(model.model)
                if model_name not in installed_models:
                    st.info(f"النموذج {model_name} غير موجود. سيتم تحميله تلقائياً...")
            except Exception as e:
                st.warning(f"تعذر التحقق من حالة النموذج: {e}")

            with st.spinner("جاري توليد الأسئلة..."):
                try:
                    prompt = build_prompt_vanilla(raw_text, lang)
                    start_time = time.time()
                    response = call_llama(prompt, model_name=model_name, temperature=TEMPERATURE)
                    generation_time = time.time() - start_time
                    out = safe_json(response, raw_text, model_name, lang, None)
                    if out is None:
                        st.error("فشل في تحليل JSON، سيتم إعادة المحاولة")
                        out = generate_questions_with_retry(
                            prompt, max_retries=2, source_text=raw_text,
                            model_name=model_name, lang=lang, retrieved=None,
                        )
                        if out is None:
                            st.error("فشل في توليد الأسئلة بعد جميع المحاولات")
                            st.stop()
                        st.success("تم توليد الأسئلة بنجاح بعد إعادة المحاولة")
                    if out is None or not isinstance(out, dict) or "mcq" not in out or "tf" not in out:
                        st.error("خطأ في تنسيق الأسئلة المولدة")
                        st.code(response)
                        st.stop()
                    out["source_text"] = raw_text
                    out["generation_time"] = generation_time
                    with st.spinner("جاري حساب المقاييس وحفظ الملف..."):
                        try:
                            from src.storage import save_questions_separate_file
                            filename, num_questions = save_questions_separate_file(
                                out, model_name, "vanilla", up.name, lang,
                            )
                            st.success(f"تم حفظ {num_questions} سؤال في: `{filename}`")
                        except Exception as e:
                            st.error(f"خطأ في حفظ الملف المنفصل: {e}")
                            save_group("A", {
                                "lang": lang, **out, "source_file": up.name,
                                "generation_time": generation_time, "method": "vanilla",
                            })
                    st.success(f"تم توليد {len(out.get('mcq', []))} MCQ و {len(out.get('tf', []))} صح/خطأ")
                    st.info(f"وقت التوليد: {generation_time:.2f} ثانية")
                    st.session_state.vanilla_questions = out
                    st.session_state.vanilla_generated = True
                except Exception as e:
                    st.error(f"خطأ في توليد الأسئلة: {e}")

with col2:
    st.markdown("#### توليد RAG (مع الاسترجاع)")
    st.caption("توليد الأسئلة مع الاسترجاع من المصادر الخارجية")
    
    # التحقق من وجود الفهرس
    if not os.path.exists(IDX_PATH):
        st.warning("لم يتم إنشاء فهرس المصادر الخارجية بعد")
        st.info("يرجى الذهاب إلى صفحة 'فهرسة المراجع' أولاً")
    else:
        if st.button("توليد أسئلة RAG", type="primary", use_container_width=True):
            if not model_name:
                st.error("اختر نموذج LLaMA أو Qwen من الأعلى أولاً")
            else:
                try:
                    import ollama
                    models = ollama.list()
                    installed_models = []
                    
                    if models and hasattr(models, 'models'):
                        if hasattr(models.models, '__iter__'):
                            for model in models.models:
                                if hasattr(model, 'model'):
                                    installed_models.append(model.model)
                    
                    if model_name not in installed_models:
                        st.info(f"النموذج {model_name} غير موجود. سيتم تحميله تلقائياً...")
                except Exception as e:
                    st.warning(f"تعذر التحقق من حالة النموذج: {e}")
                
                with st.spinner("جاري توليد الأسئلة مع RAG..."):
                    try:
                        # فهرسة الملف المرفوع مؤقتاً
                        chunks = chunk_text(clean_ar(raw_text), CHUNK_SIZE, CHUNK_OVERLAP)
                        tmp_records = [
                            {"id": i + 10_000_000, "text": ch, "metadata": {"source": up.name}} 
                            for i, ch in enumerate(chunks)
                        ]
                        build_or_update("indexes/upload.index", "indexes/upload_meta.jsonl", tmp_records)

                        # استرجاع المقاطع المشابهة (is_query=True للبحث)
                        st.info("**جاري البحث عن مصادر مشابهة...**")
                        qv = embed_texts([clean_ar(raw_text)], is_query=True)
                        D, I = search(IDX_PATH, qv, TOP_K)
                        meta = load_meta(META_PATH)
                    
                        # عرض معلومات التشخيص
                        st.info(f"**معلومات التشخيص:**")
                        st.write(f"- عدد النتائج المسترجعة من الفهرس: {len(D) if len(D) > 0 else 0}")
                        st.write(f"- درجات التشابه: {D.tolist() if len(D) > 0 else 'لا توجد نتائج'}")
                        st.write(f"- معرفات النتائج: {I.tolist() if len(I) > 0 else 'لا توجد نتائج'}")
                        st.write(f"- عتبة التشابه المطلوبة: **{SIMILARITY_THRESHOLD}**")
                    
                        retrieved = []
                    
                        # التحقق من أن النتائج ليست فارغة
                        if len(D) > 0 and len(I) > 0:
                            for sc, i in zip(D, I):
                                if i == -1:
                                    continue
                                try:
                                    sc = float(sc)
                                    st.write(f"- مقطع {i}: درجة التشابه = {sc:.3f}")
                                    if sc >= SIMILARITY_THRESHOLD:  # عتبة التشابه
                                        m = meta.get(int(i), {})
                                        retrieved.append({
                                            "text": m.get("text", ""),
                                            "filename": os.path.basename(m.get("metadata", {}).get("source", "")),
                                            "score": sc,
                                            "source_path": m.get("metadata", {}).get("source", "")
                                        })
                                        st.success(f"تم قبول المقطع {i} (درجة: {sc:.3f})")
                                    else:
                                        st.warning(f"تم رفض المقطع {i} (درجة: {sc:.3f} < {SIMILARITY_THRESHOLD})")
                                except (ValueError, TypeError) as e:
                                    st.error(f"خطأ في معالجة المقطع {i}: {e}")
                                    continue

                        st.info(f"**النتيجة النهائية:** {len(retrieved)} مقطع مقبول من أصل {len(D)}")
                    
                        # التحقق من وجود مقاطع مسترجعة قبل المتابعة
                        if not retrieved:
                            st.error("**تعذر العثور على مصادر مشابهة**")
                            st.warning(f"""
     **لم يتم العثور على مقاطع مشابهة بدرجة كافية**

    **التفاصيل:**
    - عدد النتائج المسترجعة من الفهرس: {len(D) if len(D) > 0 else 0}
    - عتبة التشابه المطلوبة: **{SIMILARITY_THRESHOLD}**
    - جميع المقاطع المسترجعة كانت بدرجة أقل من العتبة المطلوبة

    **الحلول المقترحة:**
    1. **خفض عتبة التشابه**: يمكنك تغيير العتبة في إعدادات النسخة (حالياً: {SIMILARITY_THRESHOLD})
    2. **إضافة المزيد من المصادر**: قم بفهرسة المزيد من الملفات في صفحة "فهرسة المراجع"
    3. **استخدام Vanilla**: يمكنك استخدام طريقة Vanilla التي لا تحتاج إلى مصادر خارجية
    4. **التحقق من الفهرس**: تأكد من أن الفهرس يحتوي على محتوى ذي صلة بالموضوع

    **ملاحظة:** لن يتم المتابعة بتوليد الأسئلة لأن RAG يحتاج إلى مصادر مشابهة للعمل بشكل صحيح.
                            """)
                            st.stop()  # إيقاف العملية تماماً
                    
                        st.success(f"تم العثور على {len(retrieved)} مقطع مشابه - جاري المتابعة...")

                        # بناء الـ prompt مع RAG
                        prompt = build_prompt_rag(raw_text, lang, retrieved)
                    
                        # استدعاء النموذج
                        start_time = time.time()
                        response = call_llama(prompt, model_name=model_name, temperature=TEMPERATURE)
                        generation_time = time.time() - start_time
                    
                        # عرض الرد الخام للتشخيص
                        st.info("**الرد الخام من النموذج:**")
                        with st.expander("عرض الرد الكامل"):
                            st.code(response, language="text")
                    
                        # تحليل الاستجابة مع تمرير raw_text للإصلاح التلقائي
                        st.info("**تحليل الاستجابة:**")
                        out = safe_json(response, raw_text, model_name, lang, retrieved)
                    
                        if out is None:
                            st.error("فشل في تحليل JSON، سيتم إعادة المحاولة")
                            # إعادة المحاولة مع النموذج
                            out = generate_questions_with_retry(prompt, max_retries=2, source_text=raw_text, 
                                                                 model_name=model_name, lang=lang, retrieved=retrieved)
                        
                            if out is None:
                                st.error("فشل في توليد الأسئلة بعد جميع المحاولات")
                                st.stop()
                            else:
                                st.success("تم توليد الأسئلة بنجاح بعد إعادة المحاولة")
                        else:
                            st.success("تم تحليل JSON بنجاح")
                    
                        # التحقق من صحة البيانات
                        if out is None or not isinstance(out, dict) or "mcq" not in out or "tf" not in out:
                            st.error("خطأ في تنسيق الأسئلة المولدة")
                            st.code(response)
                            st.stop()
                    
                        # إضافة مصادر الاسترجاع
                        out["sources"] = retrieved
                        out["source_text"] = raw_text
                        out["generation_time"] = generation_time
                    
                        # حفظ في ملف منفصل مع حساب المقاييس
                        with st.spinner("جاري حساب المقاييس وحفظ الملف..."):
                            try:
                                # التحقق من أن model_name صحيح
                                st.info(f"**معلومات الحفظ:**")
                                st.write(f"- النموذج: {model_name}")
                                st.write(f"- الطريقة: rag")
                                st.write(f"- الملف المصدر: {up.name}")
                            
                                from src.storage import save_questions_separate_file
                                filename, num_questions = save_questions_separate_file(
                                    out, 
                                    model_name, 
                                    "rag", 
                                    up.name, 
                                    lang
                                )
                                st.success(f"تم حفظ {num_questions} سؤال في: `{filename}`")
                                st.info(f"المسار الكامل: `outputs/{Path(up.name).stem.replace(' ', '_').replace('.', '_')}/{filename}`")
                            except Exception as e:
                                st.error(f"خطأ في حفظ الملف المنفصل: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                                st.info("جاري الحفظ في الملف الشامل...")
                                # الحفظ العادي كبديل
                                save_group("B", {
                                    "lang": lang, 
                                    **out, 
                                    "source_file": up.name,
                                    "generation_time": generation_time,
                                    "method": "rag",
                                    "retrieved_sources": len(retrieved)
                                })
                    
                        st.success(f"تم توليد {len(out.get('mcq', []))} أسئلة اختيار من متعدد و {len(out.get('tf', []))} أسئلة صح/خطأ")
                        st.info(f"وقت التوليد: {generation_time:.2f} ثانية")
                        st.info(f"تم استرجاع {len(retrieved)} مقطع من المصادر الخارجية")
                    
                        # عرض الأسئلة
                        st.session_state.rag_questions = out
                        st.session_state.rag_generated = True

                    except Exception as e:
                        st.error(f"خطأ في توليد الأسئلة: {e}")

# عرض الأسئلة المولدة
if hasattr(st.session_state, 'vanilla_generated') and st.session_state.vanilla_generated:
    st.subheader("الأسئلة المولدة (Vanilla)")
    
    # التحقق من صحة تنسيق الأسئلة وإضافة الإجابات الصحيحة المفقودة
    if not validate_questions_format(st.session_state.vanilla_questions):
        st.warning("تنسيق الأسئلة غير صحيح، سيتم محاولة إصلاحه...")
        st.session_state.vanilla_questions = add_missing_correct_answers(st.session_state.vanilla_questions)
    
    # عرض الأسئلة كواجهة تفاعلية
    answers_vanilla = render_quiz(st.session_state.vanilla_questions, prefix="vanilla_")
    
    # أزرار التحكم
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("تصحيح الإجابات (Vanilla)", key="grade_vanilla"):
            if answers_vanilla:
                score, total, results = grade(answers_vanilla, st.session_state.vanilla_questions, prefix="vanilla_")
                display_quiz_results(score, total, results)
            else:
                st.warning("يرجى حل الأسئلة أولاً قبل التصحيح")
    
    with col2:
        if st.button("حفظ الأسئلة (فقط)", key="save_vanilla", type="primary"):
            with st.spinner("جاري حفظ الأسئلة..."):
                try:
                    from src.storage import save_questions_separate_file
                    
                    # إضافة النص المصدر للبيانات
                    questions_data = st.session_state.vanilla_questions.copy()
                    questions_data["source_text"] = raw_text
                    questions_data["lang"] = lang
                    
                    filename, total_questions = save_questions_separate_file(
                        questions_data=questions_data,
                        model_name=model_name,
                        method="vanilla",
                        source_file=up.name,
                        lang=lang
                    )
                    
                    st.success(f"تم حفظ {total_questions} سؤال بنجاح!")
                    st.info(f"الملف: `outputs/{Path(up.name).stem.replace(' ', '_').replace('.', '_')}/{filename}`")
                    
                except Exception as e:
                    st.error(f"خطأ في حفظ الأسئلة: {e}")

if hasattr(st.session_state, 'rag_generated') and st.session_state.rag_generated:
    st.subheader("الأسئلة المولدة (RAG)")
    
    # عرض المصادر المستخدمة
    if st.session_state.rag_questions.get("sources"):
        with st.expander("المصادر المستخدمة في التوليد"):
            for i, source in enumerate(st.session_state.rag_questions["sources"], 1):
                st.markdown(f"""
                **المصدر {i}:**
                -  الملف: `{source['filename']}`
                -  درجة التشابه: {source['score']:.3f}
                -  النص: {source['text'][:200]}...
                """)
    
    # التحقق من صحة تنسيق الأسئلة وإضافة الإجابات الصحيحة المفقودة
    if not validate_questions_format(st.session_state.rag_questions):
        st.warning("تنسيق الأسئلة غير صحيح، سيتم محاولة إصلاحه...")
        st.session_state.rag_questions = add_missing_correct_answers(st.session_state.rag_questions)
    
    # عرض الأسئلة كواجهة تفاعلية
    answers_rag = render_quiz(st.session_state.rag_questions, prefix="rag_")
    
    # أزرار التحكم
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("تصحيح الإجابات (RAG)", key="grade_rag"):
            if answers_rag:
                score, total, results = grade(answers_rag, st.session_state.rag_questions, prefix="rag_")
                display_quiz_results(score, total, results)
            else:
                st.warning("يرجى حل الأسئلة أولاً قبل التصحيح")
    
    with col2:
        if st.button("حفظ الأسئلة (فقط)", key="save_rag", type="primary"):
            with st.spinner("جاري حفظ الأسئلة..."):
                try:
                    from src.storage import save_questions_separate_file
                    
                    # إضافة النص المصدر للبيانات
                    questions_data = st.session_state.rag_questions.copy()
                    questions_data["source_text"] = raw_text
                    questions_data["lang"] = lang
                    
                    filename, total_questions = save_questions_separate_file(
                        questions_data=questions_data,
                        model_name=model_name,
                        method="rag",
                        source_file=up.name,
                        lang=lang
                    )
                    
                    st.success(f"تم حفظ {total_questions} سؤال بنجاح!")
                    st.info(f"الملف: `outputs/{Path(up.name).stem.replace(' ', '_').replace('.', '_')}/{filename}`")
                    
                except Exception as e:
                    st.error(f"خطأ في حفظ الأسئلة: {e}")
