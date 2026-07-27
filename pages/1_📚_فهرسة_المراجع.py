"""صفحة فهرسة المراجع: رفع الملفات، تقسيم النص، وبناء/تحديث فهارس FAISS."""
import streamlit as st
import os
import uuid
import json
import sys
import time
from tqdm import tqdm
import pandas as pd

# إضافة المسار الجذر للمشروع إلى Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.loaders import load_text
    from src.chunking import chunk_text
    from src.arabic_text import clean_ar, stem_ar
    from src.faiss_store import build_or_update, calculate_file_checksum, get_existing_file_checksums, remove_duplicate_chunks, remove_chunks_by_source
except ImportError as e:
    st.error(f"خطأ في استيراد الوحدات: {e}")
    st.stop()

st.set_page_config(page_title="فهرسة المراجع", layout="wide")

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
    /* محاذاة خاصة بصفحة الفهرسة */
    .stProgress > div > div > div > div {
        direction: rtl;
    }
    
    .stProgress > div > div > div > div > div {
        direction: rtl;
    }
    
    /* محاذاة أزرار التقدم */
    .stProgress .stProgressLabel {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة الرسائل */
    .stMarkdown p {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة القوائم */
    .stMarkdown ul, .stMarkdown ol {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة الجداول */
    .stDataFrame table th,
    .stDataFrame table td {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة الـ metrics */
    .stMetric > div {
        direction: rtl;
        text-align: right;
    }
    
    .stMetric [data-testid="metric-container"] {
        direction: rtl;
        text-align: right;
    }
    
    /* محاذاة الـ expander */
    .streamlit-expanderHeader {
        direction: rtl;
        text-align: right;
    }
    
    .streamlit-expanderContent {
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 فهرسة المراجع الخارجية (RAG)")

# إعدادات المسارات
DATA_DIR = "data-sources"
os.makedirs("indexes", exist_ok=True)

# معلومات النظام - التحقق من النموذج المستخدم
from src.embeddings import get_model_name, get_model_info
from src.config import get_rag_version, set_rag_version, get_index_paths

# الحصول على مسارات الفهرس بناءً على النسخة المحددة
IDX_PATH, META_PATH = get_index_paths()
current_version = get_rag_version()

embedding_model_name = get_model_name()
model_info = get_model_info()

# اختيار النسخة (Baseline vs Improved)
st.subheader("🔄 نسخة RAG")
version_options = {
    "baseline": "📊 Baseline (الأصلية)",
    "improved": "✨ Improved (المحسّنة)"
}
selected_version = st.radio(
    "اختر النسخة للفهرسة:",
    options=["baseline", "improved"],
    index=0 if current_version == "baseline" else 1,
    format_func=lambda x: version_options[x],
    key="rag_version_indexing"
)

if selected_version != current_version:
    set_rag_version(selected_version)
    # تحديث المسارات
    IDX_PATH, META_PATH = get_index_paths()
    st.rerun()

# تحديث المسارات بناءً على النسخة الحالية
IDX_PATH, META_PATH = get_index_paths()

# معلومات النسخة
version_info = {
    "baseline": {
        "name": "Baseline (الأصلية)",
        "model": "intfloat/e5-large-v2",
        "description": "النسخة الأصلية قبل التحسين"
    },
    "improved": {
        "name": "Improved (المحسّنة)",
        "model": "aubmindlab/bert-base-arabertv2",
        "description": "النسخة المحسّنة بعد التحسين"
    }
}

info = version_info[current_version]
st.caption(f"{info['name']} — `{embedding_model_name}`")

# عرض حالة الفهرس الحالية
col1, col2, col3 = st.columns(3)

with col1:
    files_count = sum(len(f) for _,_,f in os.walk(DATA_DIR))
    st.metric("📁 عدد ملفات المصادر", files_count)

with col2:
    try:
        meta_lines = sum(1 for _ in open(META_PATH,"r",encoding="utf-8"))
        st.metric("📄 عدد القطع المفهرسة", meta_lines)
    except:
        st.metric("📄 عدد القطع المفهرسة", 0)

with col3:
    index_exists = os.path.exists(IDX_PATH)
    st.metric("🗂️ حالة الفهرس", "✅ موجود" if index_exists else "❌ غير موجود")

# عرض ملفات المصادر
st.subheader("📋 ملفات المصادر المتاحة")
if files_count > 0:
    all_files = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            all_files.append(os.path.join(root, f))
    
    # عرض الملفات في جدول
    file_data = []
    for file_path in all_files:
        file_size = os.path.getsize(file_path)
        file_ext = os.path.splitext(file_path)[1]
        file_data.append({
            "الملف": os.path.basename(file_path),
            "المجلد": os.path.dirname(file_path).replace(DATA_DIR, "").strip("/"),
            "النوع": file_ext,
            "الحجم (KB)": round(file_size / 1024, 2)
        })
    
    st.dataframe(file_data, use_container_width=True)
else:
    st.warning("لا توجد ملفات في مجلد المصادر")

st.caption("⚠️ الفهرسة قد تستغرق وقتاً — لا تغلق المتصفح أثناءها.")

# زر الفهرسة
st.subheader("🔧 عمليات الفهرسة")

col1, col2 = st.columns([1, 1])

with col1:
    reindex = st.button("🔁 إعادة الفهرسة (تزايدية)", type="primary", use_container_width=True)

with col2:
    clear_index = st.button("🗑️ مسح الفهرس", type="secondary", use_container_width=True)

col3, col4 = st.columns([1, 1])
with col3:
    remove_duplicates = st.button("🧹 تنظيف التكرارات", type="secondary", use_container_width=True)

if clear_index:
    try:
        if os.path.exists(IDX_PATH):
            os.remove(IDX_PATH)
        if os.path.exists(META_PATH):
            os.remove(META_PATH)
        st.success("تم مسح الفهرس بنجاح")
        st.rerun()
    except Exception as e:
        st.error(f"خطأ في مسح الفهرس: {e}")

if remove_duplicates:
    if not os.path.exists(IDX_PATH) or not os.path.exists(META_PATH):
        st.warning("⚠️ الفهرس غير موجود")
    else:
        with st.spinner("جاري تنظيف التكرارات..."):
            try:
                removed = remove_duplicate_chunks(IDX_PATH, META_PATH)
                if removed > 0:
                    st.success(f"✅ تم حذف {removed} قطعة مكررة من الفهرس")
                else:
                    st.info("ℹ️ لا توجد تكرارات في الفهرس")
                st.rerun()
            except Exception as e:
                st.error(f"❌ خطأ في تنظيف التكرارات: {e}")

if reindex:
    if files_count == 0:
        st.warning("لا توجد ملفات في مجلد المصادر")
    else:
        st.info("🚀 بدء عملية الفهرسة...")
        
        # حساب إجمالي الملفات
        total_files = sum(1 for _,_,f in os.walk(DATA_DIR) for _ in f)
        st.info(f"📁 سيتم معالجة {total_files} ملف(ات)")
        
        # تقدير الوقت المتوقع
        estimated_time = total_files * 2  # تقدير 2 ثانية لكل ملف
        st.info(f"⏱️ الوقت المتوقع: {estimated_time//60} دقيقة و {estimated_time%60} ثانية")
        
        # إنشاء containers للتقدم
        progress_container = st.container()
        status_container = st.container()
        stats_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            progress_text = st.empty()
        
        with status_container:
            status_text = st.empty()
            current_file_text = st.empty()
        
        with stats_container:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                files_processed_metric = st.metric("الملفات المعالجة", 0)
            with col2:
                chunks_created_metric = st.metric("القطع المولدة", 0)
            with col3:
                current_size_metric = st.metric("حجم النص المعالج", "0 KB")
            with col4:
                speed_metric = st.metric("السرعة", "0 ملف/ث")
        
        # متغيرات التتبع
        records = []
        counter = 0
        total_text_size = 0
        start_time = time.time()
        phase_timers = {
            "load_text": 0.0,
            "clean_stem": 0.0,
            "chunking": 0.0,
            "index_updates": 0.0,
        }
        
        # جمع قائمة الملفات المراد فهرستها
        files_to_index = []
        for root, _, files in os.walk(DATA_DIR):
            for f in files:
                path = os.path.join(root, f)
                files_to_index.append(path)
        
        # التحقق من الملفات المكررة وحذف القطع القديمة
        status_text.markdown("🔍 **جاري التحقق من الملفات المكررة...**")
        current_file_text.markdown("**المرحلة:** التحقق من الملفات المكررة")
        
        # إنشاء مؤشر تقدم خاص بمرحلة التحقق من الملفات المكررة
        duplicate_check_container = st.container()
        with duplicate_check_container:
            st.subheader("🔍 التحقق من الملفات المكررة")
            duplicate_progress_bar = st.progress(0)
            duplicate_progress_text = st.empty()
            duplicate_stats_col1, duplicate_stats_col2, duplicate_stats_col3 = st.columns(3)
            with duplicate_stats_col1:
                duplicate_files_checked_metric = st.metric("الملفات المفحوصة", 0)
            with duplicate_stats_col2:
                duplicate_files_found_metric = st.metric("الملفات المكررة", 0)
            with duplicate_stats_col3:
                chunks_removed_metric = st.metric("القطع المحذوفة", 0)
        
        existing_checksums = get_existing_file_checksums(META_PATH)
        total_removed_chunks = 0
        duplicate_files_count = 0
        duplicate_check_start_time = time.time()
        
        # حذف القطع القديمة للملفات الموجودة (لإزالة التكرارات)
        for idx, path in enumerate(files_to_index):
            # تحديث مؤشر التقدم
            progress = (idx + 1) / len(files_to_index)
            duplicate_progress_bar.progress(progress)
            duplicate_progress_text.text(f"التحقق من الملفات: {idx + 1}/{len(files_to_index)} ({progress*100:.1f}%)")
            
            # تحديث الملف الحالي
            current_file_text.markdown(f"**الملف الحالي:** `{os.path.basename(path)}`")
            
            file_checksum = calculate_file_checksum(path)
            normalized_path = os.path.normpath(path)
            
            # إذا كان الملف موجوداً في الفهرس، حذف القطع القديمة
            # (حتى لو كان checksum نفسه، لأننا نريد إعادة فهرسة لإزالة التكرارات)
            if normalized_path in existing_checksums:
                duplicate_files_count += 1
                duplicate_files_found_metric.metric("الملفات المكررة", duplicate_files_count)
                
                status_text.markdown(f"🗑️ **جاري حذف القطع القديمة للملف:** `{os.path.basename(path)}`")
                removed = remove_chunks_by_source(IDX_PATH, META_PATH, path)
                total_removed_chunks += removed
                chunks_removed_metric.metric("القطع المحذوفة", total_removed_chunks)
                
                if removed > 0:
                    status_text.markdown(f"✅ **تم حذف {removed} قطعة قديمة للملف:** `{os.path.basename(path)}`")
            
            # تحديث عدد الملفات المفحوصة
            duplicate_files_checked_metric.metric("الملفات المفحوصة", idx + 1)
        
        # إكمال مؤشر التقدم
        duplicate_progress_bar.progress(1.0)
        duplicate_progress_text.text(f"تم التحقق من جميع الملفات: {len(files_to_index)}/{len(files_to_index)} (100%)")
        
        duplicate_check_time = time.time() - duplicate_check_start_time
        
        if total_removed_chunks > 0:
            st.success(f"✅ **تم حذف {total_removed_chunks} قطعة قديمة من {duplicate_files_count} ملف(ات) مكرر(ة) في {duplicate_check_time:.1f} ثانية**")
        else:
            st.info(f"ℹ️ **لم يتم العثور على ملفات مكررة. تم التحقق من {len(files_to_index)} ملف(ات) في {duplicate_check_time:.1f} ثانية**")
        
        # إعادة تعيين مؤشرات التقدم للفهرسة
        st.divider()
        
        # إنشاء قسم منفصل لعملية الفهرسة
        indexing_container = st.container()
        with indexing_container:
            st.subheader("📚 عملية الفهرسة")
            indexing_progress_bar = st.progress(0)
            indexing_progress_text = st.empty()
            indexing_phase_text = st.empty()
            
            indexing_stats_col1, indexing_stats_col2, indexing_stats_col3, indexing_stats_col4 = st.columns(4)
            with indexing_stats_col1:
                indexing_files_processed_metric = st.metric("الملفات المعالجة", 0)
            with indexing_stats_col2:
                indexing_chunks_created_metric = st.metric("القطع المولدة", 0)
            with indexing_stats_col3:
                indexing_size_metric = st.metric("حجم النص المعالج", "0 KB")
            with indexing_stats_col4:
                indexing_speed_metric = st.metric("السرعة", "0 ملف/ث")
        
        status_text.markdown("📚 **بدء عملية الفهرسة...**")
        current_file_text.markdown("**المرحلة:** فهرسة الملفات")
        
        # معالجة الملفات
        for file_idx, path in enumerate(files_to_index):
            f = os.path.basename(path)
            file_start_time = time.time()
            
            try:
                # تحديث حالة الملف الحالي
                current_file_text.markdown(f"**الملف الحالي:** `{f}` ({file_idx + 1}/{len(files_to_index)})")
                
                # تحديث مؤشر التقدم العام
                file_progress = (file_idx + 1) / len(files_to_index)
                indexing_progress_bar.progress(file_progress)
                indexing_progress_text.text(f"التقدم: {file_idx + 1}/{len(files_to_index)} ملف ({file_progress*100:.1f}%)")
                
                # تحديث المرحلة الحالية
                indexing_phase_text.markdown("📖 **المرحلة:** تحميل الملف...")
                status_text.markdown(f"📖 **جاري تحميل الملف:** `{f}`")
                
                # تحميل النص
                load_start = time.time()
                text = load_text(path)
                phase_timers["load_text"] += time.time() - load_start
                if not text.strip():
                    st.warning(f"⚠️ ملف فارغ: {f}")
                    continue
                
                file_size = len(text.encode('utf-8')) / 1024  # KB
                total_text_size += file_size
                
                # تحديث المرحلة
                indexing_phase_text.markdown("🧹 **المرحلة:** تنظيف وتجذيع النص...")
                status_text.markdown(f"🧹 **جاري تنظيف النص:** `{f}`")
                
                # تنظيف النص
                clean_start = time.time()
                text = clean_ar(text)
                stemmed = stem_ar(text)
                phase_timers["clean_stem"] += time.time() - clean_start
                
                # تحديث المرحلة
                indexing_phase_text.markdown("✂️ **المرحلة:** تقسيم النص إلى قطع...")
                status_text.markdown(f"✂️ **جاري تقسيم النص إلى قطع:** `{f}`")
                
                # تقسيم إلى قطع
                chunk_start = time.time()
                chunks = chunk_text(stemmed, 500, 100)
                phase_timers["chunking"] += time.time() - chunk_start
                
                # تحديث المرحلة
                indexing_phase_text.markdown("🔄 **المرحلة:** إعداد القطع للفهرسة...")
                status_text.markdown(f"🔄 **جاري إعداد القطع:** `{f}`")
                
                # حساب checksum للملف لتتبع التغييرات
                file_checksum = calculate_file_checksum(path)
                
                # إضافة القطع مع metadata محسّن (للنسخة المحسّنة فقط)
                from src.config import get_rag_version
                is_improved = get_rag_version() == "improved"
                from datetime import datetime
                
                file_chunks = 0
                for chunk_idx, ch in enumerate(chunks):
                    if ch.strip():  # تجاهل القطع الفارغة
                        rid = int(uuid.uuid4().int % 1_000_000_000)
                        
                        # Metadata أساسي (لجميع النسخ)
                        base_metadata = {
                            "source": path,
                            "filename": f,
                            "chunk_size": len(ch),
                            "file_size_kb": file_size,
                            "file_checksum": file_checksum
                        }
                        
                        # Metadata محسّن (للنسخة المحسّنة فقط)
                        if is_improved:
                            # حساب موضع القطعة في النص
                            chunk_start = chunk_idx * (500 - 100)  # 500 حجم، 100 تداخل
                            chunk_end = min(chunk_start + len(ch), len(text))
                            
                            # استخراج السياق (قبل وبعد القطعة)
                            context_before = text[max(0, chunk_start - 100):chunk_start]
                            context_after = text[chunk_end:min(chunk_end + 100, len(text))]
                            context = (context_before + " ... " + context_after).strip()
                            
                            # معلومات إضافية
                            enhanced_metadata = {
                                **base_metadata,
                                "chunk_index": chunk_idx,
                                "chunk_position": {
                                    "start": chunk_start,
                                    "end": chunk_end,
                                    "length": len(ch)
                                },
                                "page": None,  # يمكن تحسينه لاحقاً لملفات PDF
                                "paragraph": chunk_idx,  # رقم الفقرة/القطعة
                                "context": context,
                                "file_type": os.path.splitext(f)[1].lower(),
                                "indexed_at": datetime.now().isoformat()
                            }
                            records.append({
                                "id": rid, 
                                "text": ch, 
                                "metadata": enhanced_metadata
                            })
                        else:
                            # Baseline: metadata أساسي فقط
                            records.append({
                                "id": rid, 
                                "text": ch, 
                                "metadata": base_metadata
                            })
                        file_chunks += 1
                
                # حساب الوقت والسرعة
                file_time = time.time() - file_start_time
                elapsed_time = time.time() - start_time
                speed = (file_idx + 1) / elapsed_time if elapsed_time > 0 else 0
                
                # تحديث التقدم
                counter += 1
                
                # تحديث الإحصائيات في قسم الفهرسة
                indexing_files_processed_metric.metric("الملفات المعالجة", counter)
                indexing_chunks_created_metric.metric("القطع المولدة", len(records))
                indexing_size_metric.metric("حجم النص المعالج", f"{total_text_size:.1f} KB")
                indexing_speed_metric.metric("السرعة", f"{speed:.2f} ملف/ث")
                
                # تحديث الإحصائيات القديمة (للتوافق)
                files_processed_metric.metric("الملفات المعالجة", counter)
                chunks_created_metric.metric("القطع المولدة", len(records))
                current_size_metric.metric("حجم النص المعالج", f"{total_text_size:.1f} KB")
                speed_metric.metric("السرعة", f"{speed:.2f} ملف/ث")
                
                # تحديث المرحلة
                indexing_phase_text.markdown(f"✅ **تم معالجة الملف:** `{f}` - {file_chunks} قطعة")
                
                # رسالة نجاح الملف
                status_text.markdown(f"✅ **تم معالجة:** `{f}` - {file_chunks} قطعة - {file_size:.1f} KB - {file_time:.1f}ث")
                    
                # تقدير الوقت المتبقي
                if speed > 0:
                    remaining_files = len(files_to_index) - (file_idx + 1)
                    eta_seconds = remaining_files / speed
                    eta_minutes = int(eta_seconds // 60)
                    eta_secs = int(eta_seconds % 60)
                    status_text.markdown(f"⏳ **الوقت المتبقي:** {eta_minutes}د {eta_secs}ث")
                
            except Exception as e:
                st.error(f"❌ خطأ في معالجة {f}: {e}")
                continue
        
        # حفظ الفهرس
        if records:
            indexing_phase_text.markdown("💾 **المرحلة:** حفظ الفهرس في FAISS...")
            status_text.markdown("💾 **جاري حفظ الفهرس...**")
            
            # تقسيم إلى batches لتجنب مشاكل الذاكرة
            batch_size = 16
            total_batches = (len(records) + batch_size - 1) // batch_size
            
            # إحصائيات حذف التكرارات
            total_skipped = 0
            total_removed = 0
            
            # إنشاء مؤشر تقدم خاص بحفظ الفهرس
            saving_progress_bar = st.progress(0)
            saving_progress_text = st.empty()
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                # تحديث مؤشر التقدم للحفظ
                saving_progress = batch_num / total_batches
                saving_progress_bar.progress(saving_progress)
                saving_progress_text.text(f"حفظ الدفعة: {batch_num}/{total_batches} ({saving_progress*100:.1f}%)")
                
                indexing_phase_text.markdown(f"💾 **المرحلة:** حفظ الدفعة {batch_num}/{total_batches} ({len(batch)} قطعة)...")
                status_text.markdown(f"💾 **حفظ الدفعة {batch_num}/{total_batches} ({len(batch)} قطعة)...**")
                index_start = time.time()
                
                # استخدام build_or_update مع فحص التكرارات
                # remove_existing_source=True: يحذف القطع القديمة للملف قبل الإضافة
                # check_duplicates=True: يتحقق من التكرارات في النصوص
                
                # حساب عدد القطع قبل الفحص
                before_count = len(batch)
                
                # حفظ مع فحص التكرارات (لا نحتاج remove_existing_source لأننا حذفناها مسبقاً)
                build_or_update(
                    IDX_PATH, 
                    META_PATH, 
                    batch,
                    check_duplicates=True,  # فحص التكرارات في النصوص
                    remove_existing_source=False  # تم حذف القطع القديمة مسبقاً
                )
                
                phase_timers["index_updates"] += time.time() - index_start
                
                # تحديث التقدم العام
                indexing_progress_bar.progress(1.0)
                indexing_progress_text.text(f"حفظ الفهرس: {batch_num}/{total_batches} دفعة")
            
            # إكمال مؤشرات التقدم
            saving_progress_bar.progress(1.0)
            saving_progress_text.text(f"تم حفظ جميع الدفعات: {total_batches}/{total_batches} (100%)")
            indexing_progress_bar.progress(1.0)
            indexing_progress_text.text(f"تمت الفهرسة بنجاح: {len(files_to_index)}/{len(files_to_index)} ملف (100%)")
            indexing_phase_text.markdown("✅ **اكتملت عملية الفهرسة بنجاح!**")
            progress_bar.progress(1.0)
            progress_text.text("تمت الفهرسة بنجاح!")
            status_text.markdown("🎉 **تمت الفهرسة بنجاح!**")
            
            # النتائج النهائية
            total_time = time.time() - start_time
            st.success(f"✅ **تمت فهرسة {len(records)} قطعة من {counter} ملف(ات) في {total_time:.1f} ثانية**")

            # عرض مخطط زمني مختصر لمراحل الفهرسة لتوثيقها بسهولة
            st.subheader("🧭 المخطط الزمني للفهرسة")
            phases_labels = {
                "load_text": "تحميل الملفات",
                "clean_stem": "تنظيف وتجذيع النصوص",
                "chunking": "تقطيع النصوص",
                "index_updates": "كتابة الفهرس (FAISS)",
            }
            timing_rows = []
            for key, label in phases_labels.items():
                duration = phase_timers.get(key, 0.0)
                percentage = (duration / total_time * 100) if total_time > 0 else 0
                timing_rows.append(
                    {
                        "المرحلة": label,
                        "الزمن (ثانية)": f"{duration:.1f}",
                        "الزمن (دقيقة)": f"{duration/60:.1f}",
                        "النسبة التقريبية": f"{percentage:.0f}%",
                    }
                )
            timing_rows.append(
                {
                    "المرحلة": "الإجمالي",
                    "الزمن (ثانية)": f"{total_time:.1f}",
                    "الزمن (دقيقة)": f"{total_time/60:.1f}",
                    "النسبة التقريبية": "100%",
                }
            )
            st.table(pd.DataFrame(timing_rows))
            
            # عرض إحصائيات مفصلة
            st.subheader("📊 الإحصائيات النهائية")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("الملفات المعالجة", counter)
            with col2:
                st.metric("القطع المولدة", len(records))
            with col3:
                avg_chunks = len(records) / counter if counter > 0 else 0
                st.metric("متوسط القطع/ملف", f"{avg_chunks:.1f}")
            with col4:
                avg_speed = counter / total_time if total_time > 0 else 0
                st.metric("متوسط السرعة", f"{avg_speed:.1f} ملف/ث")
            
            # إحصائيات إضافية
            col1, col2 = st.columns(2)
            with col1:
                st.metric("إجمالي حجم النصوص", f"{total_text_size:.1f} KB")
            with col2:
                st.metric("متوسط حجم الملف", f"{total_text_size/counter:.1f} KB" if counter > 0 else "0 KB")
                
        else:
            st.error("❌ لم يتم إنشاء أي قطع للفهرسة")