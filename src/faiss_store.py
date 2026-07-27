"""إنشاء وتحديث والبحث في فهرس FAISS مع ملف metadata بصيغة JSONL."""
import faiss
import json
import os
import numpy as np
import hashlib
from .embeddings import embed_texts

def build_or_update(index_path: str, meta_path: str, records: list[dict], 
                    check_duplicates: bool = True, remove_existing_source: bool = True):
    """
    بناء أو تحديث فهرس FAISS تدريجياً مع فحص التكرارات
    
    Args:
        index_path: مسار ملف الفهرس
        meta_path: مسار ملف الميتاداتا
        records: قائمة السجلات [{id, text, metadata}]
        check_duplicates: إذا كان True، يتحقق من التكرارات في النصوص
        remove_existing_source: إذا كان True، يحذف القطع القديمة للملف قبل الإضافة
    """
    if not records:
        return
    
    # إذا كان remove_existing_source فعالاً، حذف القطع القديمة للملف
    if remove_existing_source and records:
        # الحصول على مسار الملف من أول سجل
        source_path = records[0].get("metadata", {}).get("source", "")
        if source_path:
            removed_count = remove_chunks_by_source(index_path, meta_path, source_path)
            if removed_count > 0:
                print(f"تم حذف {removed_count} قطعة قديمة للملف: {source_path}")
    
    # إذا كان check_duplicates فعالاً، تصفية القطع المكررة
    if check_duplicates:
        existing_hashes = get_existing_text_hashes(meta_path)
        unique_records = []
        skipped_count = 0
        
        for r in records:
            text = r.get("text", "")
            if text:
                text_hash = calculate_text_hash(text)
                if text_hash not in existing_hashes:
                    unique_records.append(r)
                    existing_hashes.add(text_hash)  # تحديث المجموعة لتجنب التكرار في نفس الدفعة
                else:
                    skipped_count += 1
        
        if skipped_count > 0:
            print(f"تم تخطي {skipped_count} قطعة مكررة")
        
        records = unique_records
    
    if not records:
        print("لا توجد قطع جديدة للإضافة بعد تصفية التكرارات")
        return
    
    # تحويل النصوص إلى تضمينات
    texts = [r["text"] for r in records]
    # عرض معلومات النموذج المستخدم (فقط في أول مرة)
    from .embeddings import get_model_name
    if len(texts) > 0:
        model_name = get_model_name()
        print(f"[INFO] استخدام نموذج التضمين: {model_name}")
        print(f"[INFO] جاري تحويل {len(texts)} نص إلى تضمينات...")
    # is_query=False لأن هذه نصوص للفهرسة وليست استعلامات
    vecs = embed_texts(texts, is_query=False).astype("float32")
    ids = np.array([int(r["id"]) for r in records]).astype("int64")
    
    # تحميل أو إنشاء الفهرس
    index = None
    if os.path.exists(index_path):
        try:
            # التحقق من صحة ملف الفهرس
            file_size = os.path.getsize(index_path)
            if file_size > 0:
                index = faiss.read_index(index_path)
            else:
                # ملف فارغ، سننشئ فهرس جديد
                index = None
        except Exception as e:
            print(f"خطأ في قراءة الفهرس: {e}")
            # في حالة الخطأ، سننشئ فهرس جديد
            index = None
    
    # إنشاء فهرس جديد إذا لم يكن موجوداً أو كان تالفاً
    if index is None:
        d = vecs.shape[1]
        index = faiss.IndexIDMap(faiss.IndexFlatIP(d))
    
    # إضافة السجلات الجديدة
    index.add_with_ids(vecs, ids)
    
    # حفظ الفهرس
    try:
        faiss.write_index(index, index_path)
    except Exception as e:
        print(f"خطأ في حفظ الفهرس: {e}")
        # إنشاء مجلد indexes إذا لم يكن موجوداً
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
    
    # حفظ الميتاداتا
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def search(index_path: str, query_vec: np.ndarray, top_k: int = 5):
    """
    البحث في فهرس FAISS
    
    Args:
        index_path: مسار ملف الفهرس
        query_vec: متجه الاستعلام
        top_k: عدد النتائج المطلوبة
    
    Returns:
        tuple: (scores, indices)
    """
    if not os.path.exists(index_path):
        return np.array([]), np.array([])
    
    try:
        # التحقق من صحة ملف الفهرس
        file_size = os.path.getsize(index_path)
        if file_size == 0:
            return np.array([]), np.array([])
        
        index = faiss.read_index(index_path)
        
        # التأكد من أن query_vec له الشكل الصحيح
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        
        D, I = index.search(query_vec.astype("float32"), top_k)
        
        # التأكد من إرجاع arrays صحيحة
        if D.shape[0] > 0 and I.shape[0] > 0:
            return D[0], I[0]
        else:
            return np.array([]), np.array([])
            
    except Exception as e:
        print(f"خطأ في البحث في الفهرس: {e}")
        return np.array([]), np.array([])

def load_meta(meta_path: str) -> dict:
    """
    تحميل الميتاداتا من الملف
    
    Args:
        meta_path: مسار ملف الميتاداتا
    
    Returns:
        dict: قاموس الميتاداتا
    """
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        meta[r["id"]] = r
                    except json.JSONDecodeError:
                        continue
    return meta

def calculate_text_hash(text: str) -> str:
    """
    حساب hash للنص للتحقق من التكرارات
    
    Args:
        text: النص المراد حساب hash له
    
    Returns:
        str: hash النص
    """
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()

def calculate_file_checksum(file_path: str) -> str:
    """
    حساب checksum للملف لتتبع التغييرات
    
    Args:
        file_path: مسار الملف
    
    Returns:
        str: checksum الملف
    """
    import hashlib
    if not os.path.exists(file_path):
        return ""
    
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"خطأ في حساب checksum للملف {file_path}: {e}")
        return ""

def remove_chunks_by_source(index_path: str, meta_path: str, source_path: str) -> int:
    """
    حذف جميع القطع المرتبطة بملف معين من الفهرس
    
    Args:
        index_path: مسار ملف الفهرس
        meta_path: مسار ملف الميتاداتا
        source_path: مسار الملف المصدر المراد حذف قطعته
    
    Returns:
        int: عدد القطع المحذوفة
    """
    if not os.path.exists(meta_path):
        return 0
    
    # قراءة جميع السجلات
    all_records = []
    chunks_to_remove = []
    
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    r = json.loads(line)
                    # تحويل المسار إلى نفس التنسيق للمقارنة
                    record_source = r.get("metadata", {}).get("source", "")
                    # المقارنة مع normalize المسارات
                    if os.path.normpath(record_source) == os.path.normpath(source_path):
                        chunks_to_remove.append(r["id"])
                    else:
                        all_records.append(r)
                except json.JSONDecodeError:
                    continue
    
    if not chunks_to_remove:
        return 0
    
    # إذا كان هناك فهرس، حذف القطع منه
    if os.path.exists(index_path):
        try:
            file_size = os.path.getsize(index_path)
            if file_size > 0:
                index = faiss.read_index(index_path)
                
                # إنشاء فهرس جديد بدون القطع المحذوفة
                # FAISS لا يدعم الحذف المباشر، لذا سنعيد بناء الفهرس
                if len(all_records) == 0:
                    # لا توجد قطع متبقية، حذف الفهرس
                    os.remove(index_path)
                    # إعادة كتابة ملف الميتاداتا فارغ
                    with open(meta_path, "w", encoding="utf-8") as f:
                        pass
                    return len(chunks_to_remove)
                
                # إعادة بناء الفهرس من القطع المتبقية
                texts = [r["text"] for r in all_records]
                vecs = embed_texts(texts, is_query=False).astype("float32")
                ids = np.array([int(r["id"]) for r in all_records]).astype("int64")
                
                # إنشاء فهرس جديد
                d = vecs.shape[1]
                new_index = faiss.IndexIDMap(faiss.IndexFlatIP(d))
                new_index.add_with_ids(vecs, ids)
                
                # حفظ الفهرس الجديد
                faiss.write_index(new_index, index_path)
        except Exception as e:
            print(f"خطأ في حذف القطع من الفهرس: {e}")
            # إذا فشل الحذف من الفهرس، نتابع مع حذف الميتاداتا فقط
    
    # إعادة كتابة ملف الميتاداتا بدون القطع المحذوفة
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    return len(chunks_to_remove)

def get_existing_text_hashes(meta_path: str) -> set:
    """
    الحصول على جميع hashes النصوص الموجودة في الفهرس
    
    Args:
        meta_path: مسار ملف الميتاداتا
    
    Returns:
        set: مجموعة hashes النصوص الموجودة
    """
    hashes = set()
    if not os.path.exists(meta_path):
        return hashes
    
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    r = json.loads(line)
                    text = r.get("text", "")
                    if text:
                        text_hash = calculate_text_hash(text)
                        hashes.add(text_hash)
                except json.JSONDecodeError:
                    continue
    return hashes

def get_existing_file_checksums(meta_path: str) -> dict:
    """
    الحصول على checksums الملفات الموجودة في الفهرس
    
    Args:
        meta_path: مسار ملف الميتاداتا
    
    Returns:
        dict: قاموس {source_path: checksum}
    """
    checksums = {}
    if not os.path.exists(meta_path):
        return checksums
    
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    r = json.loads(line)
                    source = r.get("metadata", {}).get("source", "")
                    # إذا كان checksum موجوداً، استخدمه، وإلا استخدم source كمفتاح
                    checksum = r.get("metadata", {}).get("file_checksum", "")
                    if source:
                        normalized_source = os.path.normpath(source)
                        # إذا كان الملف موجوداً في الفهرس (حتى بدون checksum)، نعتبره موجوداً
                        checksums[normalized_source] = checksum if checksum else "exists"
                except json.JSONDecodeError:
                    continue
    return checksums

def remove_duplicate_chunks(index_path: str, meta_path: str) -> int:
    """
    إزالة جميع القطع المكررة من الفهرس
    
    Args:
        index_path: مسار ملف الفهرس
        meta_path: مسار ملف الميتاداتا
    
    Returns:
        int: عدد القطع المحذوفة
    """
    if not os.path.exists(meta_path):
        return 0
    
    # قراءة جميع السجلات
    all_records = []
    seen_hashes = {}
    duplicates = []
    
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    r = json.loads(line)
                    text = r.get("text", "")
                    if text:
                        text_hash = calculate_text_hash(text)
                        if text_hash not in seen_hashes:
                            seen_hashes[text_hash] = r["id"]
                            all_records.append(r)
                        else:
                            duplicates.append(r["id"])
                except json.JSONDecodeError:
                    continue
    
    if not duplicates:
        return 0
    
    # إعادة بناء الفهرس بدون التكرارات
    if os.path.exists(index_path):
        try:
            file_size = os.path.getsize(index_path)
            if file_size > 0 and len(all_records) > 0:
                # إعادة بناء الفهرس من القطع الفريدة
                texts = [r["text"] for r in all_records]
                vecs = embed_texts(texts, is_query=False).astype("float32")
                ids = np.array([int(r["id"]) for r in all_records]).astype("int64")
                
                # إنشاء فهرس جديد
                d = vecs.shape[1]
                new_index = faiss.IndexIDMap(faiss.IndexFlatIP(d))
                new_index.add_with_ids(vecs, ids)
                
                # حفظ الفهرس الجديد
                faiss.write_index(new_index, index_path)
        except Exception as e:
            print(f"خطأ في إعادة بناء الفهرس: {e}")
            # إذا فشل، نتابع مع حذف الميتاداتا فقط
    
    # إعادة كتابة ملف الميتاداتا بدون التكرارات
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    return len(duplicates)