"""
نسخة Baseline: RAG الأصلي (قبل التحسين)
- TOP_K: 5
- SIMILARITY_THRESHOLD: 0.82
- نموذج التضمين: e5-large-v2
"""
import numpy as np
from .embeddings_baseline import embed_texts
from ..faiss_store import search, load_meta

# إعدادات Baseline
BASELINE_TOP_K = 5
BASELINE_THRESHOLD = 0.82

def retrieve(index_path: str, meta_path: str, query: str, top_k: int = BASELINE_TOP_K, thr: float = BASELINE_THRESHOLD):
    """
    استرجاع المقاطع المشابهة من الفهرس - Baseline (النسخة الأصلية)
    
    Args:
        index_path: مسار ملف الفهرس
        meta_path: مسار ملف الميتاداتا
        query: نص الاستعلام
        top_k: عدد النتائج المطلوبة (افتراضي: 5 - Baseline)
        thr: عتبة التشابه (افتراضي: 0.82 - Baseline)
    
    Returns:
        list: قائمة المقاطع المسترجعة
    """
    if not query.strip():
        return []
    
    # تحويل الاستعلام إلى تضمين (النسخة القديمة لا تميز بين query و passage)
    qv = embed_texts([query])
    
    # البحث في الفهرس
    scores, ids = search(index_path, qv, top_k)
    
    # تحميل الميتاداتا
    meta = load_meta(meta_path)
    
    # معالجة النتائج
    results = []
    
    # التحقق من أن النتائج ليست فارغة
    if len(scores) == 0 or len(ids) == 0:
        return results
    
    # التأكد من أن scores و ids هما arrays
    if not hasattr(scores, '__iter__'):
        scores = [scores]
    if not hasattr(ids, '__iter__'):
        ids = [ids]
    
    for score, idx in zip(scores, ids):
        if idx == -1:
            continue
        
        # تحويل score إلى float بأمان
        try:
            score = float(score)
        except (ValueError, TypeError):
            continue
        
        if score >= thr:
            metadata = meta.get(int(idx), {})
            results.append({
                "text": metadata.get("text", ""),
                "filename": metadata.get("metadata", {}).get("source", ""),
                "score": score,
                "id": int(idx),
                "metadata": metadata.get("metadata", {})
            })
    
    return results







