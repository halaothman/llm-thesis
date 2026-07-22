"""
نسخة Improved: RAG المحسّن (بعد التحسين)
- TOP_K: 10
- SIMILARITY_THRESHOLD: 0.65
- نموذج التضمين: AraBERT
- Re-ranking: Cross-Encoder
- Metadata محسّن: صفحة، فقرة، سياق
"""
import os
import numpy as np
from .embeddings_improved import embed_texts
from .reranking import rerank
from ..faiss_store import search, load_meta

# إعدادات Improved
IMPROVED_TOP_K = 10
IMPROVED_THRESHOLD = 0.65
IMPROVED_RERANK_TOP_K = 5  # عدد النتائج بعد Re-ranking
USE_RERANKING = True  # تفعيل/تعطيل Re-ranking

def retrieve(index_path: str, meta_path: str, query: str, top_k: int = IMPROVED_TOP_K, thr: float = IMPROVED_THRESHOLD, use_reranking: bool = USE_RERANKING):
    """
    استرجاع المقاطع المشابهة من الفهرس - Improved (النسخة المحسّنة)
    مع Re-ranking و Metadata محسّن
    
    Args:
        index_path: مسار ملف الفهرس
        meta_path: مسار ملف الميتاداتا
        query: نص الاستعلام
        top_k: عدد النتائج المطلوبة (افتراضي: 10 - Improved)
        thr: عتبة التشابه (افتراضي: 0.65 - Improved)
        use_reranking: استخدام Re-ranking (افتراضي: True)
    
    Returns:
        list: قائمة المقاطع المسترجعة مع metadata محسّن
    """
    if not query.strip():
        return []
    
    # تحويل الاستعلام إلى تضمين (is_query=True للبحث)
    qv = embed_texts([query], is_query=True)
    
    # البحث في الفهرس (نسترجع أكثر من المطلوب للـ re-ranking)
    initial_top_k = top_k * 2 if use_reranking else top_k
    scores, ids = search(index_path, qv, initial_top_k)
    
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
    
    # جمع النتائج الأولية
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
            meta_dict = metadata.get("metadata", {})
            
            # بناء metadata محسّن
            enhanced_metadata = {
                "source": meta_dict.get("source", ""),
                "chunk_index": meta_dict.get("chunk_index", -1),
                "chunk_position": meta_dict.get("chunk_position", {}),
                "page": meta_dict.get("page", None),
                "paragraph": meta_dict.get("paragraph", None),
                "context": meta_dict.get("context", ""),
                "file_type": meta_dict.get("file_type", ""),
                "indexed_at": meta_dict.get("indexed_at", ""),
            }
            
            results.append({
                "text": metadata.get("text", ""),
                "filename": os.path.basename(meta_dict.get("source", "")) if meta_dict.get("source") else "",
                "score": score,
                "id": int(idx),
                "metadata": enhanced_metadata,
                "original_metadata": meta_dict  # للتوافق مع الكود القديم
            })
    
    # تطبيق Re-ranking إذا كان مفعلاً
    if use_reranking and len(results) > 1:
        results = rerank(query, results, top_k=IMPROVED_RERANK_TOP_K)
    
    # إرجاع top_k النهائي
    return results[:top_k]

