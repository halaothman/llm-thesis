"""
نسخة Improved: RAG المحسّن (بعد التحسين)
- TOP_K: 10
- SIMILARITY_THRESHOLD: 0.65
- نموذج التضمين: AraBERT
- Re-ranking: Cross-Encoder
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

def retrieve(index_path: str, meta_path: str, query: str, top_k: int = IMPROVED_TOP_K, thr: float = IMPROVED_THRESHOLD):
    """
    استرجاع المقاطع المشابهة من الفهرس - Improved (النسخة المحسّنة)
    مع Re-ranking دائماً
    
    Args:
        index_path: مسار ملف الفهرس
        meta_path: مسار ملف الميتاداتا
        query: نص الاستعلام
        top_k: عدد مرشّحي FAISS قبل Re-ranking (افتراضي: 10)
        thr: عتبة التشابه (افتراضي: 0.65 - Improved)
    
    Returns:
        list: أفضل IMPROVED_RERANK_TOP_K مقاطع بعد Re-ranking
    """
    if not query.strip():
        return []
    
    # تحويل الاستعلام إلى تضمين (is_query=True للبحث)
    qv = embed_texts([query], is_query=True)
    
    # FAISS: top-10 ثم Re-ranking يختار أفضل 5
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
            meta_dict = {k: v for k, v in meta_dict.items() if k != "chunk_index"}
            source = meta_dict.get("source", "")
            results.append({
                "text": metadata.get("text", ""),
                "filename": os.path.basename(source) if source else "",
                "score": score,
                "id": int(idx),
                "metadata": meta_dict,
            })
    
    return rerank(query, results, top_k=IMPROVED_RERANK_TOP_K)

