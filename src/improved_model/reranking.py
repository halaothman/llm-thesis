"""
Re-ranking للنسخة المحسّنة باستخدام Cross-Encoder
"""
from sentence_transformers import CrossEncoder
import numpy as np

# متغير عام لحفظ النموذج
_reranker = None

# نموذج Cross-Encoder للـ re-ranking
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def get_reranker():
    """
    الحصول على نموذج Re-ranking (lazy loading)
    
    Returns:
        CrossEncoder: نموذج Re-ranking
    """
    global _reranker
    if _reranker is None:
        print(f"[IMPROVED] جاري تحميل نموذج Re-ranking: {RERANKER_MODEL_NAME}")
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
        print("[IMPROVED] تم تحميل نموذج Re-ranking بنجاح")
    return _reranker

def rerank(query: str, passages: list[dict], top_k: int = None):
    """
    إعادة ترتيب المقاطع بناءً على الاستعلام
    
    Args:
        query: نص الاستعلام
        passages: قائمة المقاطع [{text, score, ...}, ...]
        top_k: عدد النتائج المطلوبة بعد إعادة الترتيب (إذا كان None، يُرجع جميع النتائج)
    
    Returns:
        list: قائمة المقاطع المعاد ترتيبها
    """
    if not passages or not query.strip():
        return passages
    
    if len(passages) == 1:
        return passages
    
    try:
        reranker = get_reranker()
        
        # إعداد أزواج (query, passage) للـ re-ranking
        pairs = [[query, p.get("text", "")] for p in passages]
        
        # حساب درجات Re-ranking
        rerank_scores = reranker.predict(pairs)
        
        # دمج الدرجات الأصلية مع درجات Re-ranking
        # يمكن استخدام متوسط مرجح أو درجات Re-ranking فقط
        for i, passage in enumerate(passages):
            original_score = passage.get("score", 0.0)
            rerank_score = float(rerank_scores[i])
            
            # استخدام متوسط مرجح (70% re-ranking, 30% original)
            # أو يمكن استخدام re-ranking فقط
            combined_score = 0.7 * rerank_score + 0.3 * original_score
            
            passage["rerank_score"] = rerank_score
            passage["combined_score"] = combined_score
            passage["original_score"] = original_score
        
        # ترتيب المقاطع بناءً على combined_score
        passages_sorted = sorted(passages, key=lambda x: x.get("combined_score", 0.0), reverse=True)
        
        # إرجاع top_k إذا كان محدداً
        if top_k is not None and top_k > 0:
            return passages_sorted[:top_k]
        
        return passages_sorted
        
    except Exception as e:
        print(f"[WARNING] خطأ في Re-ranking: {e}. سيتم إرجاع النتائج الأصلية.")
        return passages







