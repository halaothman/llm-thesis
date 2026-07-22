import numpy as np
import pandas as pd
from .embeddings import embed_texts
import re
from collections import Counter
import math

def cosine(a, b):
    """حساب التشابه الدلالي باستخدام cosine similarity"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))

def semantic_scores(questions, base_text):
    """حساب درجات التشابه الدلالي للأسئلة مع النص الأساسي"""
    qv = embed_texts(questions)
    bv = embed_texts([base_text])[0]
    sims = [float(np.dot(v, bv)) for v in qv]  # normalized => cosine
    # ممكن نعرّف relevance = sims نفسها أو نأخذ متوسط top-k ضمن النص الأصلي
    relevance = sims
    return sims, relevance

def perplexity_list(texts, lang):
    """حساب perplexity للنصوص باستخدام نماذج خفيفة"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        import math
        
        # استخدام نموذج أصغر وأسرع
        model_id = "aubmindlab/araGPT2-base" if lang == "ar" else "gpt2"
        
        # تحميل النموذج والـ tokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        ppl = []
        for t in texts:
            if not t or len(t.strip()) < 3:  # تخطي النصوص القصيرة جداً
                ppl.append(0.0)
                continue
            
            # تطبيع النص قبل حساب Perplexity
            normalized_text = normalize_text_for_perplexity(t, lang)
            if not normalized_text or len(normalized_text.strip()) < 3:
                ppl.append(0.0)
                continue
                
            try:
                enc = tok(normalized_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = mdl(**enc, labels=enc["input_ids"])
                    loss = outputs.loss
                    perplexity = math.exp(loss.item())
                    # تحديد حد أقصى للـ Perplexity لتجنب القيم الشاذة
                    # إذا كانت القيمة >= 1000، نضع 0.0 للإشارة إلى فشل الحساب (سيتم إعادة الحساب لاحقاً)
                    if perplexity >= 1000.0:
                        ppl.append(0.0)  # 0.0 يعني فشل الحساب (سيتم إعادة الحساب)
                    else:
                        ppl.append(perplexity)
            except Exception as e:
                print(f"خطأ في حساب perplexity للنص: {e}")
                ppl.append(0.0)
        return ppl
    except Exception as e:
        print(f"خطأ في تحميل النموذج: {e}")
        return [0.0] * len(texts)

def normalize_by_count(values, count):
    """تطبيع القيم حسب العدد"""
    if not values:
        return values
    arr = np.array(values, dtype=float)
    return (arr / max(count, 1)).tolist()

def tokenize_arabic(text):
    """تقسيم النص العربي إلى كلمات"""
    # إزالة علامات الترقيم والحفاظ على الكلمات العربية
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]', ' ', text)
    words = text.split()
    return [word.strip() for word in words if word.strip()]

def tokenize_english(text):
    """تقسيم النص الإنجليزي إلى كلمات"""
    # إزالة علامات الترقيم والحفاظ على الحروف الإنجليزية
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    return [word.strip().lower() for word in words if word.strip()]

def tokenize_text(text, lang="ar"):
    """تقسيم النص حسب اللغة"""
    if lang == "ar":
        return tokenize_arabic(text)
    else:
        return tokenize_english(text)

def normalize_text_for_perplexity(text, lang="ar"):
    """تطبيع النص لحساب Perplexity"""
    if not text:
        return ""
    
    # إزالة الرموز الخاصة والأسماء الأجنبية
    if lang == "ar":
        # إزالة الرموز غير العربية
        text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
        # إزالة الأسماء الأجنبية (كلمات تحتوي على حروف لاتينية)
        text = re.sub(r'\b[A-Za-z]+\b', '[اسم]', text)
        # إزالة الأرقام
        text = re.sub(r'\d+', '[رقم]', text)
        # إزالة المسافات الزائدة
        text = re.sub(r'\s+', ' ', text).strip()
    else:
        # للغة الإنجليزية
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_bleu_score(candidate, reference, lang="ar"):
    """حساب BLEU score"""
    try:
        from sacrebleu import BLEU
        
        # التأكد من وجود النصوص
        if not candidate or not reference:
            return 0.0
        
        # حساب BLEU-4 مع تفعيل effective_order
        bleu = BLEU(effective_order=True)
        score = bleu.sentence_score(candidate, [reference])
        return score.score / 100.0  # تحويل إلى نسبة من 0-1
    except Exception as e:
        print(f"خطأ في حساب BLEU: {e}")
        return 0.0

def calculate_bert_score(candidate, reference, lang="ar"):
    """حساب BERTScore"""
    try:
        from bert_score import score as bert_score
        
        # التأكد من وجود النصوص
        if not candidate or not reference:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
        
        # حساب BERTScore مع استخدام نموذج متعدد اللغات
        model_type = "bert-base-multilingual-cased" if lang == "ar" else "bert-base-uncased"
        P, R, F1 = bert_score([candidate], [reference], model_type=model_type, verbose=False)
        return {
            "precision": float(P[0]),
            "recall": float(R[0]),
            "f1_score": float(F1[0])
        }
    except Exception as e:
        print(f"خطأ في حساب BERTScore: {e}")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }

def calculate_precision_recall_f1(candidate, reference, lang="ar"):
    """حساب Precision, Recall, F1 Score"""
    try:
        # التأكد من وجود النصوص
        if not candidate or not reference:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
        candidate_tokens = tokenize_text(candidate, lang)
        reference_tokens = tokenize_text(reference, lang)
        
        if not candidate_tokens or not reference_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
        # حساب التداخل
        candidate_set = set(candidate_tokens)
        reference_set = set(reference_tokens)
        
        intersection = candidate_set.intersection(reference_set)
        
        if len(candidate_set) == 0:
            precision = 0.0
        else:
            precision = len(intersection) / len(candidate_set)
        
        if len(reference_set) == 0:
            recall = 0.0
        else:
            recall = len(intersection) / len(reference_set)
        
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
    except Exception as e:
        print(f"خطأ في حساب Precision/Recall/F1: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

def calculate_difficulty(question_text, lang="ar"):
    """تحديد مستوى صعوبة السؤال"""
    try:
        tokens = tokenize_text(question_text, lang)
        word_count = len(tokens)
        
        # معايير بسيطة لتحديد الصعوبة
        if word_count <= 5:
            return "easy"
        elif word_count <= 10:
            return "medium"
        else:
            return "hard"
    except Exception:
        return "medium"

def calculate_all_metrics(question_text, source_text, lang="ar"):
    """حساب جميع المقاييس للسؤال"""
    # التأكد من وجود النصوص
    if not question_text or not source_text:
        return {
            "bleu": 0.0,
            "bert_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "perplexity": 0.0,
            "difficulty": "medium"
        }
    
    metrics = {
        "bleu": 0.0,
        "bert_score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "perplexity": 0.0,
        "difficulty": "medium"
    }
    
    print(f"حساب المقاييس للسؤال: {question_text[:50]}...")
    print(f"النص المصدر: {source_text[:50]}...")
    
    try:
        # حساب BLEU
        try:
            bleu_score = calculate_bleu_score(question_text, source_text, lang)
            metrics["bleu"] = round(bleu_score, 4) if bleu_score is not None else 0.0
            print(f"BLEU: {metrics['bleu']:.4f}")
        except Exception as e:
            print(f"خطأ في حساب BLEU: {e}")
            metrics["bleu"] = 0.0  # التأكد من وجود قيمة افتراضية
        
        # حساب BERTScore
        try:
            bert_scores = calculate_bert_score(question_text, source_text, lang)
            if bert_scores and isinstance(bert_scores, dict) and "f1_score" in bert_scores:
                metrics["bert_score"] = round(bert_scores["f1_score"], 4) if bert_scores["f1_score"] is not None else 0.0
            else:
                metrics["bert_score"] = 0.0
            print(f"BERTScore: {metrics['bert_score']:.4f}")
        except Exception as e:
            print(f"خطأ في حساب BERTScore: {e}")
            metrics["bert_score"] = 0.0  # التأكد من وجود قيمة افتراضية
        
        # حساب Precision, Recall, F1
        try:
            prf_scores = calculate_precision_recall_f1(question_text, source_text, lang)
            if prf_scores and isinstance(prf_scores, dict):
                metrics["precision"] = round(prf_scores.get("precision", 0.0), 4) if prf_scores.get("precision") is not None else 0.0
                metrics["recall"] = round(prf_scores.get("recall", 0.0), 4) if prf_scores.get("recall") is not None else 0.0
                metrics["f1_score"] = round(prf_scores.get("f1_score", 0.0), 4) if prf_scores.get("f1_score") is not None else 0.0
            else:
                metrics["precision"] = 0.0
                metrics["recall"] = 0.0
                metrics["f1_score"] = 0.0
            print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")
        except Exception as e:
            print(f"خطأ في حساب Precision/Recall/F1: {e}")
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1_score"] = 0.0  # التأكد من وجود قيم افتراضية
        
        # حساب Perplexity
        try:
            perplexity_scores = perplexity_list([question_text], lang)
            if perplexity_scores and len(perplexity_scores) > 0 and perplexity_scores[0] is not None:
                metrics["perplexity"] = round(float(perplexity_scores[0]), 4)
            else:
                metrics["perplexity"] = 0.0
            print(f"Perplexity: {metrics['perplexity']:.4f}")
        except Exception as e:
            print(f"خطأ في حساب Perplexity: {e}")
            metrics["perplexity"] = 0.0  # التأكد من وجود قيمة افتراضية
        
        # تحديد الصعوبة
        try:
            difficulty = calculate_difficulty(question_text, lang)
            metrics["difficulty"] = difficulty
            print(f"Difficulty: {difficulty}")
        except Exception as e:
            print(f"خطأ في حساب الصعوبة: {e}")
        
        print(f"النتيجة النهائية: {metrics}")
        return metrics
        
    except Exception as e:
        print(f"خطأ عام في حساب المقاييس: {e}")
        return metrics