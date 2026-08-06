"""حساب المقاييس الآلية: BLEU و BERTScore و Perplexity و F1 وغيرها على الأسئلة."""
import math
import re


def _default_metrics() -> dict:
    """قيم افتراضية لجميع المقاييس."""
    return {
        "bleu": 0.0,
        "bert_score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "perplexity": 0.0,
    }


def perplexity_list(texts, lang):
    """حساب perplexity للنصوص باستخدام نماذج خفيفة."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        model_id = "aubmindlab/araGPT2-base" if lang == "ar" else "gpt2"

        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        ppl = []
        for t in texts:
            if not t or len(t.strip()) < 3:
                ppl.append(0.0)
                continue

            normalized_text = normalize_text_for_perplexity(t, lang)
            if not normalized_text or len(normalized_text.strip()) < 3:
                ppl.append(0.0)
                continue

            try:
                enc = tok(
                    normalized_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                with torch.no_grad():
                    outputs = mdl(**enc, labels=enc["input_ids"])
                    loss = outputs.loss
                    perplexity = math.exp(loss.item())
                    if perplexity >= 1000.0:
                        ppl.append(0.0)
                    else:
                        ppl.append(perplexity)
            except Exception:
                ppl.append(0.0)
        return ppl
    except Exception:
        return [0.0] * len(texts)


def tokenize_arabic(text):
    """تقسيم النص العربي إلى كلمات."""
    text = re.sub(
        r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]",
        " ",
        text,
    )
    return text.split()


def tokenize_english(text):
    """تقسيم النص الإنجليزي إلى كلمات."""
    text = re.sub(r"[^\w\s]", " ", text)
    return [word.lower() for word in text.split()]


def tokenize_text(text, lang="ar"):
    """تقسيم النص حسب اللغة."""
    return tokenize_arabic(text) if lang == "ar" else tokenize_english(text)


def normalize_text_for_perplexity(text, lang="ar"):
    """تطبيع النص لحساب Perplexity."""
    if not text:
        return ""

    if lang == "ar":
        text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
        text = re.sub(r"\b[A-Za-z]+\b", "[اسم]", text)
        text = re.sub(r"\d+", "[رقم]", text)
        text = re.sub(r"\s+", " ", text).strip()
    else:
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

    return text


def calculate_bleu_score(candidate, reference, lang="ar"):
    """حساب BLEU score."""
    del lang
    try:
        from sacrebleu import BLEU

        if not candidate or not reference:
            return 0.0

        bleu = BLEU(effective_order=True)
        score = bleu.sentence_score(candidate, [reference])
        return score.score / 100.0
    except Exception:
        return 0.0


def calculate_bert_score(candidate, reference, lang="ar"):
    """حساب BERTScore."""
    try:
        from bert_score import score as bert_score

        if not candidate or not reference:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        model_type = "bert-base-multilingual-cased" if lang == "ar" else "bert-base-uncased"
        p, r, f1 = bert_score([candidate], [reference], model_type=model_type, verbose=False)
        return {
            "precision": float(p[0]),
            "recall": float(r[0]),
            "f1_score": float(f1[0]),
        }
    except Exception:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}


def calculate_precision_recall_f1(candidate, reference, lang="ar"):
    """حساب Precision, Recall, F1 Score."""
    try:
        if not candidate or not reference:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        candidate_tokens = tokenize_text(candidate, lang)
        reference_tokens = tokenize_text(reference, lang)

        if not candidate_tokens or not reference_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        candidate_set = set(candidate_tokens)
        reference_set = set(reference_tokens)
        intersection = candidate_set.intersection(reference_set)

        precision = len(intersection) / len(candidate_set) if candidate_set else 0.0
        recall = len(intersection) / len(reference_set) if reference_set else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f1_score": f1_score}
    except Exception:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}


def calculate_all_metrics(question_text, source_text, lang="ar"):
    """حساب جميع المقاييس للسؤال."""
    if not question_text or not source_text:
        return _default_metrics()

    bert_scores = calculate_bert_score(question_text, source_text, lang)
    prf_scores = calculate_precision_recall_f1(question_text, source_text, lang)
    perplexity_scores = perplexity_list([question_text], lang)

    return {
        "bleu": round(calculate_bleu_score(question_text, source_text, lang), 4),
        "bert_score": round(bert_scores["f1_score"], 4),
        "precision": round(prf_scores["precision"], 4),
        "recall": round(prf_scores["recall"], 4),
        "f1_score": round(prf_scores["f1_score"], 4),
        "perplexity": round(float(perplexity_scores[0]), 4) if perplexity_scores else 0.0,
    }
