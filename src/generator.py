"""بناء prompts، استدعاء Ollama، التحقق من JSON، وتوليد الأسئلة (Vanilla/RAG)."""
import json
import re
import time
from difflib import SequenceMatcher
from typing import Literal, Optional

import ollama
from langdetect import detect

_JSON_EXAMPLE = """{
  "mcq": [{"q": "...", "options": ["a","b","c","d"], "answer": "a"}],
  "tf": [{"q": "...", "answer": true}]
}"""

# --- تعليمات مشتركة (نوع السؤال + JSON) ---
_RULES_AR = """\
أنواع الأسئلة:
• اختيار من متعدد (MCQ): 4 خيارات، إجابة واحدة صحيحة مطابقة لأحد الخيارات حرفياً.
• صح/خطأ: جملة خبرية أو سؤال يبدأ بـ «هل».

قواعد MCQ:
• كل الخيارات مستقاة من النص المسموح (لا «خيار 1» ولا عبارات عامة فارغة).

قواعد صح/خطأ:
• ممنوع أسئلة WH: من، ماذا، أين، متى، كيف، لماذا، ما.

قواعد عامة:
• حتى 20 سؤالاً، نوّع بين MCQ وصح/خطأ، ولا تكرّر نفس الفكرة.
• أعد JSON صالحاً فقط — بدون markdown ولا شرح خارج JSON.
• المفتاحان الرئيسيان في جذر JSON: "mcq" و "tf" فقط (بدون metadata أو questions)."""

_RULES_EN = """\
Question types:
• MCQ: 4 options, one correct answer matching one option exactly.
• True/False: declarative statement or question starting with Is/Are (or Arabic equivalent).

MCQ rules:
• All options must come from the allowed text (no "option 1" placeholders).

T/F rules:
• No WH-questions (what, where, why, how, when, who, which).

General:
• Up to 20 questions, mix MCQ and T/F, no duplicate ideas.
• Return valid JSON only — no markdown or text outside JSON.
• Top-level keys must be exactly "mcq" and "tf" (no metadata or questions wrapper)."""

_VANILLA_TASK_AR = """\
المهمة: أنشئ أسئلة تعليمية من «النص المرفوع» أدناه فقط.
• المصدر الوحيد المسموح: النص المرفوع.
• لا تستخدم معرفة خارج النص."""

_VANILLA_TASK_EN = """\
Task: Create educational questions from the uploaded text below only.
• The uploaded text is the only allowed source.
• Do not use outside knowledge."""

_RAG_TASK_AR = """\
المهمة: أنشئ أسئلة تعليمية بالاعتماد على مصدرين:

1) النص الأساسي (الملف المرفوع):
   • أساس صياغة الأسئلة وإجاباتها.
   • يجب أن تكون الإجابة الصحيحة قابلة للاستنتاج من هذا النص.

2) المقاطع المسترجعة (RAG):
   • مراجع إضافية من فهرس خارجي.
   • يمكن استخدامها لإثراء خيارات MCQ أو دعم المفهوم.
   • لا تُلغِ النص الأساسي: لا تكتب أسئلة لا علاقة لها بالملف المرفوع."""

_RAG_TASK_EN = """\
Task: Create educational questions using two sources:

1) Main text (uploaded file):
   • Primary basis for questions and correct answers.
   • The correct answer must be inferable from this text.

2) Retrieved passages (RAG):
   • Extra reference chunks from an external index.
   • May enrich MCQ options or support the concept.
   • Do not ignore the main text — no questions unrelated to the upload."""

_GENERIC_OPTION = re.compile(
    r"^(خيار\s*[1-4]|جواب\s*[1-4]|إجابة\s*[أ-د]|option\s*[1-4]|choice\s*[1-4])$",
    re.I,
)
_TF_INTERROGATIVE = (
    "من",
    "أين",
    "ماذا",
    "متى",
    "كيف",
    "لماذا",
    "ما",
    "what",
    "where",
    "why",
    "how",
    "when",
    "who",
    "which",
)

MAX_SOURCE_CHARS = 4500
MAX_RAG_MAIN_CHARS = 2500
MAX_RAG_PASSAGE_CHARS = 350
MAX_RAG_PASSAGES = 3


def _clip_source(text: str, max_chars: int = MAX_SOURCE_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... تم اختصار النص للتوليد ...]"


def detect_lang(text: str) -> Literal["ar", "en"]:
    """اكتشاف لغة النص لاختيار البرومبت."""
    try:
        return "ar" if detect(text) == "ar" else "en"
    except Exception:
        return "ar"


def _rag_passages(retrieved: list[dict], lang: str) -> str:
    if not retrieved:
        return ""
    lines = []
    for i, r in enumerate(retrieved[:MAX_RAG_PASSAGES], 1):
        name = r.get("filename", "?" if lang == "ar" else "source")
        body = (r.get("text") or "")[:MAX_RAG_PASSAGE_CHARS]
        lines.append(f"[{i}: {name}]\n{body}")
    if lang == "ar":
        return "=== مقاطع مسترجعة (RAG) ===\n" + "\n\n".join(lines)
    return "=== Retrieved passages (RAG) ===\n" + "\n\n".join(lines)


def build_prompt_vanilla(text: str, lang: str) -> str:
    """برومبت Vanilla — النص المرفوع فقط."""
    text = _clip_source(text)
    if lang == "ar":
        return (
            "أنت معلّم خبير في صياغة أسئلة تعليمية.\n\n"
            f"{_VANILLA_TASK_AR}\n\n"
            f"{_RULES_AR}\n\n"
            f"=== النص المرفوع (المصدر الوحيد) ===\n{text}\n\n"
            f"=== شكل JSON المطلوب ===\n{_JSON_EXAMPLE}"
        )
    return (
        "You are an expert teacher writing educational questions.\n\n"
        f"{_VANILLA_TASK_EN}\n\n"
        f"{_RULES_EN}\n\n"
        f"=== Uploaded text (only source) ===\n{text}\n\n"
        f"=== Required JSON shape ===\n{_JSON_EXAMPLE}"
    )


def build_prompt_rag(text: str, lang: str, retrieved: list[dict]) -> str:
    """برومبت RAG — نص أساسي + مقاطع مسترجعة."""
    text = _clip_source(text, max_chars=MAX_RAG_MAIN_CHARS)
    rag = _rag_passages(retrieved, lang)
    if lang == "ar":
        parts = [
            "أنت معلّم خبير في صياغة أسئلة تعليمية.\n",
            _RAG_TASK_AR,
            _RULES_AR,
            f"=== النص الأساسي (الملف المرفوع) ===\n{text}",
        ]
        if rag:
            parts.append(rag)
        parts.append(f"=== شكل JSON المطلوب ===\n{_JSON_EXAMPLE}")
        return "\n\n".join(parts)
    parts = [
        "You are an expert teacher writing educational questions.\n",
        _RAG_TASK_EN,
        _RULES_EN,
        f"=== Main text (uploaded file) ===\n{text}",
    ]
    if rag:
        parts.append(rag)
    parts.append(f"=== Required JSON shape ===\n{_JSON_EXAMPLE}")
    return "\n\n".join(parts)


def _extract_chat_response(resp) -> str:
    content = ""
    if hasattr(resp, "message"):
        content = getattr(resp.message, "content", None) or ""
    elif isinstance(resp, dict):
        msg = resp.get("message") or {}
        content = (msg.get("content") if isinstance(msg, dict) else "") or resp.get("content") or ""
    return (content or "").strip()


def check_and_pull_model(model_name: str) -> bool:
    """التأكد من وجود النموذج في Ollama (سحب بسيط عند الحاجة)."""
    try:
        listed = ollama.list()
        names = []
        if listed and hasattr(listed, "models"):
            for m in listed.models:
                if hasattr(m, "model"):
                    names.append(m.model)
        if model_name in names:
            return True
        print(f"Pulling {model_name}...")
        ollama.pull(model_name)
        return True
    except Exception as e:
        print(f"Model check failed: {e}")
        return False


def call_llama(
    prompt: str,
    max_retries: int = 3,
    model_name: str = "llama3.2:3b",
    temperature: float = 0.7,
    json_mode: bool = True,
) -> str:
    """استدعاء Ollama مع إعادة المحاولة."""
    check_and_pull_model(model_name)
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "options": {
                    "num_ctx": 16384,
                    "num_predict": 4096,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                },
            }
            if json_mode:
                kwargs["format"] = "json"
            resp = ollama.chat(**kwargs)
            text = _extract_chat_response(resp)
            if text and (not json_mode or len(text.strip()) >= 30):
                return text
        except Exception as e:
            print(f"Ollama attempt {attempt + 1} failed: {e}")
        if attempt < max_retries - 1:
            time.sleep(3 * (attempt + 1))
    return ""


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def validate_question(question_data: dict, question_type: str, _source_text: str = "") -> tuple[bool, list[str]]:
    """التحقق من MCQ (4 خيارات، لا generic) أو TF (لا أداة استفهام)."""
    errors = []
    if question_type == "mcq":
        options = question_data.get("options", [])
        if len(options) != 4:
            errors.append("options count")
        seen = set()
        unique_opts = 0
        for i, opt in enumerate(options):
            s = str(opt).strip()
            if _GENERIC_OPTION.match(s):
                errors.append(f"generic option {i + 1}")
            key = s.lower()
            if key not in seen:
                unique_opts += 1
            seen.add(key)
        if unique_opts < 2:
            errors.append("too few unique options")
    elif question_type == "tf":
        q = str(question_data.get("q", "")).strip().lower()
        if not q:
            errors.append("empty question")
        for w in _TF_INTERROGATIVE:
            if q.startswith(w.lower()):
                errors.append(f"interrogative: {w}")
                break
    return len(errors) == 0, errors


def remove_duplicate_questions(result: dict) -> dict:
    """إزالة أسئلة متشابهة جداً (>85%)."""
    if "mcq" in result and isinstance(result["mcq"], list):
        unique, seen = [], []
        for mcq in result["mcq"]:
            if not isinstance(mcq, dict):
                continue
            q = str(mcq.get("q", "")).strip()
            if any(_similarity(q, s) > 0.85 for s in seen):
                continue
            seen.append(q.lower())
            unique.append(mcq)
        result["mcq"] = unique
    if "tf" in result and isinstance(result["tf"], list):
        unique, seen = [], []
        for tf in result["tf"]:
            if not isinstance(tf, dict):
                continue
            q = str(tf.get("q", "")).strip()
            if any(_similarity(q, s) > 0.85 for s in seen):
                continue
            seen.append(q.lower())
            unique.append(tf)
        result["tf"] = unique
    return result


def postprocess_questions(result: dict) -> dict:
    """إزالة التكرار وحذف الأسئلة غير الصالحة """
    result = remove_duplicate_questions(result)
    if isinstance(result.get("mcq"), list):
        result["mcq"] = [
            q
            for q in result["mcq"]
            if isinstance(q, dict) and validate_question(q, "mcq")[0]
        ]
    if isinstance(result.get("tf"), list):
        result["tf"] = [
            q
            for q in result["tf"]
            if isinstance(q, dict) and validate_question(q, "tf")[0]
        ]
    return result


def _ensure_mcq_options(result: dict) -> None:
    for mcq in result.get("mcq") or []:
        if not isinstance(mcq, dict):
            continue
        opts = mcq.get("options")
        if not isinstance(opts, list):
            mcq["options"] = []
            opts = mcq["options"]
        while len(opts) < 4:
            opts.append(f"— {len(opts) + 1}")
        mcq["options"] = opts[:4]


def _repair_json_text(s: str) -> str:
    s = re.sub(r"```json|```", "", s, flags=re.I).strip()
    if s.count("{") > s.count("}"):
        s += "}" * (s.count("{") - s.count("}"))
    if s.count("[") > s.count("]"):
        s += "]" * (s.count("[") - s.count("]"))
    lines = s.split("\n")
    if lines and not lines[-1].strip().endswith(("}", "]", ",")):
        lines = lines[:-1]
        s = "\n".join(lines)
        if s.count("{") > s.count("}"):
            s += "\n}"
        if s.count("[") > s.count("]"):
            s += "\n]"
    return s.strip()


def _extract_json_blob(raw: str) -> str:
    """استخراج كائن JSON من نص قد يحتوي markdown أو شرحاً قبل/بعد."""
    s = re.sub(r"```json\s*|```", "", raw, flags=re.I).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


def _loads_questions(raw: str) -> Optional[dict]:
    for candidate in (raw, _extract_json_blob(raw), _repair_json_text(_extract_json_blob(raw))):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _normalize_question_item(item: dict, qtype: str) -> Optional[dict]:
    if not isinstance(item, dict):
        return None
    q = (item.get("q") or item.get("question") or "").strip()
    if not q:
        return None
    out = dict(item)
    out["q"] = q
    if qtype == "mcq":
        opts = out.get("options")
        if not isinstance(opts, list):
            opts = out.get("choices") or out.get("answers") or []
        out["options"] = [str(o).strip() for o in opts if str(o).strip()]
        ans = out.get("answer") or out.get("correct_answer") or out.get("correct")
        if ans is not None:
            out["answer"] = str(ans).strip()
    else:
        ans = out.get("answer")
        if isinstance(ans, str):
            low = ans.strip().lower()
            if low in ("true", "صح", "صحيح", "yes", "نعم"):
                out["answer"] = True
            elif low in ("false", "خطأ", "خطا", "no", "لا"):
                out["answer"] = False
    return out


def _coerce_question_payload(result: dict) -> dict:
    """توحيد أشكال JSON مختلفة (questions/metadata/أسماء حقول بديلة)."""
    nested = result.get("questions")
    if isinstance(nested, dict):
        for key in ("mcq", "tf", "true_false", "true/false"):
            if key in nested and key not in result:
                result[key] = nested[key]
    if isinstance(nested, list):
        mcq, tf = [], []
        for item in nested:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("type", "")).lower()
            if kind in ("mcq", "multiple_choice", "choice"):
                mcq.append(item)
            elif kind in ("tf", "true_false", "boolean"):
                tf.append(item)
            elif item.get("options"):
                mcq.append(item)
            else:
                tf.append(item)
        result.setdefault("mcq", mcq)
        result.setdefault("tf", tf)

    for alt_tf in ("true_false", "true/false", "trueFalse"):
        if alt_tf in result and "tf" not in result:
            result["tf"] = result.pop(alt_tf)

    mcq = [
        norm
        for item in (result.get("mcq") or [])
        if (norm := _normalize_question_item(item, "mcq"))
    ]
    tf = [
        norm
        for item in (result.get("tf") or [])
        if (norm := _normalize_question_item(item, "tf"))
    ]
    result["mcq"] = mcq
    result["tf"] = tf
    return _normalize_question_payload(result)


def _normalize_question_payload(result: dict) -> dict:
    if not isinstance(result.get("mcq"), list):
        result["mcq"] = []
    if not isinstance(result.get("tf"), list):
        result["tf"] = []
    return result


def safe_json(
    s: str,
    source_text: str = "",
    model_name: str = "llama3.2:3b",
    lang: str = "ar",
    retrieved: Optional[list] = None,
):
    """تحليل JSON من استجابة النموذج مع معالجة لاحقة."""
    del source_text, model_name, lang, retrieved  # kept for call-site compatibility
    if not s or not s.strip():
        return None
    result = _loads_questions(s)
    if not isinstance(result, dict):
        return None
    result = _coerce_question_payload(result)
    _ensure_mcq_options(result)
    result = postprocess_questions(result)
    if not (result.get("mcq") or result.get("tf")):
        return None
    return result


def generate_questions_with_retry(
    prompt: str,
    max_retries: int = 3,
    source_text: str = "",
    model_name: str = "llama3.2:3b",
    lang: str = "ar",
    retrieved: Optional[list] = None,
):
    """توليد الأسئلة مع إعادة المحاولة حتى JSON صالح."""
    for attempt in range(max_retries):
        response = call_llama(prompt, model_name=model_name, max_retries=2, json_mode=True)
        if not response:
            continue
        questions = safe_json(response, source_text, model_name, lang, retrieved)
        if (
            isinstance(questions, dict)
            and "mcq" in questions
            and "tf" in questions
            and (questions.get("mcq") or questions.get("tf"))
        ):
            return questions
        time.sleep(3)
    return None
