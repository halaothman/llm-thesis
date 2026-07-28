"""بناء prompts، استدعاء Ollama، التحقق من JSON، وتوليد الأسئلة (Vanilla/RAG)."""
import json
import re
import time
from difflib import SequenceMatcher
from typing import Literal, Optional

import ollama
from langdetect import detect

SYS_AR = (
    "أنت معلّم خبير. أنشئ حتى 20 سؤالاً (MCQ بـ4 خيارات + صح/خطأ) من النص فقط. "
    "خيارات MCQ إجابات حقيقية من النص — ممنوع «خيار 1» أو خيارات عامة. "
    "صح/خطأ: جملة خبرية أو «هل» — بدون من/أين/ماذا/لماذا/كيف/متى/ما. "
    "JSON فقط."
)
SYS_EN = (
    "Expert teacher: up to 20 questions (MCQ with 4 options + T/F) from the text only. "
    "Real MCQ options from text — no 'option 1' placeholders. "
    "T/F: declarative or Is/Are — no what/where/why/how/when/who. JSON only."
)

_JSON_EXAMPLE = """{
  "mcq": [{"q": "...", "options": ["a","b","c","d"], "answer": "a"}],
  "tf": [{"q": "...", "answer": true}]
}"""

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
    for i, r in enumerate(retrieved, 1):
        name = r.get("filename", "?" if lang == "ar" else "source")
        lines.append(f"[{i}: {name}]\n{r['text'][:500]}")
    if lang == "ar":
        return "=== مقاطع مسترجعة (RAG) ===\n" + "\n\n".join(lines)
    return "=== Retrieved passages (RAG) ===\n" + "\n\n".join(lines)


def build_prompt_vanilla(text: str, lang: str) -> str:
    """برومبت Vanilla — النص المرفوع فقط."""
    head = SYS_AR if lang == "ar" else SYS_EN
    if lang == "ar":
        body = f"=== النص (المصدر الوحيد) ===\n{text}\n\n"
        rules = (
            "استخدم هذا النص فقط. نوّع بين MCQ وصح/خطأ. لا تكرر الأسئلة.\n"
            f"مثال JSON:\n{_JSON_EXAMPLE}"
        )
    else:
        body = f"=== Text (only source) ===\n{text}\n\n"
        rules = f"Use this text only. Mix MCQ and T/F. No duplicate questions.\nExample:\n{_JSON_EXAMPLE}"
    return f"{head}\n\n{body}{rules}"


def build_prompt_rag(text: str, lang: str, retrieved: list[dict]) -> str:
    """برومبت RAG — نص أساسي + مقاطع مسترجعة."""
    head = SYS_AR if lang == "ar" else SYS_EN
    rag = _rag_passages(retrieved, lang)
    if lang == "ar":
        body = f"=== النص الأساسي ===\n{text}\n\n{rag}\n\n" if rag else f"=== النص ===\n{text}\n\n"
        rules = (
            "أسئلة من النص الأساسي مع إمكانية إثراء الخيارات من مقاطع RAG.\n"
            f"مثال JSON:\n{_JSON_EXAMPLE}"
        )
    else:
        body = f"=== Main text ===\n{text}\n\n{rag}\n\n" if rag else f"=== Text ===\n{text}\n\n"
        rules = (
            "Questions grounded in main text; options may use RAG passages.\n"
            f"Example:\n{_JSON_EXAMPLE}"
        )
    return f"{head}\n\n{body}{rules}"


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
) -> str:
    """استدعاء Ollama مع إعادة المحاولة."""
    check_and_pull_model(model_name)
    for attempt in range(max_retries):
        try:
            resp = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_ctx": 16384,
                    "num_predict": 4096,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "stop": ["```", "---", "==="],
                },
            )
            text = _extract_chat_response(resp)
            if text:
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
        for i, opt in enumerate(options):
            s = str(opt).strip()
            if _GENERIC_OPTION.match(s):
                errors.append(f"generic option {i + 1}")
            key = s.lower()
            if key in seen:
                errors.append("duplicate options")
            seen.add(key)
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
    """إزالة التكرار وحذف الأسئلة غير الصالحة (بدون إعادة استدعاء LLM)."""
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


def _loads_questions(raw: str) -> Optional[dict]:
    for candidate in (raw, _repair_json_text(raw)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


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
    result = postprocess_questions(result)
    _ensure_mcq_options(result)
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
        response = call_llama(prompt, model_name=model_name, max_retries=1)
        if not response:
            continue
        questions = safe_json(response, source_text, model_name, lang, retrieved)
        if isinstance(questions, dict) and "mcq" in questions and "tf" in questions:
            return questions
        time.sleep(3)
    return None
