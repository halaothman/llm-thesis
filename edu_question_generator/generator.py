"""توليد أسئلة MCQ عبر DeepSeek: prompts، تحليل الرد، تطبيع، وتنظيف."""
from __future__ import annotations

import json
import re
from typing import Literal

from langdetect import detect

from .config import DEEPSEEK_MODEL
from .llm_client import chat_complete
from .prompts.deepseek import build_deepseek_prompt, build_deepseek_system_message

Lang = Literal["ar", "en"]
Difficulty = Literal["Easy", "Medium", "Hard"]

# رموز غير مسموحة في نص السؤال (CJK، Cyrillic، …)
FORBIDDEN_CHARS = re.compile(
    "["
    "\u4e00-\u9fff"  # CJK Unified Ideographs
    "\u3400-\u4dbf"  # CJK Extension A
    "\u3040-\u30ff"  # Hiragana + Katakana
    "\u31f0-\u31ff"  # Katakana extensions
    "\uac00-\ud7af"  # Hangul syllables
    "\u1100-\u11ff"  # Hangul jamo
    "\u0400-\u04ff"  # Cyrillic
    "\u0900-\u097f"  # Devanagari
    "\u0980-\u09ff"  # Bengali
    "\u0e00-\u0e7f"  # Thai
    "]+"
)


def detect_lang(text: str) -> Lang:
    """كشف لغة المستند (ar/en)."""
    try:
        return "ar" if detect(text) == "ar" else "en"
    except Exception:
        return "en"


def sanitize_text(text: str) -> str:
    """إزالة رموز CJK/Cyrillic وغيرها من نص السؤال."""
    if not text:
        return text
    cleaned = FORBIDDEN_CHARS.sub("", str(text))
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def sanitize_payload(payload: dict) -> dict:
    """تنظيف حقول MCQ بعد التوليد."""
    for item in payload.get("mcq", []):
        item["q"] = sanitize_text(item.get("q", ""))
        item["solution"] = sanitize_text(item.get("solution") or item.get("explanation", ""))
        item["explanation"] = item["solution"]
        if item.get("question_kind"):
            item["question_kind"] = sanitize_text(str(item["question_kind"]))
        item["options"] = [sanitize_text(option) for option in item.get("options", [])]
        if not isinstance(item.get("answer"), bool):
            item["answer"] = sanitize_text(str(item.get("answer", "")))
    return payload


def _mcq_item_from_raw(item: dict, default_difficulty: str) -> dict:
    """تحويل عنصر MCQ خام من JSON النموذج إلى الشكل الداخلي."""
    options = _resolve_mcq_options(item.get("options", []))
    answer = _resolve_mcq_answer(options, item.get("correct_answer", item.get("answer", "")))
    question_kind = item.get("type", item.get("question_kind", ""))
    if str(question_kind).lower() in {"mcq", "true_false", "tf", "short"}:
        question_kind = item.get("question_kind", "")
    return {
        "q": item.get("question") or item.get("q", ""),
        "options": options,
        "answer": answer,
        "solution": item.get("solution") or item.get("explanation", ""),
        "question_kind": question_kind,
        "difficulty": item.get("difficulty", default_difficulty),
    }


def _strip_model_artifacts(text: str) -> str:
    """إزالة think/redacted_thinking من رد النموذج."""
    cleaned = str(text or "")
    think_open = "<" + "think" + ">"
    think_close = "</" + "think" + ">"
    cleaned = re.sub(
        rf"{re.escape(think_open)}.*?{re.escape(think_close)}",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
    return cleaned.strip()


def _extract_json_object(text: str) -> str:
    """استخراج {…} من markdown أو نص محيط."""
    text = _strip_model_artifacts(text)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _fix_invalid_json_escapes(text: str) -> str:
    """إصلاح escape غير صالح في JSON."""
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def _remove_trailing_commas(text: str) -> str:
    """حذف فاصلة زائدة قبل } أو ]."""
    return re.sub(r",(\s*[}\]])", r"\1", text)


def safe_json(raw: str) -> dict:
    """تحليل JSON من رد LLM مع محاولات إصلاح."""
    text = _extract_json_object(raw)
    candidates = [
        text,
        _fix_invalid_json_escapes(text),
        _remove_trailing_commas(text),
        _remove_trailing_commas(_fix_invalid_json_escapes(text)),
    ]
    last_error: json.JSONDecodeError | None = None
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON object found", raw, 0)


def _resolve_mcq_options(options: object) -> list[str]:
    """توحيد الخيارات (قائمة أو dict A-D)."""
    if isinstance(options, dict):
        return [str(options.get(key, "")) for key in ("A", "B", "C", "D")]
    if isinstance(options, list):
        return [str(option) for option in options[:4]]
    return []


def _resolve_mcq_answer(options: list[str], answer: object) -> str:
    """تحويل حرف A-D إلى نص الخيار إن لزم."""
    if isinstance(answer, str) and len(answer) == 1 and answer.upper() in "ABCD":
        index = "ABCD".index(answer.upper())
        if index < len(options):
            return options[index]
    return str(answer) if answer is not None else ""


def normalize_payload(raw: dict, default_difficulty: str = "Hard") -> dict:
    """توحيد JSON النموذج إلى قائمة mcq."""
    mcq: list[dict] = []

    if isinstance(raw.get("mcq"), list):
        for item in raw["mcq"]:
            mcq.append(_mcq_item_from_raw(item, default_difficulty))
        return {"mcq": mcq}

    for item in raw.get("questions", []):
        question_type = str(item.get("type", "")).lower().replace("-", "_").replace(" ", "_")
        if question_type in {"mcq", "analytical", "computational", "analysis", "computation", "application", "understanding"}:
            mcq.append(_mcq_item_from_raw(item, default_difficulty))

    return {"mcq": mcq}


def generate_questions(
    context: str,
    lang: Lang,
    difficulty: Difficulty,
    num_questions: int | None = None,
    model: str = "",
    api_key: str | None = None,
) -> dict:
    """استدعاء DeepSeek لتوليد MCQ من مقطع واحد (مع إعادة محاولة JSON)."""
    prompt = build_deepseek_prompt(context, difficulty, num_questions)
    system = build_deepseek_system_message(lang)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    resolved_model = model or DEEPSEEK_MODEL
    last_error: json.JSONDecodeError | None = None
    for attempt in range(2):
        try:
            content = chat_complete(resolved_model, messages, api_key=api_key)
            return sanitize_payload(
                normalize_payload(safe_json(content), default_difficulty=difficulty)
            )
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt == 0:
                continue
            raise
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON object found", "", 0)
