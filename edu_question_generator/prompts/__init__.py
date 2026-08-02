from __future__ import annotations

from ._shared import Difficulty
from .deepseek import build_deepseek_prompt

SYSTEM_MESSAGES: dict[str, dict[str, str]] = {
    "deepseek": {
        "en": (
            "University final exam MCQs in Arabic: mix analysis/application (why, what-if) "
            "with multi-step computation (understand then calculate). "
            "No mechanical subtraction-only items; confident short solutions. JSON only."
        ),
        "ar": (
            "أنت عضو هيئة تدريس تعد امتحاناً نهائياً. "
            "مزيج: تحليل/تطبيق (لماذا، ماذا لو) + حساب multi-step (فهم ثم حساب) — "
            "لا طرح ميكانيكي ولا solution متردد. JSON فقط."
        ),
    },
}


def build_prompt(
    context: str,
    lang: str,
    difficulty: Difficulty,
    types: list[str],
    num_questions: int | None,
    provider: str,
    *,
    math_focus: bool = False,
    dl_focus: bool = False,
) -> str:
    del lang, types, provider, math_focus, dl_focus
    return build_deepseek_prompt(context, difficulty, num_questions)


def build_system_message(lang: str, provider: str) -> str:
    del provider
    messages = SYSTEM_MESSAGES["deepseek"]
    return messages["ar" if lang == "ar" else "en"]
