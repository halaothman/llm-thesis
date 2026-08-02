"""اختيار وتنويع أسئلة MCQ: توزيع، ترجيح، وحد أقصى للعدد."""
from __future__ import annotations

import re

# أنواع MCQ حسب حقل type من JSON النموذج
_COMPUTATION_TYPES = frozenset(
    {"computation", "computational", "حساب", "حسابي", "calculation", "numeric"}
)

_ANALYSIS_APPLICATION_TYPES = frozenset(
    {"analysis", "application", "تحليل", "تطبيق", "analytical", "applied"}
)


def distribute_question_counts(num_questions: int, segment_count: int) -> list[int]:
    """توزيع ميزانية الأسئلة على مقاطع المستند (باقي → المقاطع الأولى)."""
    if segment_count <= 0:
        return []
    if num_questions <= 0:
        return [0] * segment_count
    if num_questions <= segment_count:
        return [1 if index < num_questions else 0 for index in range(segment_count)]
    base = num_questions // segment_count
    remainder = num_questions % segment_count
    return [base + (1 if index < remainder else 0) for index in range(segment_count)]


def _question_type(item: dict) -> str:
    """نوع السؤال من حقل type أو question_kind."""
    raw = str(item.get("question_kind") or item.get("type") or "").strip().lower()
    return raw.replace("-", "_").replace(" ", "_")


def _is_computation(item: dict) -> bool:
    """هل السؤال من نوع computation حسب حقل type؟"""
    return _question_type(item) in _COMPUTATION_TYPES


def _is_analysis_or_application(item: dict) -> bool:
    """هل السؤال تحليل أو تطبيق حسب حقل type؟"""
    return _question_type(item) in _ANALYSIS_APPLICATION_TYPES


def _concept_key(item: dict) -> str:
    """مفتاح مفهوم لتجنب تكرار نفس الفكرة عبر مقاطع مختلفة."""
    q = re.sub(r"\s+", " ", str(item.get("q", "")).strip().lower())
    if re.search(r"embedding|تضمين", q) and re.search(r"parameter|معامل|params", q):
        return "concept:embedding_params"
    if re.search(r"max_words?|max_word|أقصى\s+كلم", q):
        return "concept:max_words"
    if re.search(r"padding|حشو|pad", q):
        return "concept:padding"
    if re.search(r"lstm|gru|rnn", q) and re.search(r"parameter|معامل", q):
        return "concept:rnn_params"
    return f"q:{q[:100]}"


def _difficulty_rank(item: dict) -> int:
    """hard=2, medium=1, else=0."""
    diff = str(item.get("difficulty", "")).strip().lower()
    if diff == "hard":
        return 2
    if diff in {"medium", "med"}:
        return 1
    return 0


def _quality_score(item: dict) -> float:
    """ترتيب بسيط: صعوبة ثم نوع (من JSON بعد validator)."""
    score = float(_difficulty_rank(item) * 3)
    kind = _question_type(item)
    if kind in _ANALYSIS_APPLICATION_TYPES:
        score += 5.0
    elif kind in _COMPUTATION_TYPES:
        score += 3.0
    elif kind == "understanding":
        score += 1.0
    options = item.get("options") or []
    if isinstance(options, list) and len(options) >= 4:
        score += 0.5
    return score


def cap_and_diversify_mcq(
    items: list[dict],
    *,
    max_total: int,
    min_computation: int,
    min_analysis_application: int,
) -> list[dict]:
    """اختيار أفضل MCQ مع حد أدنى لتحليل/حساب وتنويع المفاهيم."""
    if max_total <= 0 or not items:
        return []

    ranked = sorted(items, key=_quality_score, reverse=True)
    selected: list[dict] = []
    seen_questions: set[str] = set()
    seen_concepts: set[str] = set()

    def try_add(item: dict) -> bool:
        if len(selected) >= max_total:
            return False
        qkey = re.sub(r"\s+", " ", str(item.get("q", "")).strip().lower())
        if not qkey or qkey in seen_questions:
            return False
        ckey = _concept_key(item)
        if ckey.startswith("concept:") and ckey in seen_concepts:
            return False
        seen_questions.add(qkey)
        if ckey.startswith("concept:"):
            seen_concepts.add(ckey)
        selected.append(item)
        return True

    for item in ranked:
        if sum(1 for x in selected if _is_analysis_or_application(x)) >= min_analysis_application:
            break
        if _is_analysis_or_application(item):
            try_add(item)

    for item in ranked:
        if sum(1 for x in selected if _is_computation(x)) >= min_computation:
            break
        if _is_computation(item):
            try_add(item)

    for item in ranked:
        try_add(item)

    return selected[:max_total]
