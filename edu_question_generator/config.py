import os
from typing import Optional


def _detect_default_provider() -> str:
    return os.getenv("LLM_PROVIDER", "deepseek")


DEFAULT_PROVIDER = _detect_default_provider()

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# deepseek-reasoner = DeepSeek R1 | deepseek-chat = V3 (أسرع، بدون chain-of-thought)
DEFAULT_DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_REASONER_MAX_TOKENS = int(os.getenv("DEEPSEEK_REASONER_MAX_TOKENS", "8192"))


def deepseek_model_display_name(model_id: Optional[str] = None) -> str:
    model = model_id or DEFAULT_DEEPSEEK_MODEL
    if "reasoner" in model.lower():
        return "DeepSeek R1"
    if "chat" in model.lower():
        return "DeepSeek Chat (V3)"
    return model


CHUNKED_PIPELINE_PROVIDERS = ("deepseek",)
JSON_MODE_PROVIDERS = frozenset({"deepseek"})
LLM_MAX_COMPLETION_TOKENS = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "8192"))

CHUNK_SIZE = 650
CHUNK_OVERLAP = 100

# Universal document pipeline (all providers/models):
# 1) extract text  2) chunk to ~2500-3500 chars  3) prompt+chunk per request
# 4) merge questions  5) dedupe  6) validate against full source
SEGMENT_MIN_CHARS = 2500
SEGMENT_MAX_CHARS_LIMIT = 3500
SEGMENT_MAX_CHARS = max(
    SEGMENT_MIN_CHARS,
    min(int(os.getenv("SEGMENT_MAX_CHARS", "3000")), SEGMENT_MAX_CHARS_LIMIT),
)
MAX_SEGMENTS_PER_RUN = int(os.getenv("MAX_SEGMENTS_PER_RUN", "20"))

# Exam pipeline defaults (large documents: few logical sections, fixed question budget)
TARGET_QUESTIONS_TOTAL = int(os.getenv("TARGET_QUESTIONS_TOTAL", "20"))
TARGET_COMPUTATION_MIN = int(os.getenv("TARGET_COMPUTATION_MIN", "6"))
TARGET_ANALYSIS_APPLICATION_MIN = int(os.getenv("TARGET_ANALYSIS_APPLICATION_MIN", "8"))
MAX_LOGICAL_SEGMENTS = int(os.getenv("MAX_LOGICAL_SEGMENTS", "10"))
LOGICAL_SEGMENT_MAX_CHARS = int(os.getenv("LOGICAL_SEGMENT_MAX_CHARS", "14000"))

LLM_REQUEST_TOO_LARGE = "LLM_REQUEST_TOO_LARGE"
LLM_LIMIT_ERROR = "LLM_LIMIT_ERROR"
LLM_INSUFFICIENT_BALANCE = "LLM_INSUFFICIENT_BALANCE"
PIPELINE_ALL_SEGMENTS_FAILED = "PIPELINE_ALL_SEGMENTS_FAILED"

MAX_CHUNK_GROUPS = MAX_SEGMENTS_PER_RUN
