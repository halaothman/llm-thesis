"""إعدادات Edu Question Generator: DeepSeek API، تقسيم المستند، وأهداف عدد الأسئلة."""
import os

# --- اتصال DeepSeek API ---
# النموذج الفعلي يُقرأ من .streamlit/secrets.toml (DEEPSEEK_MODEL) عبر ui.get_deepseek_model()
# النماذج الشائعة: deepseek-chat | deepseek-reasoner | deepseek-v4-pro | deepseek-v4-flash
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "8192"))

# aliases للتوافق مع imports قديمة
DEEPSEEK_R1_MODEL = DEEPSEEK_MODEL
DEEPSEEK_R1_MAX_TOKENS = DEEPSEEK_MAX_TOKENS
# --- تقسيم حرفي ثابت (مسار legacy في build_segments) ---
CHUNK_SIZE = 650
CHUNK_OVERLAP = 100

# حجم المقطع المنطقي عند دمج القطع الصغيرة (~2500–3500 حرف)
SEGMENT_MIN_CHARS = 2500
SEGMENT_MAX_CHARS_LIMIT = 3500
SEGMENT_MAX_CHARS = max(
    SEGMENT_MIN_CHARS,
    min(int(os.getenv("SEGMENT_MAX_CHARS", "3000")), SEGMENT_MAX_CHARS_LIMIT),
)
MAX_SEGMENTS_PER_RUN = int(os.getenv("MAX_SEGMENTS_PER_RUN", "20"))

# --- أهداف pipeline الامتحان (مستندات كبيرة) ---
TARGET_QUESTIONS_TOTAL = int(os.getenv("TARGET_QUESTIONS_TOTAL", "20"))
TARGET_COMPUTATION_MIN = int(os.getenv("TARGET_COMPUTATION_MIN", "6"))
TARGET_ANALYSIS_APPLICATION_MIN = int(os.getenv("TARGET_ANALYSIS_APPLICATION_MIN", "8"))
MAX_LOGICAL_SEGMENTS = int(os.getenv("MAX_LOGICAL_SEGMENTS", "10"))
LOGICAL_SEGMENT_MAX_CHARS = int(os.getenv("LOGICAL_SEGMENT_MAX_CHARS", "14000"))

# رموز أخطاء موحّدة يُرمى بها من llm_client / pipeline
LLM_REQUEST_TOO_LARGE = "LLM_REQUEST_TOO_LARGE"
LLM_LIMIT_ERROR = "LLM_LIMIT_ERROR"
LLM_INSUFFICIENT_BALANCE = "LLM_INSUFFICIENT_BALANCE"
PIPELINE_ALL_SEGMENTS_FAILED = "PIPELINE_ALL_SEGMENTS_FAILED"
LLM_INVALID_MODEL = "LLM_INVALID_MODEL"  # اسم نموذج غير مدعوم (HTTP 400)
