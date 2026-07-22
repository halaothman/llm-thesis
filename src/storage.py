import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

PATH = "outputs/questions.json"

def save_group(group: str, payload: dict):
    """
    حفظ مجموعة أسئلة في ملف JSON
    
    Args:
        group: 'A' (Vanilla) | 'B' (RAG)
        payload: {"lang":..., "mcq":[...], "tf":[...], "sources": optional}
    """
    # إنشاء مجلد outputs إذا لم يكن موجوداً
    os.makedirs("outputs", exist_ok=True)
    
    # تحميل البيانات الموجودة
    data = {}
    if os.path.exists(PATH):
        try:
            with open(PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {}
    
    # إضافة البيانات الجديدة
    if group not in data:
        data[group] = []
    
    data[group].append(payload)
    
    # حفظ البيانات
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_questions_with_metrics(questions_data: Dict[str, Any], model_name: str, method: str, source_file: str, lang: str = "ar"):
    """
    حفظ الأسئلة في مجلد outputs/<اسم_الملف>/ (نفس سلوك التوليد التلقائي).
    """
    filename, count = save_questions_separate_file(
        questions_data=questions_data,
        model_name=model_name,
        method=method,
        source_file=source_file,
        lang=lang,
    )
    print(f"✅ save_questions_with_metrics -> {filename} ({count} questions)")
    return count

def load_all():
    """
    تحميل جميع البيانات المحفوظة من الملف الرئيسي
    
    Returns:
        dict: البيانات المحفوظة
    """
    if os.path.exists(PATH):
        try:
            with open(PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

def save_questions_separate_file(questions_data: Dict[str, Any], model_name: str, method: str, source_file: str, lang: str = "ar"):
    """
    حفظ الأسئلة في ملف منفصل حسب (نموذج + طريقة + ملف مرفوع)
    
    Args:
        questions_data: البيانات المحتوية على الأسئلة
        model_name: اسم النموذج المستخدم
        method: طريقة التوليد (vanilla, rag)
        source_file: اسم الملف المصدر
        lang: اللغة
    """
    from .evals import calculate_all_metrics
    from pathlib import Path
    
    # إنشاء مجلد outputs إذا لم يكن موجوداً
    os.makedirs("outputs", exist_ok=True)
    
    # تحديد اسم النموذج المختصر
    model_short = (
        "llama" if "llama" in model_name.lower()
        else "deepseek" if "deepseek" in model_name.lower()
        else "qwen"
    )
    if method.lower() == "vanilla":
        method_short = "vanilla"
    elif method.lower() == "math":
        method_short = "math"
    else:
        method_short = "rag"
    
    # طباعة معلومات التشخيص
    print(f"🔍 معلومات الحفظ:")
    print(f"  - model_name: {model_name}")
    print(f"  - model_short: {model_short}")
    print(f"  - method: {method}")
    print(f"  - method_short: {method_short}")
    print(f"  - source_file: {source_file}")
    
    # تنظيف اسم الملف المصدر
    clean_source = Path(source_file).stem.replace(" ", "_").replace(".", "_")
    print(f"  - clean_source: {clean_source}")
    
    # إنشاء مجلد للملف المصدر إذا لم يكن موجوداً
    source_folder = clean_source
    source_folder_path = os.path.join("outputs", source_folder)
    os.makedirs(source_folder_path, exist_ok=True)
    
    # اسم الملف النهائي (يُحفظ داخل مجلد الملف المصدر)
    # إضافة "new" لتجنب استبدال الملفات القديمة
    filename = f"questions_{model_short}_{method_short}_{clean_source}_new.json"
    file_path = os.path.join(source_folder_path, filename)
    
    print(f"📁 سيتم حفظ الملف في: {file_path}")
    print(f"📝 اسم الملف: {filename}")
    print(f"📂 المجلد: {source_folder_path}")
    
    # إنشاء الهيكل
    data = {
        "metadata": {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "source_file": source_file,
            "total_questions": 0,
            "model": model_name,
            "method": method,
            "created_from": "user_upload"
        },
        "questions": []
    }
    
    question_id = 1
    
    # معالجة أسئلة الاختيار من متعدد
    for mcq in questions_data.get("mcq", []):
        question_text = mcq.get("q", mcq.get("question", ""))
        if not question_text:
            continue
        
        source_text = questions_data.get("source_text", "")
        
        # حساب المقاييس
        try:
            metrics = calculate_all_metrics(question_text, source_text, lang)
        except Exception as e:
            print(f"خطأ في حساب المقاييس: {e}")
            metrics = {
                "perplexity": 0.0,
                "bleu": 0.0,
                "bert_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "difficulty": "medium"
            }
        
        # إنشاء كائن السؤال
        question_obj = {
            "id": f"q_{question_id:03d}",
            "type": "mcq",
            "question": question_text,
            "options": mcq.get("options", []),
            "correct_answer": mcq.get("answer", ""),
            "source": {
                "file": source_file,
                "page": 1,
                "passage": source_text[:200] + "..." if len(source_text) > 200 else source_text
            },
            "generation": {
                "model": model_name,
                "method": method,
                "timestamp": datetime.now().isoformat(),
                "generation_time": questions_data.get("generation_time", 0.0)
            },
            "metrics": metrics
        }
        
        data["questions"].append(question_obj)
        question_id += 1
    
    # معالجة أسئلة الصح/الخطأ
    for tf in questions_data.get("tf", []):
        question_text = tf.get("q", tf.get("question", ""))
        if not question_text:
            continue
        
        source_text = questions_data.get("source_text", "")
        
        # حساب المقاييس
        try:
            metrics = calculate_all_metrics(question_text, source_text, lang)
        except Exception as e:
            print(f"خطأ في حساب المقاييس: {e}")
            metrics = {
                "perplexity": 0.0,
                "bleu": 0.0,
                "bert_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "difficulty": "medium"
            }
        
        # إنشاء كائن السؤال
        question_obj = {
            "id": f"q_{question_id:03d}",
            "type": "tf",
            "question": question_text,
            "correct_answer": tf.get("answer", False),
            "source": {
                "file": source_file,
                "page": 1,
                "passage": source_text[:200] + "..." if len(source_text) > 200 else source_text
            },
            "generation": {
                "model": model_name,
                "method": method,
                "timestamp": datetime.now().isoformat(),
                "generation_time": questions_data.get("generation_time", 0.0)
            },
            "metrics": metrics
        }
        
        data["questions"].append(question_obj)
        question_id += 1
    
    # التحقق من وجود أسئلة
    if len(data["questions"]) == 0:
        raise ValueError("لا توجد أسئلة للحفظ")
    
    # تحديث العدد الإجمالي
    data["metadata"]["total_questions"] = len(data["questions"])
    
    # حفظ البيانات
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # التحقق من أن الملف تم حفظه بالفعل
        if not os.path.exists(file_path):
            raise IOError(f"فشل في حفظ الملف: {file_path}")
        
        print(f"✅ تم حفظ {len(data['questions'])} سؤال في ملف: {filename}")
        print(f"📂 المسار الكامل: {os.path.abspath(file_path)}")
        return filename, len(data["questions"])
    except Exception as e:
        print(f"❌ خطأ في حفظ الملف: {e}")
        raise



def _normalize_math_mcq(mcq_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """توحيد حقول أسئلة MCQ الرياضية قبل الحفظ."""
    normalized = []
    for i, mcq in enumerate(mcq_list[:5]):
        if not isinstance(mcq, dict):
            continue
        question = mcq.get("q", mcq.get("question", "")).strip()
        if not question:
            continue
        q_type = mcq.get("type") or ("computational" if i < 3 else "analytical")
        normalized.append({
            "q": question,
            "options": mcq.get("options", []),
            "answer": mcq.get("answer", ""),
            "solution": mcq.get("solution", ""),
            "type": q_type,
            "difficulty": mcq.get("difficulty", "hard"),
        })
    return normalized


def save_math_questions_file(questions_data: Dict[str, Any], source_file: str) -> tuple:
    """
    حفظ الأسئلة الرياضية بقالب JSON المخصص (metadata + questions.mcq + pipeline_meta).
    """
    from .generator import MATH_PROMPT_SET

    os.makedirs("outputs", exist_ok=True)
    clean_source = Path(source_file).stem.replace(" ", "_").replace(".", "_")
    source_folder_path = os.path.join("outputs", clean_source)
    os.makedirs(source_folder_path, exist_ok=True)

    mcq = _normalize_math_mcq(questions_data.get("mcq", []))
    if not mcq:
        raise ValueError("لا توجد أسئلة رياضية للحفظ")

    total = len(mcq)
    generated_at = datetime.now().isoformat()
    json_source_name = f"{Path(source_file).stem}.json"

    data = {
        "metadata": {
            "source_file": source_file,
            "generated_at": generated_at,
            "total_questions": total,
            "prompt_set": MATH_PROMPT_SET,
        },
        "questions": {
            "mcq": mcq,
            "pipeline_meta": {
                "source_file": json_source_name,
                "verified_count": total,
                "stubs_generated": total,
                "rejected_count": 0,
            },
            "metadata": {
                "source_file": source_file,
                "total_questions": total,
                "prompt_set": MATH_PROMPT_SET,
            },
        },
    }

    filename = f"math_{clean_source}.json"
    file_path = os.path.join(source_folder_path, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if not os.path.exists(file_path):
        raise IOError(f"فشل في حفظ الملف: {file_path}")

    print(f"✅ تم حفظ {total} سؤال رياضي في: {file_path}")
    return filename, total


def load_questions_by_model_method(model_name: str, method: str, source_file: str = None):
    """
    تحميل الأسئلة من ملف محدد حسب النموذج والطريقة والملف المصدر
    
    Args:
        model_name: اسم النموذج (llama3.2:3b, qwen2.5:7b, etc.)
        method: طريقة التوليد (vanilla, rag)
        source_file: اسم الملف المصدر (اختياري - إذا كان موجوداً يبحث في المجلد الخاص به)
    
    Returns:
        dict: البيانات المحفوظة أو None
    """
    # تحديد اسم الملف - استخدام نفس المنطق المستخدم في الحفظ
    if "llama" in model_name.lower():
        model_short = "llama"
    elif "qwen" in model_name.lower():
        model_short = "qwen"
    else:
        model_short = "llama"  # افتراضي
    
    method_short = "vanilla" if method.lower() == "vanilla" else "rag"
    
    # إذا كان هناك ملف مصدر محدد، البحث في مجلده
    if source_file:
        clean_source = Path(source_file).stem.replace(" ", "_").replace(".", "_")
        source_folder_path = os.path.join("outputs", clean_source)
        filename = f"questions_{model_short}_{method_short}_{clean_source}.json"
        file_path = os.path.join(source_folder_path, filename)
    else:
        # البحث في الملف العام (للتوافق مع الكود القديم)
        filename = f"questions_{model_short}_{method_short}.json"
        file_path = os.path.join("outputs", filename)
    
    # محاولة تحميل الملف
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    # إذا لم يُوجد في المكان المحدد، البحث في جميع المجلدات
    if not source_file:
        outputs_dir = "outputs"
        if os.path.exists(outputs_dir):
            # البحث في جميع المجلدات الفرعية
            for folder in os.listdir(outputs_dir):
                folder_path = os.path.join(outputs_dir, folder)
                if os.path.isdir(folder_path):
                    potential_file = os.path.join(folder_path, filename)
                    if os.path.exists(potential_file):
                        try:
                            with open(potential_file, "r", encoding="utf-8") as f:
                                return json.load(f)
                        except (json.JSONDecodeError, FileNotFoundError):
                            continue
    
    return None

def list_all_question_files():
    """
    قائمة بجميع ملفات الأسئلة المتاحة (في outputs و جميع المجلدات الفرعية)
    
    Returns:
        list: قائمة بأسماء الملفات مع المسارات النسبية
    """
    import glob
    files = []
    
    # البحث في outputs مباشرة
    files.extend(glob.glob("outputs/questions_*.json"))
    
    # البحث في جميع المجلدات الفرعية
    outputs_dir = "outputs"
    if os.path.exists(outputs_dir):
        for folder in os.listdir(outputs_dir):
            folder_path = os.path.join(outputs_dir, folder)
            if os.path.isdir(folder_path):
                files.extend(glob.glob(os.path.join(folder_path, "questions_*.json")))
    
    # إرجاع أسماء الملفات فقط (بدون المسار الكامل)
    return [os.path.basename(f) for f in files]

def find_questions_by_source_file(source_file: str):
    """
    البحث عن جميع ملفات الأسئلة لملف مصدر محدد
    
    Args:
        source_file: اسم الملف المصدر
    
    Returns:
        dict: قاموس يحتوي على ملفات الأسئلة لكل نموذج/طريقة
    """
    clean_source = Path(source_file).stem.replace(" ", "_").replace(".", "_")
    source_folder_path = os.path.join("outputs", clean_source)
    
    if not os.path.exists(source_folder_path):
        return {}
    
    results = {}
    for file in os.listdir(source_folder_path):
        if file.startswith("questions_") and file.endswith(".json"):
            # استخراج النموذج والطريقة من اسم الملف
            parts = file.replace("questions_", "").replace(f"_{clean_source}.json", "").split("_")
            if len(parts) >= 2:
                model_short = parts[0]  # llama أو qwen
                method_short = parts[1]  # vanilla أو rag
                key = f"{model_short}_{method_short}"
                file_path = os.path.join(source_folder_path, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        results[key] = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    continue
    
    return results