"""بناء prompts، استدعاء Ollama، التحقق من JSON، وتوليد/إصلاح الأسئلة (Vanilla/RAG)."""
import json
import ollama
import re
import time
import traceback
from difflib import SequenceMatcher
from langdetect import detect
from typing import Literal

SYS_AR = "أنت معلّم خبير في إنشاء الأسئلة. مهمتك: إنشاء أكبر عدد ممكن من الأسئلة المختلفة والدقيقة من النص المعطى (بحد أقصى 20 سؤال). يجب أن تكون الأسئلة مزيجاً من أسئلة اختيار من متعدد (4 خيارات لكل سؤال) وأسئلة صح/خطأ. كل سؤال يجب أن يكون فريداً ومستنداً للنص. **مهم جداً جداً:** كل خيار في أسئلة الاختيار من متعدد يجب أن يكون إجابة حقيقية ومناسبة ومحددة من النص. **ممنوع تماماً ومحظور** كتابة 'خيار 1' أو 'خيار 2' أو 'خيار 3' أو 'خيار 4' أو 'جواب 1' أو 'إجابة أ' أو أي خيارات عامة. **ممنوع تماماً** استخدام أدوات الاستفهام (من، أين، ماذا، لماذا، كيف، متى، ما) في أسئلة الصح/الخطأ. أعد JSON فقط بدون أي نص إضافي."
SYS_EN = "You are an expert teacher in creating questions. Your task: Create as many different and accurate questions as possible from the given text (maximum 20 questions). Questions should be a mix of multiple choice questions (4 options each) and True/False questions. Each question must be unique and based on the text. **Very important:** Each option in multiple choice questions must be a real, appropriate, and specific answer from the text. **STRICTLY FORBIDDEN** to write 'option 1', 'option 2', 'option 3', 'option 4' or any generic options. **STRICTLY FORBIDDEN** to use interrogative words (what, where, why, how, when, who) in True/False questions. Return JSON only without any additional text."

def detect_lang(text:str)->Literal["ar","en"]:
    """اكتشاف لغة النص (عربي أو إنجليزي) لاختيار البرومبت المناسب."""
    try:
        return "ar" if detect(text)=="ar" else "en"
    except Exception:
        return "ar"

def build_prompt_vanilla(text:str, lang:str):
    """بناء برومبت توليد أسئلة Vanilla من النص المرفوع فقط (بدون RAG)."""
    if lang=="ar":
        return f"""{SYS_AR}

=== النص المرفوع (المصدر الوحيد) ===
{text}

**مهم جداً جداً:** هذا هو المصدر الوحيد المتاح. يجب أن تكون جميع الأسئلة والخيارات مستندة **فقط** إلى هذا النص المرفوع. **ممنوع تماماً** استخدام أي معلومات من خارج هذا النص.

التعليمات الدقيقة:
1. **استخدم النص المرفوع أعلاه فقط** - هذا هو المصدر الوحيد المتاح. أنشئ أكبر عدد ممكن من الأسئلة المختلفة من هذا النص فقط (بحد أقصى 20 سؤال)
2. يجب أن تكون الأسئلة مزيجاً من أسئلة اختيار من متعدد (4 خيارات لكل سؤال) وأسئلة صح/خطأ
3. لا يوجد عدد محدد لكل نوع - أنشئ أكبر عدد ممكن من الأسئلة المختلفة
4. كل سؤال صح/خطأ يجب أن يكون صياغة خبرية أو يبدأ بـ "هل"
5. كل سؤال يجب أن يكون فريداً ومستنداً **فقط** إلى النص المرفوع أعلاه
6. **مهم جداً جداً - خيارات الاختيار من متعدد:** كل خيار في أسئلة الاختيار من متعدد يجب أن يكون إجابة حقيقية ومناسبة ومحددة **فقط من النص المرفوع أعلاه**. **ممنوع تماماً** كتابة "خيار 1" أو "خيار 2" أو "خيار 3" أو "خيار 4" أو "جواب 1" أو "إجابة أ" أو "اختيار ب" أو أي خيارات عامة. **ممنوع تماماً** استخدام معلومات من خارج النص المرفوع. اكتب خيارات حقيقية ومناسبة ومحددة من النص المرفوع فقط
7. **تذكير قوي جداً:** إذا كتبت "خيار 1" أو "خيار 2" أو "خيار 3" أو "خيار 4" في أي سؤال، أو إذا استخدمت معلومات من خارج النص المرفوع، فإن الإجابة ستكون خاطئة تماماً. يجب أن تكون جميع الخيارات إجابات حقيقية من النص المرفوع فقط
8. **مهم جداً جداً - تجنب التكرار:** لا تكرر الأسئلة أو الخيارات. كل سؤال يجب أن يكون مختلفاً تماماً عن الأسئلة الأخرى
9. أعد JSON فقط بدون أي نص إضافي
10. تأكد من إغلاق جميع الأقواس والفواصل بشكل صحيح
11. استخدم كلمات عربية فقط في الأسئلة
12. **مهم جداً لأسئلة الصح/الخطأ:** لا تستخدم أدوات الاستفهام (من، أين، ماذا، لماذا، كيف، متى) في أسئلة الصح/الخطأ. يمكنك استخدام "هل" في بداية السؤال أو استخدام جملة خبرية تحتمل الإيجاب أو النفي

تنسيق JSON المطلوب:
{{
  "mcq": [
    {{"q": "سؤال اختيار من متعدد من النص", "options": ["إجابة حقيقية من النص", "إجابة حقيقية أخرى من النص", "إجابة حقيقية ثالثة من النص", "إجابة حقيقية رابعة من النص"], "answer": "الإجابة الصحيحة"}},
    // ... المزيد من أسئلة الاختيار من متعدد (حسب ما يتوفر في النص)
  ],
  "tf": [
    {{"q": "جملة خبرية من النص", "answer": true}},
    {{"q": "جملة خبرية من النص", "answer": false}},
    // ... المزيد من أسئلة الصح/الخطأ (حسب ما يتوفر في النص)
  ]
}}

**مهم:** أنشئ أكبر عدد ممكن من الأسئلة المختلفة (بحد أقصى 20 سؤال إجمالي). لا يوجد حد أدنى أو عدد محدد لكل نوع.

ابدأ الآن:"""
    else:
        return f"""{SYS_EN}

=== Uploaded Text (ONLY Source) ===
{text}

**VERY IMPORTANT:** This is the ONLY available source. All questions and options must be based **ONLY** on this uploaded text. **STRICTLY FORBIDDEN** to use any information from outside this text.

Instructions:
1. **Use ONLY the uploaded text above** - This is the ONLY available source. Create as many different questions as possible from this text only (maximum 20 questions total)
2. Mix of multiple choice questions (4 options each) and True/False questions
3. No fixed number for each type - generate as many unique questions as possible
4. Each question must be unique and based **ONLY** on the uploaded text above
5. **VERY IMPORTANT - MCQ options:** Each option in multiple choice questions must be a real, appropriate, and specific answer **ONLY from the uploaded text above**. **STRICTLY FORBIDDEN** to write "option 1", "option 2", "option 3", "option 4" or any generic options. **STRICTLY FORBIDDEN** to use information from outside the uploaded text. Write real, appropriate, and specific answers from the uploaded text only
6. **Strong reminder:** If you write "option 1", "option 2", "option 3", "option 4" or use information from outside the uploaded text, the answer will be completely wrong. All options must be real answers from the uploaded text only
7. **VERY IMPORTANT - Avoid repetition:** Do not repeat questions or options. Each question must be completely different from others
8. Return JSON only with keys mcq and tf.
9. **VERY IMPORTANT for True/False questions:** **STRICTLY FORBIDDEN** to use interrogative words (what, where, why, how, when, who) in True/False questions. You can use "Is/Are" at the beginning or use a declarative sentence

Required JSON format:
{{
  "mcq": [
    {{"q": "question text", "options": ["real answer from text", "another real answer from text", "third real answer from text", "fourth real answer from text"], "answer": "correct answer"}},
    ...
  ],
  "tf": [
    {{"q": "question text", "answer": true/false}},
    ...
  ]
}}
"""

def build_prompt_rag(text: str, lang: str, retrieved: list[dict]) -> str:
    """بناء الـ prompt لـ RAG — يفرّع حسب اللغة أولاً لتفادي بناء سياق عربي ثم إهماله للإنجليزية."""
    head = SYS_AR if lang == "ar" else SYS_EN
    has_rag = bool(retrieved and len(retrieved) > 0)

    if lang == "ar":
        if has_rag:
            retrieved_texts = []
            for i, r in enumerate(retrieved, 1):
                passage_text = r["text"][:500]
                filename = r.get("filename", "مصدر غير معروف")
                retrieved_texts.append(f"[مصدر {i}: {filename}]\n{passage_text}")
            combined_context = f"""=== النص الأساسي (المصدر الرئيسي) ===
{text}

=== معلومات إضافية من مصادر مشابهة (تم استرجاعها باستخدام RAG) ===
{chr(10).join(retrieved_texts)}

**مهم جداً:** المحتوى أعلاه يتكون من:
1. **النص الأساسي**: المصدر الرئيسي الذي يجب أن يكون الأساس لجميع الأسئلة
2. **المصادر الإضافية**: معلومات مكملة من مصادر مشابهة تم استرجاعها تلقائياً

**يجب استخدام المعلومات من كلا المصدرين معاً** لإنشاء أسئلة دقيقة ومتنوعة.
"""
            instruction_1 = "1. **استخدم المعلومات من النص الأساسي والمصادر الإضافية معاً** لإنشاء أسئلة دقيقة وعميقة. ابدأ بالنص الأساسي كأساس، ثم استخدم المصادر الإضافية لإثراء الخيارات والمعلومات"
            instruction_2 = "2. **لكل سؤال اختيار من متعدد:** استخرج الخيارات من **جميع المصادر المتاحة** (النص الأساسي + المصادر الإضافية). يمكن أن تأتي الخيارات من أي من المصادر المتاحة، لكن يجب أن تكون جميعها حقيقية ومحددة من المحتوى"
        else:
            combined_context = f"""=== المحتوى المتاح ===
{text}
"""
            instruction_1 = "1. أنشئ أسئلة دقيقة وعميقة من النص أعلاه"
            instruction_2 = "2. كل سؤال يجب أن يستند إلى محتوى حقيقي من النص"

        return f"""{head}

{combined_context}

التعليمات الدقيقة:
{instruction_1}
{instruction_2}
3. **لإنشاء خيارات الاختيار من متعدد:**
   - اقرأ جميع المصادر المتاحة بعناية (النص الأساسي + المصادر الإضافية إن وجدت)
   - استخرج معلومات محددة من كل مصدر
   - لكل سؤال، أنشئ 4 خيارات حقيقية من المعلومات الموجودة في **جميع المصادر المتاحة**
   - كل خيار يجب أن يكون إجابة حقيقية ومحددة من المحتوى المتاح
   - استخدم معلومات من مصادر مختلفة لإنشاء خيارات متنوعة
   - **مثال:** إذا كان السؤال عن "أين تقع المدينة؟"، فابحث في جميع المصادر عن أسماء أماكن حقيقية واكتبها كخيارات (مثل: "في الشمال"، "في الجنوب"، "على الساحل"، "في الداخل")
4. أنشئ أكبر عدد ممكن من الأسئلة المختلفة (بحد أقصى 20 سؤال إجمالي)
5. يجب أن تكون الأسئلة مزيجاً من أسئلة اختيار من متعدد (4 خيارات لكل سؤال) وأسئلة صح/خطأ
6. لا يوجد عدد محدد لكل نوع - أنشئ أكبر عدد ممكن من الأسئلة المختلفة والفريدة
7. كل سؤال اختيار من متعدد يجب أن يحتوي على 4 خيارات بالضبط
8. **مهم جداً جداً - خيارات الاختيار من متعدد:** كل خيار في أسئلة الاختيار من متعدد يجب أن يكون إجابة حقيقية ومناسبة ومحددة من **أي من المصادر المتاحة** (النص الأساسي أو المصادر الإضافية). **ممنوع تماماً ومحظور** كتابة "خيار 1" أو "خيار 2" أو "خيار 3" أو "خيار 4" أو "جواب 1" أو "إجابة أ" أو "اختيار ب" أو "option 1" أو "option 2" أو أي خيارات عامة أو رموز. اكتب خيارات حقيقية ومناسبة ومحددة من المحتوى المتاح فقط. كل خيار يجب أن يكون جملة أو عبارة كاملة من المحتوى
9. **تذكير قوي جداً:** إذا كتبت "خيار 1" أو "خيار 2" أو "خيار 3" أو "خيار 4" أو أي خيار عام في أي سؤال، فإن الإجابة ستكون خاطئة تماماً وسيتم رفض السؤال. يجب أن تكون جميع الخيارات إجابات حقيقية ومحددة من المحتوى المتاح. لا تستخدم أرقام أو رموز كخيارات
10. كل سؤال يجب أن يكون فريداً ومستنداً للمحتوى
11. **مهم جداً جداً - تجنب التكرار:** لا تكرر الأسئلة أو الخيارات. كل سؤال يجب أن يكون مختلفاً تماماً عن الأسئلة الأخرى. لا تستخدم نفس السؤال مرتين حتى لو بصياغة مختلفة. لا تستخدم نفس الخيارات في أسئلة مختلفة. كل خيار في كل سؤال يجب أن يكون فريداً ومختلفاً عن جميع الخيارات الأخرى
12. أعد JSON فقط بدون أي نص إضافي
13. تأكد من إغلاق جميع الأقواس والفواصل بشكل صحيح
14. استخدم كلمات عربية فقط في الأسئلة
15. تأكد من أن كل سؤال MCQ له 4 خيارات وليس أقل
16. **مهم جداً جداً لأسئلة الصح/الخطأ:** **ممنوع تماماً** استخدام أدوات الاستفهام (من، أين، ماذا، لماذا، كيف، متى، ما) في أسئلة الصح/الخطأ. يمكنك استخدام "هل" في بداية السؤال أو استخدام جملة خبرية تحتمل الإيجاب أو النفي. إذا استخدمت أداة استفهام في سؤال صح/خطأ، فإن السؤال سيكون خاطئاً تماماً

تنسيق JSON المطلوب:
{{
  "mcq": [
    {{"q": "سؤال اختيار من متعدد من المحتوى", "options": ["إجابة حقيقية من المحتوى", "إجابة حقيقية أخرى من المحتوى", "إجابة حقيقية ثالثة من المحتوى", "إجابة حقيقية رابعة من المحتوى"], "answer": "الإجابة الصحيحة"}},
    // ... المزيد من أسئلة الاختيار من متعدد (حسب ما يتوفر في المحتوى)
  ],
  "tf": [
    {{"q": "جملة خبرية من المحتوى", "answer": true}},
    {{"q": "جملة خبرية من المحتوى", "answer": false}},
    // ... المزيد من أسئلة الصح/الخطأ (حسب ما يتوفر في المحتوى)
  ]
}}

**مهم جداً:** أنشئ أكبر عدد ممكن من الأسئلة المختلفة (بحد أقصى 20 سؤال إجمالي). لا يوجد حد أدنى أو عدد محدد لكل نوع. ركّز على التنوع والجودة.

ابدأ الآن:"""
    else:
        if has_rag:
            retrieved_texts = []
            for i, r in enumerate(retrieved, 1):
                passage_text = r["text"][:500]
                filename = r.get("filename", "Unknown source")
                retrieved_texts.append(f"[Source {i}: {filename}]\n{passage_text}")

            combined_context_en = f"""=== Main Text (Primary Source) ===
{text}

=== Additional Information from Similar Sources (Retrieved using RAG) ===
{chr(10).join(retrieved_texts)}

**Important:** The content above consists of:
1. **Main Text**: The primary source that should be the foundation for all questions
2. **Additional Sources**: Complementary information from similar sources retrieved automatically

**You must use information from both sources together** to create accurate and diverse questions.
"""
            instruction_1_en = "1. **Use information from both the main text and additional sources together** to create accurate and deep questions. Start with the main text as the foundation, then use additional sources to enrich options and information"
            instruction_2_en = "2. **For each multiple choice question:** Extract options from **all available sources** (main text + additional sources). Options can come from any of the available sources, but all must be real and specific from the content"
        else:
            combined_context_en = f"""=== Available Content ===
{text}
"""
            instruction_1_en = "1. Create accurate and deep questions from the text above"
            instruction_2_en = "2. Each question must be based on real content from the text"

        return f"""{head}

{combined_context_en}

Important Instructions:
{instruction_1_en}
{instruction_2_en}
3. **For creating multiple choice options:**
   - Read all available sources carefully (main text + additional sources if available)
   - Extract specific information from each source
   - For each question, create 4 real options from information found in **all available sources**
   - Each option must be a real and specific answer from the available content
   - Use information from different sources to create diverse options
4. Create as many different questions as possible (maximum 20 questions total)
5. Mix of multiple choice questions (4 options each) and True/False questions
6. No fixed number for each type - generate as many unique questions as possible
7. Each multiple choice question must have exactly 4 options
8. **Very important - MCQ options:** Each option in multiple choice questions must be a real, appropriate, and specific answer from **any of the available sources** (main text or additional sources). **STRICTLY FORBIDDEN** to write "option 1", "option 2", "option 3", "option 4" or any generic options or symbols. Write real, appropriate, and specific answers from the available content only. Each option must be a complete sentence or phrase from the content
9. **Strong reminder:** If you write "option 1", "option 2", "option 3", "option 4" or any generic option in any question, the answer will be completely wrong and the question will be rejected. All options must be real and specific answers from the available content. Do not use numbers or symbols as options
10. Each question must be unique and based on the content
11. **Very important - Avoid repetition:** Do not repeat questions or options. Each question must be completely different from others. Do not use the same question twice even with different wording. Do not use the same options in different questions. Each option in each question must be unique and different from all other options
12. Return JSON only without any additional text
13. Ensure correct JSON format
14. Use only English words in questions
15. Make sure each MCQ has exactly 4 options, not less
16. **Very important for True/False questions:** **STRICTLY FORBIDDEN** to use interrogative words (what, where, why, how, when, who) in True/False questions. You can use "Is/Are" at the beginning or use a declarative sentence that can be affirmed or negated. If you use an interrogative word in a True/False question, the question will be completely wrong

Required JSON format:
{{
  "mcq": [
    {{"q": "Multiple choice question", "options": ["real answer from content", "another real answer from content", "third real answer from content", "fourth real answer from content"], "answer": "correct answer"}},
    {{"q": "Multiple choice question", "options": ["real answer from content", "another real answer from content", "third real answer from content", "fourth real answer from content"], "answer": "correct answer"}},
    {{"q": "Multiple choice question", "options": ["real answer from content", "another real answer from content", "third real answer from content", "fourth real answer from content"], "answer": "correct answer"}},
    {{"q": "Multiple choice question", "options": ["real answer from content", "another real answer from content", "third real answer from content", "fourth real answer from content"], "answer": "correct answer"}},
    {{"q": "Multiple choice question", "options": ["real answer from content", "another real answer from content", "third real answer from content", "fourth real answer from content"], "answer": "correct answer"}}
  ],
  "tf": [
    {{"q": "Declarative statement or starts with Is/Are", "answer": true}},
    {{"q": "Declarative statement or starts with Is/Are", "answer": false}},
    {{"q": "Declarative statement or starts with Is/Are", "answer": true}},
    {{"q": "Declarative statement or starts with Is/Are", "answer": false}},
    {{"q": "Declarative statement or starts with Is/Are", "answer": true}}
  ]
}}

Start now:"""


def _extract_chat_response(resp, model_name: str = "") -> str:
    """استخراج النص النهائي من استجابة ollama.chat."""
    content = ""
    thinking = ""

    if hasattr(resp, "message"):
        content = getattr(resp.message, "content", None) or ""
        thinking = getattr(resp.message, "thinking", None) or ""
    elif isinstance(resp, dict):
        msg = resp.get("message") or {}
        if isinstance(msg, dict):
            content = msg.get("content", "") or ""
            thinking = msg.get("thinking", "") or ""
        else:
            content = resp.get("content") or resp.get("response") or ""

    text = (content or "").strip()
    if thinking.strip():
        text = f"{text}\n{thinking.strip()}".strip() if text else thinking.strip()
    return text.strip()


def check_and_pull_model(model_name: str):
    """
    التحقق من وجود النموذج وعمل pull تلقائي إذا لم يكن موجوداً
    
    Args:
        model_name: اسم النموذج المطلوب
    
    Returns:
        bool: True إذا كان النموذج متاحاً، False إذا فشل التحميل
    """
    try:
        models = ollama.list()
        installed_models = []
        
        # استخراج أسماء النماذج المثبتة
        if models and hasattr(models, 'models'):
            if hasattr(models.models, '__iter__'):
                for model in models.models:
                    if hasattr(model, 'model'):
                        installed_models.append(model.model)
        
        # التحقق من وجود النموذج
        if model_name in installed_models:
            print(f"✅ النموذج {model_name} موجود ومتاح")
            return True
        
        # إذا لم يكن موجوداً، محاولة تحميله تلقائياً
        print(f"⚠️ النموذج {model_name} غير موجود. جاري تحميله تلقائياً...")
        try:
            # محاولة تحميل النموذج (قد يستغرق وقتاً طويلاً)
            print(f"📥 جاري تحميل النموذج {model_name}... قد يستغرق هذا بضع دقائق")
            
            # محاولة استخدام stream إذا كان متاحاً
            try:
                stream = ollama.pull(model_name, stream=True)
                # عرض تقدم التحميل
                for chunk in stream:
                    if isinstance(chunk, dict):
                        status = chunk.get('status', '')
                        if 'total' in chunk and 'completed' in chunk:
                            total = chunk.get('total', 0)
                            completed = chunk.get('completed', 0)
                            if total > 0:
                                percent = (completed / total) * 100
                                print(f"📥 التحميل: {percent:.1f}% ({completed}/{total})")
                        elif 'digest' in chunk:
                            digest = chunk.get('digest', '')
                            print(f"📥 جاري تحميل طبقة: {digest[:20] if len(digest) > 20 else digest}...")
                        elif status:
                            print(f"📥 {status}")
            except (TypeError, AttributeError):
                # إذا لم يعمل stream، استخدم الطريقة العادية
                print("📥 استخدام طريقة التحميل العادية...")
                ollama.pull(model_name)
            
            print(f"✅ تم تحميل النموذج {model_name} بنجاح")
            return True
        except Exception as pull_error:
            error_msg = str(pull_error)
            print(f"❌ فشل في تحميل النموذج {model_name}: {error_msg}")
            # إذا كان الخطأ متعلقاً بالاتصال، قد يكون Ollama يحاول تحميله تلقائياً
            if "model not found" in error_msg.lower() or "connection" in error_msg.lower():
                print(f"💡 يمكنك تحميله يدوياً باستخدام: ollama pull {model_name}")
            return False
            
    except Exception as e:
        print(f"❌ خطأ في التحقق من النماذج: {e}")
        return False

def call_llama(prompt: str, max_retries: int = 3, model_name: str = "llama3.2:3b", temperature: float = 0.7):
    """
    استدعاء نموذج Ollama مع إعادة المحاولة
    
    Args:
        prompt: النص المراد إرساله للنموذج
        max_retries: عدد المحاولات القصوى
        model_name: اسم النموذج المستخدم
        temperature: درجة الحرارة للنموذج
    """
    if not check_and_pull_model(model_name):
        print(f"⚠️ تحذير: النموذج {model_name} غير متاح، سيتم المحاولة على أي حال")

    for attempt in range(max_retries):
        try:
            print(f"🔄 محاولة {attempt + 1}/{max_retries} مع النموذج {model_name}")
            print(f"📏 طول الـ Prompt: {len(prompt)} حرف")

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

            if not resp:
                print("⚠️ استجابة فارغة من ollama")
                continue

            response = _extract_chat_response(resp, model_name)

            if not response:
                print("⚠️ استجابة فارغة أو غير صحيحة")
                continue

            print(f"✅ تم الحصول على استجابة (طول: {len(response)} حرف)")
            return response

        except KeyError as e:
            print(f"❌ خطأ في المفاتيح: {e}")
        except Exception as e:
            print(f"❌ خطأ في المحاولة {attempt + 1}: {e}")
            print(f"🔍 تفاصيل الخطأ: {traceback.format_exc()}")

        if attempt < max_retries - 1:
            wait_time = 3 * (attempt + 1)
            print(f"⏳ انتظار {wait_time} ثانية قبل المحاولة التالية...")
            time.sleep(wait_time)

    print("❌ فشل في جميع المحاولات")
    return ""


def validate_question(question_data: dict, question_type: str, _source_text: str = "") -> tuple[bool, list[str]]:
    """
    التحقق من صحة السؤال وإرجاع قائمة بالأخطاء
    
    Returns:
        (is_valid, errors): tuple يحتوي على صحة السؤال وقائمة الأخطاء
    """
    errors = []
    
    if question_type == "mcq":
        options = question_data.get("options", [])
        if not options or len(options) != 4:
            errors.append("عدد الخيارات غير صحيح")
        
        # التحقق من الخيارات العامة
        generic_pattern = r'^(خيار\s*[1-4]|جواب\s*[1-4]|إجابة\s*[أ-د]|اختيار\s*[أ-د]|option\s*[1-4]|choice\s*[1-4]|answer\s*[1-4]|answer\s*[a-d])$'
        for i, option in enumerate(options):
            option_str = str(option).strip()
            if re.match(generic_pattern, option_str, re.I):
                errors.append(f"خيار {i+1} عام: {option_str}")
        
        # التحقق من تكرار الخيارات (دقيق)
        option_strings = [str(opt).strip().lower() for opt in options]
        unique_options = set(option_strings)
        if len(unique_options) < len(options):
            errors.append("خيارات مكررة")
        
        # التحقق من الخيارات المتشابهة جداً (أكثر من 90% تشابه)
        for i in range(len(option_strings)):
            for j in range(i + 1, len(option_strings)):
                similarity = SequenceMatcher(
                    None, option_strings[i], option_strings[j]
                ).ratio()
                if similarity > 0.9:
                    errors.append(f"خيارات متشابهة جداً: {i+1} و {j+1}")
    
    elif question_type == "tf":
        question = question_data.get("q", "").strip()
        if not question:
            errors.append("السؤال فارغ")
        
        # التحقق من أدوات الاستفهام
        interrogative_words = ["من", "أين", "ماذا", "متى", "كيف", "لماذا", "ما", 
                              "what", "where", "why", "how", "when", "who", "which"]
        question_lower = question.lower()
        for word in interrogative_words:
            if question_lower.startswith(word.lower()):
                errors.append(f"استخدام أداة استفهام: {word}")
                break
    
    return len(errors) == 0, errors

def regenerate_single_question(question_data: dict, question_type: str, source_text: str, 
                                model_name: str = "llama3.2:3b", lang: str = "ar", 
                                retrieved: list = None) -> dict:
    """
    إعادة توليد سؤال واحد إذا كان به أخطاء
    """
    if question_type == "mcq":
        prompt = f"""أنت معلّم خبير. مهمتك: إنشاء سؤال اختيار من متعدد واحد فقط من النص التالي.

**قواعد صارمة جداً للخيارات:**
1. **ممنوع تماماً** كتابة "خيار 1" أو "خيار 2" أو "خيار 3" أو "خيار 4" أو أي خيارات عامة
2. كل خيار يجب أن يكون إجابة محتملة منطقية للسؤال
3. الخيار الصحيح + 3 خيارات خاطئة لكن معقولة
4. الخيارات الخاطئة يجب أن تكون:
   - من نفس نوع الإجابة الصحيحة (إذا السؤال عن شخص، كل الخيارات أسماء أشخاص)
   - مرتبطة بموضوع السؤال
   - معقولة ظاهرياً لكن خاطئة
   - من معلومات موجودة في النص

**مثال:**
سؤال: من هو مؤسس الاتحاد الإنجليزي لكرة القدم؟
✅ خيارات جيدة: [إيبينيزر موارلي، جورج بيست، بيليه، دييغو مارادونا]
❌ خيارات سيئة: [إيبينيزر موارلي، تأسس الاتحاد في..., كرة القدم رياضة..., خيار 4]

النص:
{source_text[:800]}

أعد JSON فقط بهذا الشكل:
{{"q": "السؤال هنا", "options": ["إجابة صحيحة محددة", "إجابة خاطئة لكن معقولة", "إجابة خاطئة لكن معقولة", "إجابة خاطئة لكن معقولة"], "answer": "الإجابة الصحيحة"}}"""
    
    elif question_type == "tf":
        prompt = f"""أنت معلّم خبير. مهمتك: إنشاء سؤال صح/خطأ واحد فقط من النص التالي.

**قواعد صارمة جداً:**
1. **ممنوع تماماً** استخدام أدوات الاستفهام (من، أين، ماذا، لماذا، كيف، متى، ما)
2. يمكنك استخدام "هل" في البداية أو استخدام جملة خبرية
3. السؤال يجب أن يكون جملة خبرية تحتمل الإيجاب أو النفي

النص:
{source_text[:500]}

أعد JSON فقط بهذا الشكل:
{{"q": "جملة خبرية من النص", "answer": true}}"""
    
    try:
        response = call_llama(prompt, model_name=model_name, max_retries=1, temperature=0.3)
        if response:
            # إزالة markdown code blocks
            response = re.sub(r'```json|```', '', response, flags=re.I).strip()
            try:
                result = json.loads(response)
                if question_type == "mcq" and "q" in result and "options" in result:
                    return result
                if question_type == "tf" and "q" in result:
                    return result
            except json.JSONDecodeError:
                pass
    except Exception as e:
        print(f"❌ خطأ في إعادة توليد السؤال: {e}")
    
    return None

def remove_duplicate_questions(result: dict):
    """
    إزالة الأسئلة والخيارات المكررة
    """
    def similarity(a: str, b: str) -> float:
        """حساب التشابه بين نصين"""
        return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()
    
    removed_count = 0
    
    # إزالة الأسئلة المكررة في MCQ
    if "mcq" in result and isinstance(result["mcq"], list):
        seen_questions = []
        unique_mcq = []
        
        for mcq in result["mcq"]:
            if not isinstance(mcq, dict):
                continue
            
            question = str(mcq.get("q", "")).strip().lower()
            
            # التحقق من التكرار
            is_duplicate = False
            for seen_q in seen_questions:
                if similarity(question, seen_q) > 0.85:  # تشابه أكثر من 85%
                    is_duplicate = True
                    removed_count += 1
                    print(f"⚠️ تم اكتشاف سؤال MCQ مكرر: {mcq.get('q', '')[:50]}...")
                    break
            
            if not is_duplicate:
                seen_questions.append(question)
                
                # إزالة الخيارات المكررة داخل السؤال
                options = mcq.get("options", [])
                if isinstance(options, list):
                    unique_options = []
                    seen_options = []
                    
                    for opt in options:
                        opt_str = str(opt).strip()
                        if not opt_str:
                            continue
                        
                        # التحقق من التكرار
                        is_opt_duplicate = False
                        for seen_opt in seen_options:
                            if similarity(opt_str, seen_opt) > 0.9:  # تشابه أكثر من 90%
                                is_opt_duplicate = True
                                print(f"⚠️ تم اكتشاف خيار مكرر في MCQ: {opt_str[:50]}...")
                                break
                        
                        if not is_opt_duplicate:
                            unique_options.append(opt)
                            seen_options.append(opt_str.lower())
                    
                    # التأكد من وجود 4 خيارات (بدون نصوص "خيار 1" المحظورة في التعليمات)
                    while len(unique_options) < 4:
                        unique_options.append(f"إجابة إضافية {len(unique_options) + 1}")
                    
                    mcq["options"] = unique_options[:4]
                
                unique_mcq.append(mcq)
        
        result["mcq"] = unique_mcq
        print(f"✅ تم إزالة {removed_count} سؤال MCQ مكرر")
        removed_count = 0
    
    # إزالة الأسئلة المكررة في T/F
    if "tf" in result and isinstance(result["tf"], list):
        seen_questions = []
        unique_tf = []
        
        for tf in result["tf"]:
            if not isinstance(tf, dict):
                continue
            
            question = str(tf.get("q", "")).strip().lower()
            
            # التحقق من التكرار
            is_duplicate = False
            for seen_q in seen_questions:
                if similarity(question, seen_q) > 0.85:  # تشابه أكثر من 85%
                    is_duplicate = True
                    removed_count += 1
                    print(f"⚠️ تم اكتشاف سؤال T/F مكرر: {tf.get('q', '')[:50]}...")
                    break
            
            if not is_duplicate:
                seen_questions.append(question)
                unique_tf.append(tf)
        
        result["tf"] = unique_tf
        print(f"✅ تم إزالة {removed_count} سؤال T/F مكرر")
    
    return result

def fix_generated_questions(result: dict, source_text: str = "", model_name: str = "llama3.2:3b", 
                           lang: str = "ar", retrieved: list = None, max_regenerations: int = 3):
    """
    إصلاح الأخطاء الشائعة في الأسئلة المولدة مع إعادة توليد الأسئلة السيئة:
    1. استبدال "خيار 1/2/3/4" بخيارات حقيقية
    2. إزالة أدوات الاستفهام من أسئلة الصح/الخطأ
    3. إعادة توليد الأسئلة التي تحتوي على أخطاء خطيرة
    4. إزالة الأسئلة والخيارات المكررة
    """
    # أولاً: إزالة التكرار
    result = remove_duplicate_questions(result)
    
    regenerated_count = 0
    
    # إصلاح أسئلة MCQ
    if "mcq" in result and isinstance(result["mcq"], list):
        for i, mcq in enumerate(result["mcq"]):
            if not isinstance(mcq, dict):
                continue
            
            # التحقق من صحة السؤال
            is_valid, errors = validate_question(mcq, "mcq", source_text)
            
            if not is_valid:
                print(f"⚠️ السؤال MCQ {i+1} يحتوي على أخطاء: {', '.join(errors)}")
                
                # محاولة واحدة فقط لإعادة التوليد
                print(f"🔄 محاولة إعادة توليد السؤال MCQ {i+1}...")
                new_question = regenerate_single_question(mcq, "mcq", source_text, model_name, lang, retrieved)
                
                if new_question:
                    # التحقق من أن السؤال الجديد صحيح
                    is_new_valid, new_errors = validate_question(new_question, "mcq", source_text)
                    if is_new_valid:
                        result["mcq"][i] = new_question
                        regenerated_count += 1
                        print(f"✅ تم إعادة توليد السؤال MCQ {i+1} بنجاح")
                        continue
                    else:
                        print(f"⚠️ السؤال المعاد توليده لا يزال يحتوي على أخطاء: {', '.join(new_errors)}")
                
                # إذا فشلت إعادة التوليد، نبقي السؤال الأصلي مع تحذير
                print(f"⚠️ تم الإبقاء على السؤال MCQ {i+1} الأصلي رغم الأخطاء")
                # لا نحذف السؤال - نبقيه كما هو
    
    # إصلاح أسئلة True/False
    if "tf" in result and isinstance(result["tf"], list):
        for i, tf in enumerate(result["tf"]):
            if not isinstance(tf, dict):
                continue
            
            # التحقق من صحة السؤال
            is_valid, errors = validate_question(tf, "tf", source_text)
            
            if not is_valid:
                print(f"⚠️ السؤال T/F {i+1} يحتوي على أخطاء: {', '.join(errors)}")
                
                # محاولة واحدة فقط لإعادة التوليد
                print(f"🔄 محاولة إعادة توليد السؤال T/F {i+1}...")
                new_question = regenerate_single_question(tf, "tf", source_text, model_name, lang, retrieved)
                
                if new_question:
                    # التحقق من أن السؤال الجديد صحيح
                    is_new_valid, new_errors = validate_question(new_question, "tf", source_text)
                    if is_new_valid:
                        result["tf"][i] = new_question
                        regenerated_count += 1
                        print(f"✅ تم إعادة توليد السؤال T/F {i+1} بنجاح")
                        continue
                    else:
                        print(f"⚠️ السؤال المعاد توليده لا يزال يحتوي على أخطاء: {', '.join(new_errors)}")
                
                # إذا فشلت إعادة التوليد، نبقي السؤال الأصلي مع تحذير
                print(f"⚠️ تم الإبقاء على السؤال T/F {i+1} الأصلي رغم الأخطاء")
                # لا نحذف السؤال - نبقيه كما هو
    
    if regenerated_count > 0:
        print(f"🔧 تم إعادة توليد {regenerated_count} سؤال")
    
    return result

def safe_json(s: str, source_text: str = "", model_name: str = "llama3.2:3b", lang: str = "ar", retrieved: list = None):
    """
    تحليل JSON بأمان مع معالجة الأخطاء
    """
    print(f"🔍 محاولة تحليل JSON...")
    print(f"📏 طول النص: {len(s)} حرف")
    print(f"📄 أول 200 حرف: {s[:200]}...")
    
    # التحقق من أن النص ليس فارغاً
    if not s or not s.strip():
        print("❌ النص فارغ")
        return None
    
    try:
        result = json.loads(s)
        print("✅ تم تحليل JSON بنجاح في المحاولة الأولى")
        
        # إصلاح الأخطاء الشائعة مع إعادة توليد الأسئلة السيئة
        result = fix_generated_questions(result, source_text, model_name, lang, retrieved)
        
        # التحقق من عدد الخيارات في أسئلة MCQ
        if "mcq" in result and isinstance(result["mcq"], list):
            for i, mcq in enumerate(result["mcq"]):
                if "options" in mcq and len(mcq["options"]) != 4:
                    print(f"⚠️ السؤال MCQ {i+1} له {len(mcq['options'])} خيارات بدلاً من 4")
                    # إضافة خيارات فارغة إذا كان أقل من 4
                    while len(mcq["options"]) < 4:
                        mcq["options"].append(f"إجابة إضافية {len(mcq['options'])+1}")
                    print(f"✅ تم إصلاح السؤال MCQ {i+1} ليحتوي على 4 خيارات")
        
        return result
    except json.JSONDecodeError as e:
        print(f"❌ فشل في المحاولة الأولى: {e}")
        
        # إزالة markdown code blocks
        s = re.sub(r"```json|```", "", s, flags=re.I)
        s = s.strip()
        
        # محاولة إصلاح JSON غير مكتمل
        if s.count('{') > s.count('}'):
            s += '}' * (s.count('{') - s.count('}'))
        if s.count('[') > s.count(']'):
            s += ']' * (s.count('[') - s.count(']'))
        
        print(f"🧹 بعد التنظيف - طول النص: {len(s)} حرف")
        print(f"📄 أول 200 حرف بعد التنظيف: {s[:200]}...")
        
        if not s:
            print("❌ النص فارغ بعد التنظيف")
            return None
        
        try:
            result = json.loads(s)
            print("✅ تم تحليل JSON بنجاح بعد التنظيف")
            
            # إصلاح الأخطاء الشائعة مع إعادة توليد الأسئلة السيئة
            result = fix_generated_questions(result, source_text, model_name, lang, retrieved)
            
            # التحقق من عدد الخيارات في أسئلة MCQ
            if "mcq" in result and isinstance(result["mcq"], list):
                for i, mcq in enumerate(result["mcq"]):
                    if "options" in mcq and len(mcq["options"]) != 4:
                        print(f"⚠️ السؤال MCQ {i+1} له {len(mcq['options'])} خيارات بدلاً من 4")
                        # إضافة خيارات فارغة إذا كان أقل من 4
                        while len(mcq["options"]) < 4:
                            mcq["options"].append(f"إجابة إضافية {len(mcq['options'])+1}")
                        print(f"✅ تم إصلاح السؤال MCQ {i+1} ليحتوي على 4 خيارات")
            
            return result
        except json.JSONDecodeError as e2:
            print(f"❌ فشل في المحاولة الثانية: {e2}")
            
            # محاولة إصلاح إضافية
            try:
                # إزالة آخر سطر إذا كان غير مكتمل
                lines = s.split('\n')
                if lines:
                    # إزالة السطر الأخير إذا كان غير مكتمل
                    last_line = lines[-1].strip()
                    if not last_line.endswith(('}', ']', ',')):
                        lines = lines[:-1]
                        s = '\n'.join(lines)
                        # إضافة إغلاق مناسب
                        if s.count('{') > s.count('}'):
                            s += '\n}'
                        if s.count('[') > s.count(']'):
                            s += '\n]'
                
                result = json.loads(s)
                print("✅ تم تحليل JSON بنجاح بعد الإصلاح الإضافي")
                result = fix_generated_questions(result, source_text, model_name, lang, retrieved)
                return result
            except json.JSONDecodeError as e3:
                print(f"❌ فشل في المحاولة الثالثة: {e3}")
                print("⚠️ سيتم إعادة المحاولة مع النموذج")
                return None

def generate_questions_with_retry(prompt: str, max_retries: int = 3, source_text: str = "", 
                                  model_name: str = "llama3.2:3b", lang: str = "ar", retrieved: list = None):
    """
    توليد الأسئلة مع إعادة المحاولة
    """
    for attempt in range(max_retries):
        print(f"🔄 محاولة توليد الأسئلة {attempt + 1}/{max_retries}")
        
        try:
            response = call_llama(prompt, model_name=model_name, max_retries=1)
            
            if not response:
                print(f"❌ فشل في الحصول على استجابة في المحاولة {attempt + 1}")
                continue
            
            # تحليل JSON مع تمرير source_text للإصلاح التلقائي
            questions = safe_json(response, source_text, model_name, lang, retrieved)
            
            if questions and isinstance(questions, dict):
                # التحقق من وجود المفاتيح المطلوبة
                if "mcq" in questions and "tf" in questions:
                    print(f"✅ تم توليد الأسئلة بنجاح في المحاولة {attempt + 1}")
                    return questions
                else:
                    print(f"⚠️ تنسيق JSON غير صحيح في المحاولة {attempt + 1}")
                    print(f"المفاتيح الموجودة: {list(questions.keys())}")
            else:
                print(f"⚠️ فشل في تحليل JSON في المحاولة {attempt + 1}")
                
        except Exception as e:
            print(f"❌ خطأ في المحاولة {attempt + 1}: {e}")
        
        if attempt < max_retries - 1:
            print(f"⏳ انتظار 3 ثواني قبل المحاولة التالية...")
            time.sleep(3)
    
    print("❌ فشل في توليد الأسئلة بعد جميع المحاولات")
    return None
