"""
واجهة المستخدم لحل الأسئلة وتصحيح الإجابات
Quiz UI and Grading System
"""

import streamlit as st
from typing import Dict, List, Any, Tuple
import json


def render_quiz(questions: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    """
    عرض الأسئلة كواجهة تفاعلية للمستخدم
    
    Args:
        questions: قاموس يحتوي على الأسئلة المولدة
        prefix: بادئة لتمييز الأسئلة (vanilla_ أو rag_)
    
    Returns:
        Dict[str, str]: قاموس يحتوي على إجابات المستخدم
    """
    if not questions or not isinstance(questions, dict):
        st.error("لا توجد أسئلة لعرضها")
        return {}
    
    # إضافة CSS مخصص للمحاذاة
    st.markdown("""
    <style>
    .stRadio > div > label {
        text-align: right !important;
        direction: rtl !important;
    }
    .stRadio > div > label > div {
        text-align: right !important;
        direction: rtl !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    user_answers = {}
    
    # عرض أسئلة الاختيار من متعدد
    if "mcq" in questions and questions["mcq"]:
        st.markdown("<div dir='rtl' style='text-align: right;'><h3> أسئلة الاختيار من متعدد</h3></div>", unsafe_allow_html=True)
        
        for i, mcq in enumerate(questions["mcq"]):
            if not isinstance(mcq, dict):
                continue
            
            # دعم التنسيقين: "question" أو "q"
            question_text = mcq.get("question") or mcq.get("q", "")
            if not question_text:
                continue
                
            question_key = f"{prefix}mcq_{i}"
            
            st.markdown(f"<div dir='rtl' style='text-align: right;'><strong>السؤال {i+1}:</strong> {question_text}</div>", unsafe_allow_html=True)
            
            # عرض الخيارات
            options = mcq.get("options", [])
            if len(options) != 4:
                st.warning(f"السؤال {i+1} يجب أن يحتوي على 4 خيارات (موجود {len(options)})")
                continue
            
            # خيارات الراديو
            st.markdown("<div dir='rtl' style='text-align: right;'><strong>اختر الإجابة:</strong></div>", unsafe_allow_html=True)
            
            # عرض الخيارات مع تنسيق محسن ومحاذاة لليمين
            answer = st.radio(
                "",
                options=options,
                key=question_key,
                index=None,
                format_func=lambda x: f" {x}"  # رمز دائرة بدون HTML
            )
            
            if answer:
                user_answers[question_key] = answer
            
            st.divider()
    
    # عرض أسئلة صح/خطأ
    if "tf" in questions and questions["tf"]:
        st.markdown("<div dir='rtl' style='text-align: right;'><h3> أسئلة صح/خطأ</h3></div>", unsafe_allow_html=True)
        
        for i, tf in enumerate(questions["tf"]):
            if not isinstance(tf, dict):
                continue
            
            # دعم التنسيقين: "question" أو "q"
            question_text = tf.get("question") or tf.get("q", "")
            if not question_text:
                continue
                
            question_key = f"{prefix}tf_{i}"
            
            st.markdown(f"<div dir='rtl' style='text-align: right;'><strong>السؤال {i+1}:</strong> {question_text}</div>", unsafe_allow_html=True)
            
            # خيارات صح/خطأ
            tf_options = ["صح", "خطأ"]
            st.markdown("<div dir='rtl' style='text-align: right;'><strong>اختر الإجابة:</strong></div>", unsafe_allow_html=True)
            answer = st.radio(
                "",
                options=tf_options,
                key=question_key,
                index=None,
                format_func=lambda x: f" {x}"  # رمز دائرة بدون HTML
            )
            
            if answer:
                user_answers[question_key] = answer
            
            st.divider()
    
    return user_answers


def grade(user_answers: Dict[str, str], questions: Dict[str, Any], prefix: str = "") -> Tuple[int, int, List[str]]:
    """
    تصحيح إجابات المستخدم
    
    Args:
        user_answers: إجابات المستخدم
        questions: الأسئلة الأصلية مع الإجابات الصحيحة
        prefix: بادئة الأسئلة
    
    Returns:
        Tuple[int, int, List[str]]: (الدرجة، المجموع، تفاصيل التصحيح)
    """
    if not user_answers or not questions:
        return 0, 0, [" لا توجد إجابات أو أسئلة للتصحيح"]
    
    score = 0
    total = 0
    results = []
    
    # تصحيح أسئلة الاختيار من متعدد
    if "mcq" in questions and questions["mcq"]:
        for i, mcq in enumerate(questions["mcq"]):
            if not isinstance(mcq, dict):
                continue
                
            question_key = f"{prefix}mcq_{i}"
            total += 1
            
            user_answer = user_answers.get(question_key, "")
            # دعم التنسيقين: "correct_answer" أو "answer"
            correct_answer = mcq.get("correct_answer") or mcq.get("answer", "")
            
            if user_answer == correct_answer:
                score += 1
                results.append(f" السؤال {i+1} (MCQ): صحيح - {user_answer}")
            else:
                results.append(f" السؤال {i+1} (MCQ): خطأ - إجابتك: {user_answer}, الصحيح: {correct_answer}")
    
    # تصحيح أسئلة صح/خطأ
    if "tf" in questions and questions["tf"]:
        for i, tf in enumerate(questions["tf"]):
            if not isinstance(tf, dict):
                continue
                
            question_key = f"{prefix}tf_{i}"
            total += 1
            
            user_answer = user_answers.get(question_key, "")
            # دعم التنسيقين: "correct_answer" أو "answer" (مع تحويل boolean إلى نص)
            correct_answer_raw = tf.get("correct_answer") or tf.get("answer", "")
            
            # تحويل boolean إلى نص عربي
            if isinstance(correct_answer_raw, bool):
                correct_answer = "صح" if correct_answer_raw else "خطأ"
            else:
                correct_answer = str(correct_answer_raw)
            
            if user_answer == correct_answer:
                score += 1
                results.append(f" السؤال {i+1} (T/F): صحيح - {user_answer}")
            else:
                results.append(f" السؤال {i+1} (T/F): خطأ - إجابتك: {user_answer}, الصحيح: {correct_answer}")
    
    return score, total, results


def display_quiz_results(score: int, total: int, results: List[str]):
    """
    عرض نتائج الاختبار
    
    Args:
        score: الدرجة المحققة
        total: المجموع الكلي
        results: تفاصيل التصحيح
    """
    if total == 0:
        st.error("لا توجد أسئلة للتصحيح")
        return
    
    # حساب النسبة المئوية
    percentage = (score / total) * 100
    
    # عرض النتيجة الرئيسية
    if percentage >= 90:
        st.success(f"ممتاز! نتيجتك: {score}/{total} ({percentage:.1f}%)")
    elif percentage >= 80:
        st.success(f"جيد جداً! نتيجتك: {score}/{total} ({percentage:.1f}%)")
    elif percentage >= 70:
        st.info(f"جيد! نتيجتك: {score}/{total} ({percentage:.1f}%)")
    elif percentage >= 60:
        st.warning(f"مقبول! نتيجتك: {score}/{total} ({percentage:.1f}%)")
    else:
        st.error(f"ضعيف! نتيجتك: {score}/{total} ({percentage:.1f}%)")
    
    # عرض تفاصيل التصحيح
    with st.expander("تفاصيل التصحيح"):
        for result in results:
            st.write(result)
    
    # عرض إحصائيات إضافية
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("الدرجة المحققة", score)
    
    with col2:
        st.metric("النسبة المئوية", f"{percentage:.1f}%")
    
    with col3:
        st.metric("الأسئلة الخاطئة", total - score)


def validate_questions_format(questions: Dict[str, Any]) -> bool:
    """
    التحقق من صحة تنسيق الأسئلة
    
    Args:
        questions: الأسئلة المراد التحقق منها
    
    Returns:
        bool: True إذا كان التنسيق صحيح
    """
    if not questions or not isinstance(questions, dict):
        return False
    
    # التحقق من وجود أسئلة MCQ
    if "mcq" in questions:
        if not isinstance(questions["mcq"], list):
            return False
        
        for i, mcq in enumerate(questions["mcq"]):
            if not isinstance(mcq, dict):
                return False
            
            # دعم التنسيقين: "question" أو "q"
            question_field = mcq.get("question") or mcq.get("q")
            if not question_field:
                st.warning(f"السؤال MCQ {i+1} يفتقر إلى حقل السؤال")
                return False
            
            # التحقق من وجود options
            if "options" not in mcq:
                st.warning(f"السؤال MCQ {i+1} يفتقر إلى حقل الخيارات")
                return False
            
            # التحقق من وجود الإجابة الصحيحة (دعم التنسيقين)
            if "correct_answer" not in mcq and "answer" not in mcq:
                st.warning(f"السؤال MCQ {i+1} يفتقر إلى الإجابة الصحيحة")
                return False
            
            # التحقق من وجود 4 خيارات
            if len(mcq.get("options", [])) != 4:
                st.warning(f"السؤال MCQ {i+1} يجب أن يحتوي على 4 خيارات")
                return False
    
    # التحقق من وجود أسئلة True/False
    if "tf" in questions:
        if not isinstance(questions["tf"], list):
            return False
        
        for i, tf in enumerate(questions["tf"]):
            if not isinstance(tf, dict):
                return False
            
            # دعم التنسيقين: "question" أو "q"
            question_field = tf.get("question") or tf.get("q")
            if not question_field:
                st.warning(f"السؤال T/F {i+1} يفتقر إلى حقل السؤال")
                return False
            
            # التحقق من وجود الإجابة الصحيحة (دعم التنسيقين)
            if "correct_answer" not in tf and "answer" not in tf:
                st.warning(f"السؤال T/F {i+1} يفتقر إلى الإجابة الصحيحة")
                return False
            
            # التحقق من أن الإجابة الصحيحة صحيحة (دعم التنسيقين)
            correct_answer = tf.get("correct_answer") or tf.get("answer", "")
            if isinstance(correct_answer, bool):
                # إذا كانت boolean، فهي صحيحة
                pass
            elif correct_answer not in ["صح", "خطأ", "true", "false", True, False]:
                st.warning(f"السؤال T/F {i+1} يجب أن تكون الإجابة الصحيحة 'صح' أو 'خطأ' أو boolean")
                return False
    
    return True


def add_missing_correct_answers(questions: Dict[str, Any]) -> Dict[str, Any]:
    """
    إضافة الإجابات الصحيحة المفقودة (للأسئلة المولدة بدون إجابات صحيحة)
    
    Args:
        questions: الأسئلة المراد إضافة الإجابات لها
    
    Returns:
        Dict[str, Any]: الأسئلة مع الإجابات المضافة
    """
    if not questions or not isinstance(questions, dict):
        return questions
    
    # نسخ الأسئلة لتجنب تعديل الأصلية
    updated_questions = questions.copy()
    
    # إضافة الإجابات الصحيحة لأسئلة MCQ
    if "mcq" in updated_questions:
        for i, mcq in enumerate(updated_questions["mcq"]):
            if isinstance(mcq, dict) and "correct_answer" not in mcq and "answer" not in mcq:
                # اختيار أول خيار كإجابة صحيحة (يمكن تحسين هذا لاحقاً)
                options = mcq.get("options", [])
                if options:
                    updated_questions["mcq"][i]["correct_answer"] = options[0]
                    st.info(f"تم إضافة إجابة صحيحة للسؤال MCQ {i+1}: {options[0]}")
    
    # إضافة الإجابات الصحيحة لأسئلة True/False
    if "tf" in updated_questions:
        for i, tf in enumerate(updated_questions["tf"]):
            if isinstance(tf, dict) and "correct_answer" not in tf and "answer" not in tf:
                # اختيار "صح" كإجابة افتراضية (يمكن تحسين هذا لاحقاً)
                updated_questions["tf"][i]["correct_answer"] = "صح"
                st.info(f"تم إضافة إجابة صحيحة للسؤال T/F {i+1}: صح")
    
    return updated_questions

