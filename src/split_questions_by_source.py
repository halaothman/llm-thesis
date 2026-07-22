import json
import os
from pathlib import Path
from collections import defaultdict

def clean_filename(filename):
    """تنظيف اسم الملف لاستخدامه في التسمية"""
    # إزالة الامتداد
    name = Path(filename).stem
    # إزالة المسافات والرموز الخاصة
    name = name.replace(" ", "_").replace(".", "_")
    return name

def split_questions_by_source(input_file, output_dir="outputs"):
    """
    فصل الأسئلة من ملف JSON شامل إلى ملفات فرعية حسب source.file
    
    Args:
        input_file: مسار الملف الشامل
        output_dir: مجلد الإخراج
    """
    print(f"🔍 قراءة: {input_file}")
    
    # قراءة الملف الشامل
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # تجميع الأسئلة حسب source.file
    questions_by_source = defaultdict(list)
    
    for question in data.get("questions", []):
        source_file = question.get("source", {}).get("file", "unknown")
        questions_by_source[source_file].append(question)
    
    # استخراج معلومات من اسم الملف
    input_filename = Path(input_file).stem  # مثال: questions_llama_vanilla
    
    # معلومات metadata من الملف الأصلي
    original_metadata = data.get("metadata", {})
    model = original_metadata.get("models_used", ["unknown"])[0] if original_metadata.get("models_used") else "unknown"
    
    # تحديد الطريقة (vanilla أو rag)
    method = "vanilla" if "vanilla" in input_filename else "rag" if "rag" in input_filename else "unknown"
    
    print(f"📊 عدد الملفات المصدرية: {len(questions_by_source)}")
    
    # إنشاء ملف فرعي لكل مصدر
    created_files = []
    for source_file, questions in questions_by_source.items():
        # تنظيف اسم الملف المصدر
        clean_source = clean_filename(source_file)
        
        # اسم الملف الفرعي الجديد
        output_filename = f"{input_filename}_{clean_source}.json"
        output_path = Path(output_dir) / output_filename
        
        # إنشاء metadata للملف الفرعي
        sub_metadata = {
            "version": "1.0",
            "parent_file": Path(input_file).name,
            "source_file": source_file,
            "total_questions": len(questions),
            "model": model,
            "method": method,
            "created_from": "split_by_source"
        }
        
        # بناء البيانات
        output_data = {
            "metadata": sub_metadata,
            "questions": questions
        }
        
        # حفظ الملف
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        created_files.append(output_filename)
        print(f"  ✅ تم إنشاء: {output_filename} ({len(questions)} أسئلة)")
    
    return created_files

def main():
    """معالجة جميع الملفات الشاملة"""
    output_dir = "outputs"
    
    # قائمة الملفات الشاملة
    main_files = [
        "outputs/questions_llama_vanilla.json",
        "outputs/questions_llama_rag.json",
        "outputs/questions_qwen_vanilla.json",
        "outputs/questions_qwen_rag.json"
    ]
    
    print("=" * 60)
    print("🚀 بدء فصل الأسئلة حسب المصدر")
    print("=" * 60)
    print()
    
    all_created_files = []
    
    for main_file in main_files:
        if os.path.exists(main_file):
            print(f"📂 معالجة: {main_file}")
            created = split_questions_by_source(main_file, output_dir)
            all_created_files.extend(created)
            print()
        else:
            print(f"⚠️ الملف غير موجود: {main_file}")
            print()
    
    print("=" * 60)
    print("✨ تم الانتهاء!")
    print(f"📁 عدد الملفات المنشأة: {len(all_created_files)}")
    print("=" * 60)
    print()
    print("📋 الملفات المنشأة:")
    for filename in sorted(all_created_files):
        print(f"  • {filename}")

if __name__ == "__main__":
    main()


