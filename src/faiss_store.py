"""إنشاء فهرس FAISS والبحث فيه مع metadata بصيغة JSONL."""
import json
import os

import faiss
import numpy as np

from .embeddings import embed_texts


def clear_index(index_path: str, meta_path: str) -> None:
    """حذف ملفات الفهرس والميتاداتا إن وُجدت."""
    for path in (index_path, meta_path):
        if path and os.path.exists(path):
            os.remove(path)


def build_index(
    index_path: str,
    meta_path: str,
    records: list[dict],
    *,
    append: bool = True,
) -> None:
    """
    بناء فهرس FAISS من قائمة سجلات [{id, text, metadata}].

    append=False: يمسح الفهرس الحالي ثم يكتب من جديد.
    append=True: يضيف إلى الفهرس الموجود (أو ينشئ فهرساً جديداً إن لم يوجد).
    """
    if not records:
        return

    if not append:
        clear_index(index_path, meta_path)

    texts = [r["text"] for r in records]
    vecs = embed_texts(texts, is_query=False).astype("float32")
    ids = np.array([int(r["id"]) for r in records], dtype="int64")

    index = None
    if append and os.path.exists(index_path) and os.path.getsize(index_path) > 0:
        try:
            index = faiss.read_index(index_path)
        except Exception as e:
            print(f"خطأ في قراءة الفهرس: {e}")

    if index is None:
        d = vecs.shape[1]
        index = faiss.IndexIDMap(faiss.IndexFlatIP(d))

    index.add_with_ids(vecs, ids)

    index_dir = os.path.dirname(index_path)
    if index_dir:
        os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, index_path)

    meta_dir = os.path.dirname(meta_path)
    if meta_dir:
        os.makedirs(meta_dir, exist_ok=True)
    with open(meta_path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def search(index_path: str, query_vec: np.ndarray, top_k: int = 5):
    """البحث في فهرس FAISS — يُرجع (scores, indices)."""
    if not os.path.exists(index_path):
        return np.array([]), np.array([])

    try:
        if os.path.getsize(index_path) == 0:
            return np.array([]), np.array([])

        index = faiss.read_index(index_path)

        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        d, i = index.search(query_vec.astype("float32"), top_k)
        if d.shape[0] > 0 and i.shape[0] > 0:
            return d[0], i[0]
        return np.array([]), np.array([])
    except Exception as e:
        print(f"خطأ في البحث في الفهرس: {e}")
        return np.array([]), np.array([])


def load_meta(meta_path: str) -> dict:
    """تحميل الميتاداتا من JSONL إلى قاموس id → سجل."""
    meta = {}
    if not os.path.exists(meta_path):
        return meta
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                meta[r["id"]] = r
            except json.JSONDecodeError:
                continue
    return meta
