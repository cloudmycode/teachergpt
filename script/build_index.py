#!/usr/local/bin/python3.13
# -*- coding: utf-8 -*-
"""
向量索引构建：读 A.2 JSONL → bge 编码 → 写入 ChromaDB。

输入：data/chinese_units/<课程>/*.jsonl
      每行 A.2 检索单元 JSON（unit_id, text, lesson, tags, summary, source_refs）

输出：data/vecdb/  ChromaDB 持久化向量库
      collection: teacher_units
      - documents: text（讲课原文）
      - embeddings: bge-large-zh-v1.5 向量
      - metadatas: lesson, t_start, tags, summary, paras

依赖：
  pip install chromadb sentence-transformers

用法：
  python3 script/build_index.py                    # 增量索引
  python3 script/build_index.py --rebuild          # 清库重建
  python3 script/build_index.py --dry-run          # 检查但不入库
  python3 script/build_index.py --model-dir ./bge/models  # 指定模型目录
"""

import argparse
import json
import sys
import time
from pathlib import Path

# bge 编码器在 bge/encode.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bge.encode import Encoder

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SRC = PROJECT_ROOT / "data" / "chinese_units"
DEFAULT_DB = PROJECT_ROOT / "data" / "vecdb"
COLLECTION_NAME = "teacher_units"


# ------------------------------------------------------------------- 数据加载

def load_units(src: Path) -> list[dict]:
    """读所有 JSONL，返回 unit dict 列表。"""
    units: list[dict] = []
    for f in sorted(src.rglob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            units.append(json.loads(line))
    return units


# ------------------------------------------------------------------- 入库

def build_index(
    units: list[dict],
    db_path: Path,
    model_dir: "str | None" = None,
    batch_size: int = 64,
    dry_run: bool = False,
    rebuild: bool = False,
) -> str:
    """构建/更新索引。返回状态信息。"""
    import chromadb

    if dry_run:
        return f"dry: {len(units)} 个单元待索引"

    print("  加载 BGE 编码器 ...")
    encoder = Encoder(model_dir)

    client = chromadb.PersistentClient(path=str(db_path))

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 查已有 ID，跳过已入库的单元
    existing = set()
    if not rebuild:
        try:
            existing = set(col.get()["ids"])
        except Exception:
            pass

    new_units = [u for u in units if u["unit_id"] not in existing]
    if not new_units:
        return "无新单元"

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    for u in new_units:
        ids.append(u["unit_id"])
        # 实体+摘要放前面增强关键词密度，解决人名查询信号弱的问题
        enrich = " ".join(u.get("entities", []) + u.get("tags", []))
        texts.append(enrich + " " + u.get("summary", "") + " " + u["text"])
        paras = [
            str(r["para"])
            for r in u.get("source_refs", [])
            if isinstance(r, dict) and "para" in r
        ]
        metadatas.append({
            "lesson": u["lesson"],
            "t_start": u["t_start"],
            "tags": ",".join(u.get("tags", [])),
            "summary": u.get("summary", ""),
            "paras": ",".join(paras),
        })

    print(f"  编码 {len(new_units)} 个单元 ...")
    t_enc = time.time()
    embeddings = encoder.encode(texts, batch_size)
    dt_enc = int(time.time() - t_enc)

    print(f"  写入 ChromaDB ...")
    t_write = time.time()
    col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    dt_write = int(time.time() - t_write)

    total_count = col.count()
    return (
        f"+{len(new_units)} 单元（总 {total_count}）"
        f"  编码 {dt_enc}s  写入 {dt_write}s"
    )


# ------------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="向量索引构建：读 A.2 JSONL → ChromaDB")
    p.add_argument("--src", type=Path, default=DEFAULT_SRC, help="A.2 检索单元目录")
    p.add_argument("--db", type=Path, default=DEFAULT_DB, help="ChromaDB 持久化路径")
    p.add_argument("--model-dir", type=str, default=None, help="bge 模型缓存目录")
    p.add_argument("--batch-size", type=int, default=64, help="编码 batch size")
    p.add_argument("--rebuild", action="store_true", help="清库重建")
    p.add_argument("--dry-run", action="store_true", help="不实际索引，仅检查")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = args.src.resolve()

    if not src.exists():
        print(f"✗ 源目录不存在: {src}")
        sys.exit(1)

    print(f"源目录  : {src}")
    print(f"向量库  : {args.db}")
    units = load_units(src)
    if not units:
        print("没有找到检索单元。")
        return

    print(f"待索引  : {len(units)} 个单元"
          f"{'（重建）' if args.rebuild else ''}"
          f"{'（dry-run）' if args.dry_run else ''}\n")

    msg = build_index(
        units, args.db, args.model_dir, args.batch_size, args.dry_run, args.rebuild,
    )
    print(f"完成: {msg}")


if __name__ == "__main__":
    main()
