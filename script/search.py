#!/usr/local/bin/python3.13
# -*- coding: utf-8 -*-
"""
语义搜索：输入查询文本 → bge 编码 → ChromaDB 检索 → （可选 reranker 精排）。

用法：
  python3 script/search.py "背影"
  python3 script/search.py "蹒跚" --top 5
  python3 script/search.py "怎么导入课文" --lesson "世说新语精读"
  python3 script/search.py "父爱细节" --rerank   # 粗召回20→精排top-5
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from bge.encode import Encoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="语义搜索检索单元")
    p.add_argument("query", type=str, help="查询文本")
    p.add_argument("--top", type=int, default=5, help="返回条数（默认5）")
    p.add_argument("--db", type=Path,
                   default=PROJECT_ROOT / "data" / "vecdb",
                   help="ChromaDB 路径")
    p.add_argument("--model-dir", type=str, default=None,
                   help="bge 模型目录")
    p.add_argument("--lesson", type=str, default=None,
                   help="限定课文名（如 '世说新语精读'）")
    p.add_argument("--rerank", action="store_true",
                   help="启用 bge-reranker 精排（粗召回20→精排top-N）")
    p.add_argument("--recall", type=int, default=20,
                   help="粗召回数（与 --rerank 配合，默认20）")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="显示完整 text")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import chromadb

    client = chromadb.PersistentClient(path=str(args.db))
    col = client.get_collection("teacher_units")

    enc = Encoder(args.model_dir)
    print(f"查询: {args.query}\n")
    q_vec = enc.encode([args.query])[0]

    where = {"lesson": args.lesson} if args.lesson else None

    if args.rerank:
        # 粗召回 top-20
        recall = min(args.recall, col.count())
        res = col.query(query_embeddings=[q_vec], n_results=recall, where=where)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        ids = res["ids"][0]

        # 精排
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", cache_folder=args.model_dir)
        pairs = [[args.query, doc] for doc in docs]
        scores = reranker.predict(pairs)

        ranked = sorted(zip(scores, ids, docs, metas), reverse=True)[:args.top]
        for rank, (score, doc_id, doc, meta) in enumerate(ranked):
            _print_result(rank + 1, doc_id, doc, meta, score, args.verbose)
    else:
        res = col.query(query_embeddings=[q_vec], n_results=args.top, where=where)
        for i, (doc_id, doc, meta, dist) in enumerate(
            zip(res["ids"][0], res["documents"][0],
                res["metadatas"][0], res["distances"][0])
        ):
            _print_result(i + 1, doc_id, doc, meta, 1 - dist, args.verbose)


def _print_result(
    rank: int, doc_id: str, doc: str,
    meta: dict, score: float, verbose: bool,
) -> None:
    print(f"--- #{rank}  {doc_id}  ({'精排' if score > 1 else '相似度'} {score:.4f})")
    print(f"    课文: {meta.get('lesson', '')}")
    print(f"    标签: {meta.get('tags', '')}")
    print(f"    摘要: {meta.get('summary', '')}")
    text = doc if verbose else doc[:120].replace("\n", " ")
    suffix = "" if verbose or len(doc) <= 120 else "……"
    print(f"    原文: {text}{suffix}")
    print()


if __name__ == "__main__":
    main()
