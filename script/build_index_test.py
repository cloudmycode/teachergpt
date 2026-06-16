#!/usr/local/bin/python3.13
# -*- coding: utf-8 -*-
"""
向量索引测试脚本：验证 build_index.py 的入库效果。

测试项：
1. 向量库状态（单元数、集合信息）
2. 元数据完整性（lesson, tags, summary, paras）
3. 语义检索效果（给定 query，返回 top-k 结果）
4. 元数据过滤检索（按 lesson 过滤）

用法：
  python3 script/build_index_test.py                    # 完整测试
  python3 script/build_index_test.py --query "父爱"     # 指定查询
  python3 script/build_index_test.py --top-k 10         # 返回前 10 条
  python3 script/build_index_test.py --lesson "背影"    # 按课文过滤
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "vecdb"
COLLECTION_NAME = "teacher_units"


def test_connection(client):
    """测试连接和集合状态。"""
    print("\n" + "=" * 60)
    print("1. 向量库连接测试")
    print("=" * 60)
    
    try:
        col = client.get_or_create_collection(COLLECTION_NAME)
        count = col.count()
        print(f"  ✓ 集合 '{COLLECTION_NAME}' 已就绪")
        print(f"  ✓ 总单元数: {count}")
        return col, count
    except Exception as e:
        print(f"  ✗ 连接失败: {e}")
        return None, 0


def test_metadata(col):
    """测试元数据完整性。"""
    print("\n" + "=" * 60)
    print("2. 元数据完整性测试")
    print("=" * 60)
    
    if col.count() == 0:
        print("  ⚠ 向量库为空，跳过")
        return
    
    # 取几条样本检查元数据
    sample_size = min(5, col.count())
    results = col.get(limit=sample_size)
    
    required_fields = ["lesson", "tags", "summary", "paras"]
    stats = {f: 0 for f in required_fields}
    
    print(f"  样本数: {sample_size}")
    print()
    
    for i, (meta, doc) in enumerate(zip(results["metadatas"], results["documents"])):
        print(f"  --- 样本 {i + 1} ---")
        print(f"  unit_id: {results['ids'][i]}")
        
        for field in required_fields:
            value = meta.get(field, "")
            if value:
                stats[field] += 1
                status = "✓"
            else:
                status = "✗"
            # 截断显示
            display = value[:50] + "..." if len(str(value)) > 50 else value
            print(f"  {status} {field}: {display}")
        
        # 显示文本片段
        text_preview = doc[:80] + "..." if len(doc) > 80 else doc
        print(f"  文本: {text_preview}")
        print()
    
    # 统计汇总
    print("  --- 完整性统计 ---")
    for field, count in stats.items():
        ratio = count / sample_size * 100
        print(f"  {field}: {count}/{sample_size} ({ratio:.0f}%)")


def test_semantic_search(col, query: str, top_k: int = 5):
    """测试语义检索效果。"""
    print("\n" + "=" * 60)
    print("3. 语义检索测试")
    print("=" * 60)
    
    if col.count() == 0:
        print("  ⚠ 向量库为空，跳过")
        return
    
    print(f"  Query: {query}")
    print(f"  Top-K: {top_k}")
    print()
    
    # 编码 query
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-large-zh-v1.5", cache_folder=str(PROJECT_ROOT / "bge" / "models"))
        query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    except Exception as e:
        print(f"  ✗ BGE 模型加载失败: {e}")
        print("  跳过语义检索测试")
        return
    
    # 检索
    try:
        results = col.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"  ✓ 检索成功，返回 {len(results['ids'][0])} 条结果")
        print()
        
        for i, (uid, doc, meta, dist) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            print(f"  [{i + 1}] 相似度: {1 - dist:.4f}")
            print(f"      unit_id: {uid}")
            print(f"      lesson: {meta.get('lesson', 'N/A')}")
            print(f"      tags: {meta.get('tags', 'N/A')}")
            text_preview = doc[:100] + "..." if len(doc) > 100 else doc
            print(f"      文本: {text_preview}")
            print()
            
    except Exception as e:
        print(f"  ✗ 检索失败: {e}")


def test_filtered_search(col, lesson: str, top_k: int = 5):
    """测试按 lesson 过滤检索。"""
    print("\n" + "=" * 60)
    print("4. 元数据过滤检索测试")
    print("=" * 60)
    
    if not lesson:
        print("  ⚠ 未指定 --lesson 参数，跳过")
        return
    
    if col.count() == 0:
        print("  ⚠ 向量库为空，跳过")
        return
    
    print(f"  过滤条件: lesson = {lesson}")
    print(f"  Top-K: {top_k}")
    print()
    
    try:
        # 先查该 lesson 有多少条
        all_results = col.get(where={"lesson": lesson})
        print(f"  该课文共 {len(all_results['ids'])} 条单元")
        
        if len(all_results['ids']) == 0:
            print(f"  ⚠ 未找到 lesson='{lesson}' 的数据")
            return
        
        # 显示前几条
        show_count = min(top_k, len(all_results['ids']))
        for i in range(show_count):
            uid = all_results['ids'][i]
            meta = all_results['metadatas'][i]
            doc = all_results['documents'][i]
            print(f"  [{i + 1}] {uid}")
            print(f"      tags: {meta.get('tags', 'N/A')}")
            text_preview = doc[:80] + "..." if len(doc) > 80 else doc
            print(f"      文本: {text_preview}")
            print()
            
    except Exception as e:
        print(f"  ✗ 过滤检索失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="向量索引测试")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB), help="ChromaDB 路径")
    parser.add_argument("--query", type=str, default="父亲爬月台买橘子", help="测试查询语句")
    parser.add_argument("--top-k", type=int, default=5, help="返回前 K 条结果")
    parser.add_argument("--lesson", type=str, default=None, help="按课文名过滤")
    args = parser.parse_args()
    
    print(f"向量库路径: {args.db}")
    
    # 导入 chromadb
    try:
        import chromadb
    except ImportError:
        print("✗ chromadb 未安装，请运行: pip3 install chromadb")
        sys.exit(1)
    
    # 连接
    client = chromadb.PersistentClient(path=args.db)
    col, count = test_connection(client)
    
    if col is None:
        print("\n测试中止")
        sys.exit(1)
    
    # 运行测试
    test_metadata(col)
    test_semantic_search(col, args.query, args.top_k)
    test_filtered_search(col, args.lesson, args.top_k)
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
