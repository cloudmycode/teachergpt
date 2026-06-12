#!/usr/local/bin/python3.13
# -*- coding: utf-8 -*-
"""
BGE 模型编码器：加载模型，提供文本向量化。

用法：
  from bge.encode import Encoder
  enc = Encoder(model_dir="./bge/models")
  vecs = enc.encode(["文本1", "文本2"])  # -> list[list[float]]

也可直接命令行测试：
  python3 bge/encode.py --text "背影"
"""

import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "models"


class Encoder:
    """BGE 编码器，加载一次即可反复使用。"""

    def __init__(self, model_dir: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(
            "BAAI/bge-large-zh-v1.5",
            cache_folder=model_dir or str(DEFAULT_MODEL_DIR),
        )

    def encode(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """批量编码，返回归一化向量。"""
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
        ).tolist()


# ------------------------------------------------------------------- CLI 测试

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="BGE 编码器测试")
    p.add_argument("--text", type=str, required=True, help="待编码文本")
    p.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR),
                   help="模型缓存目录")
    args = p.parse_args()
    enc = Encoder(args.model_dir)
    vec = enc.encode([args.text])[0]
    print(f"文本: {args.text}")
    print(f"向量维度: {len(vec)}")
    print(f"前5维: {[round(v, 4) for v in vec[:5]]}")
