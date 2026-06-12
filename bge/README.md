# bge — BGE Embedding 模型

基于 `BAAI/bge-large-zh-v1.5`，提供中文文本向量化。

## 文件

- `download_model.py` — 下载/更新模型
- `encode.py` — 编码器，加载模型并提供 `Encoder.encode()` 接口
- `models/` — 模型文件缓存目录

## 快速开始

```bash
# 1. 下载模型
python3 bge/download_model.py --download

# 2. 测试编码
python3 bge/encode.py --text "背影"
```

## 用法

```bash
# 模型管理
python3 bge/download_model.py                       # 检查更新
python3 bge/download_model.py --download            # 下载/更新
python3 bge/download_model.py --download --force    # 清空重下

# 编码测试
python3 bge/encode.py --text "背影"
python3 bge/encode.py --text "蹒跚" --model-dir ./bge/models
```

## 在代码中使用

```python
from bge.encode import Encoder

enc = Encoder(model_dir="./bge/models")
vecs = enc.encode(["背影", "蹒跚地走到铁道边"])
```

## 调度方

- `script/build_index.py` — 读 A.2 JSONL → 调 `bge.encode.Encoder` → 写入 ChromaDB
  - 额外依赖：`pip3 install chromadb`

## 依赖

```bash
pip install sentence-transformers huggingface-hub
```
