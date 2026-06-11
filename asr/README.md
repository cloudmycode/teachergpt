# asr/

独立的音频转写小工具集，基于 `faster-whisper`。

## 目录

```
asr/
  upgrade_model.py    下载/升级 faster-whisper 模型到 models/faster-whisper-<size>/
  transcribe.py       用本地模型把音频转成文本，输出到 ./output/
  output/             转写结果默认输出目录（自动创建）
```

模型权重统一放在 `../models/` 下，供 `diaryofawimpykit/extract_word_timeline_raw.py` 复用。

## 依赖

```bash
pip install faster-whisper huggingface_hub
```

## 用法

升级 / 下载模型：

```bash
cd /Users/wang/Project/video_en/asr

# 默认：只查远端 commit 跟本地对比，告诉你有没有更新
python3.13 upgrade_model.py

# 首次下载 / 执行更新（脚本会打印这条命令）
python3.13 upgrade_model.py --download

# 已下载过：从 ~/.cache/huggingface 复制（最快）
python3.13 upgrade_model.py --download --from-cache

# 覆盖式更新：清空目录后重下
python3.13 upgrade_model.py --download --force

# 换模型
python3.13 upgrade_model.py --repo Systran/faster-whisper-small \
    --to models/faster-whisper-small
```

更新机制：用 `HfApi.repo_info()` 拿远端 `main` 分支最新 commit sha，存在目标目录的 `.revision` 文件里；下次跑默认命令时会对比两个 sha，给出"已是最新 / 有更新 / 未下载"的状态，并打印对应的下载命令。

转写音频：

```bash
# 最小用法：输出到 ./output/<stem>.txt
python3.13 transcribe.py /path/to/audio.mp3

# 自定义输出目录 / 同时输出 json 时间戳 / 切语言
python3.13 transcribe.py /path/to/audio.mp3 \
    --out ./output \
    --json \
    --lang en

# 用其他模型
python3.13 transcribe.py /path/to/audio.mp3 \
    --model models/faster-whisper-small
```

## 环境变量

- `WHISPER_MODEL`：模型路径或 HF 名（`transcribe.py` 用，默认指向 `models/faster-whisper-medium`）
- `HF_ENDPOINT`：HuggingFace 镜像（`upgrade_model.py` 用，国内可设 `https://hf-mirror.com`）
- `HF_TOKEN`：私有模型 token

## 输出文件命名

`/path/to/foo.mp3` → `output/foo.txt`（每行格式：`{start}s → {end}s | {text}`，带秒级时间戳）。

加 `--json` 会同时输出 `output/foo.json`（段级 + 词级时间戳）。