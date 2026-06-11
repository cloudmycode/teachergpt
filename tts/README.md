# tts/

独立的文本转语音小工具集，基于 `edge-tts`（微软 Edge 免费云端 TTS）

## 目录

```
tts/
  synthesize.py      文本 → mp3，自动剥离 transcribe.py 输出的时间戳前缀
  output/            默认输出目录（自动创建）
```

## 依赖

```bash
pip install edge-tts
```

## 用法

```bash
cd /Users/wang/Project/video_en/tts

# 最小用法：把 ./output/<stem>.mp3 写出来
python3.13 synthesize.py path/to/text.txt

# 切音色 / 语速
python3.13 synthesize.py path/to/text.txt \
    --voice zh-CN-YunxiNeural \
    --rate "+0%"

# 自定义输出名
python3.13 synthesize.py path/to/text.txt --name my_clip
# → output/my_clip.mp3
```

## 与 asr/transcribe.py 配对使用

`transcribe.py` 输出 `.txt` 时每行带时间戳前缀（`695.73s → 697.03s | But I'm sorry.`）。

`synthesize.py` 会自动用正则把这层前缀剥掉，只把纯文本送去 TTS。

```bash
# 完整链路：音频 → 文本 → 翻译 → 音频
python3.13 ../asr/transcribe.py some.mp3 --json
# → some.txt（带时间戳）

# ... 自己翻译成中文得到 some.zh.txt ...

python3.13 synthesize.py some.zh.txt
# → output/some.zh.mp3
```

## 环境变量

- `TTS_VOICE`：默认音色（默认 `zh-CN-XiaoxiaoNeural`，跟 `diaryofawimpykit/make_video.py` 一致）
- `TTS_RATE`：默认语速（默认 `+10%`）

## 常用音色

| 语言 | 音色 |
|---|---|
| 中文（女，温柔）| `zh-CN-XiaoxiaoNeural` |
| 中文（男，活泼）| `zh-CN-YunxiNeural` |
| 中文（男，新闻）| `zh-CN-YunjianNeural` |
| 英文（女）| `en-US-AriaNeural` |
| 英文（男）| `en-US-GuyNeural` |

更多音色：https://speech.microsoft.com/portal/voicegallery

## 关于离线

`edge-tts` 走微软云端，**需要联网**。如果要离线 TTS，把脚本里的 `import edge_tts` 换成 `piper-tts` 即可（piper 模型几十 MB，下载一次后本地推理）。