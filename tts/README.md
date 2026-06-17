# tts/

文本转语音工具集。

## 目录

```
tts/
  edge_tts.py   微软 Edge TTS（免费云端）
  coze_tts.py   Coze TTS（商业 API，中文效果好）
  output/       默认输出目录（自动创建）
```

## 依赖

```bash
pip3 install edge-tts      # edge_tts.py
pip3 install requests      # coze_tts.py
```

## 用法

### Edge TTS（微软免费）

```bash
# 基础用法
python3 edge_tts.py path/to/text.txt

# 切音色 / 语速
python3 edge_tts.py path/to/text.txt --voice zh-CN-YunxiNeural --rate "+0%"

# 自定义输出名
python3 edge_tts.py path/to/text.txt --name my_clip
# → output/my_clip.mp3
```

### Coze TTS

```bash
# 基础用法
python3 coze_tts.py "你好"
# → output.mp3

# 指定输出文件
python3 coze_tts.py "你好" out.mp3
```

或在代码中调用：

```python
from tts.coze_tts import synthesize
synthesize("床前明月光", Path("out.mp3"), speed=0.9)
```

## 配置

Coze TTS 的 API Key 配置在 `script/config.toml`：

```toml
[coze]
api_key = "pat_填入你的Coze密钥"
```

也可通过环境变量 `COZE_API_KEY` 覆盖。

## 常用 Edge 音色

| 语言 | 音色 |
|---|---|
| 中文（女，温柔）| `zh-CN-XiaoxiaoNeural` |
| 中文（男，活泼）| `zh-CN-YunxiNeural` |
| 中文（男，新闻）| `zh-CN-YunjianNeural` |
| 英文（女）| `en-US-AriaNeural` |
| 英文（男）| `en-US-GuyNeural` |

更多：https://speech.microsoft.com/portal/voicegallery
