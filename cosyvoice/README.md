# cosyvoice/ — CosyVoice 2.0 TTS

本地文本转语音，基于阿里 CosyVoice 2.0（开源，离线，支持情感控制）。

## 文件

- `download_model.py` — 下载/检查模型
- `synthesize.py` — 文本 → mp3，支持情感标签

## 快速开始

```bash
# 1. 克隆 CosyVoice 源码 + 安装依赖
cd ~
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
pip3 install -r requirements.txt

# 2. 下载模型
cd /path/to/teachergpt
python3 cosyvoice/download_model.py --download

# 3. 测试合成
export COSYVOICE_HOME=~/CosyVoice
python3 cosyvoice/synthesize.py <(echo '<calm>静以修身。</calm>')
```

## 用法

```bash
# 模型管理
python3 cosyvoice/download_model.py                       # 检查状态
python3 cosyvoice/download_model.py --download            # 下载/更新
python3 cosyvoice/download_model.py --download --test     # 下载后跑测试合成

# TTS 合成
python3 cosyvoice/synthesize.py path/to/text.txt
python3 cosyvoice/synthesize.py path/to/text.txt --name my_clip
# → cosyvoice/output/my_clip.mp3
```

## 情感标签

在文本中嵌入 XML 标签即可控制语气：

```
<happy>这句话讲得真开心！</happy>
<sad>但有时候也会难过。</sad>
<calm>静以修身，俭以养德。</calm>
<angry>这是不对的！</angry>
```

可用标签：`<happy>` `<sad>` `<calm>` `<angry>` `<fearful>` `<disgusted>` `<surprised>` `<neutral>`

## 环境变量

- `COSYVOICE_HOME`：CosyVoice 源码目录（默认 `~/CosyVoice`）
- `COSYVOICE_DEVICE`：推理设备（默认 `cpu`）

## 依赖

CosyVoice 非 PyPI 包，需从 GitHub 克隆安装：

```bash
cd ~
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
pip3 install -r requirements.txt
```
