#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文本 → mp3（基于 Coze TTS，单次最多 1024 字符）。"""

import os
import re
import requests
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python < 3.11

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = PROJECT_ROOT / "script" / "config.toml"

API_URL = "https://api.coze.cn/v1/audio/speech"
DEFAULT_VOICE_ID = 7468512265134899251
MAX_INPUT_LEN = 1000  # Coze 限制 1024，留余量


def _load_api_key() -> str:
    key = os.environ.get("COZE_API_KEY")
    if key:
        return key
    if DEFAULT_CONFIG.exists():
        cfg = tomllib.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        return cfg.get("coze", {}).get("api_key", "")
    return ""


def _synthesize_single(text: str, voice_id: int, speed: float, pitch: float,
                       api_key: str) -> bytes:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "tts-1", "input": text, "voice_id": voice_id,
               "language": "zh-CN", "speed": speed, "pitch": pitch,
               "format": "mp3", "sample_rate": 24000}
    resp = requests.post(API_URL, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.content


def _split_text(text: str, limit: int = MAX_INPUT_LEN) -> list[str]:
    """把文本切成每段 <= limit 的若干段。

    先按句末标点切句，再贪心合并到 limit 内；单句仍超长则按 limit 硬切。
    """
    # 按句末标点切句，保留标点
    sentences = re.split(r"(?<=[。！？；\n])", text)
    sentences = [s for s in sentences if s]

    # 拆开仍然超长的单句
    pieces = []
    for s in sentences:
        if len(s) <= limit:
            pieces.append(s)
        else:
            for i in range(0, len(s), limit):
                pieces.append(s[i:i + limit])

    # 贪心合并相邻片段到 limit 内
    chunks = []
    buf = ""
    for p in pieces:
        if buf and len(buf) + len(p) > limit:
            chunks.append(buf)
            buf = ""
        buf += p
    if buf:
        chunks.append(buf)
    return chunks


def synthesize(text: str, out_path: Path, voice_id: int = None,
               speed: float = 1.0, pitch: float = 1.0,
               api_key: str = None) -> None:
    voice_id = voice_id or DEFAULT_VOICE_ID
    api_key = api_key or _load_api_key()

    chunks = _split_text(text) if len(text) > MAX_INPUT_LEN else [text]

    # 分段合成后直接拼接字节（同编码 mp3 可直接 concat）
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as out_f:
        for chunk in chunks:
            data = _synthesize_single(chunk, voice_id, speed, pitch, api_key)
            out_f.write(data)

    print(f"✓ {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    import sys

    text = sys.argv[1] if len(sys.argv) > 1 else "你好，欢迎使用 Coze TTS！"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output.mp3")
    synthesize(text, out)
