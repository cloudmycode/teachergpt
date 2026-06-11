#!/usr/local/bin/python3.13
# -*- coding: utf-8 -*-
"""
文本 → mp3（基于 edge-tts，调用微软 Edge 免费 TTS，云端）。

自动识别 txt 是否带 transcribe.py 的时间戳前缀（形如 `12.34s → 56.78s | text`），有则剥离。

用法:
  python3.13 synthesize.py path/to/text.txt
  python3.13 synthesize.py path/to/text.txt --out ./output
  python3.13 synthesize.py path/to/text.txt --voice zh-CN-YunxiNeural --rate "+0%"
  python3.13 synthesize.py path/to/text.txt --name my_clip

环境变量:
  TTS_VOICE  默认音色（默认 zh-CN-XiaoxiaoNeural，跟 diaryofawimpykit/make_video.py 一致）
  TTS_RATE   语速（默认 +10%）
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / "output"

# 默认参数：跟 diaryofawimpykit/make_video.py 保持一致
DEFAULT_VOICE = os.environ.get("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
DEFAULT_RATE = os.environ.get("TTS_RATE", "+10%")

# 匹配 transcribe.py 输出格式: "12.34s → 56.78s | text"
TS_PREFIX_RE = re.compile(r"^\s*\d+(?:\.\d+)?s\s*[→\-]\s*\d+(?:\.\d+)?s\s*\|\s*")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="文本 → mp3（edge-tts）")
    p.add_argument("txt", help="文本文件路径（自动剥离时间戳前缀）")
    p.add_argument("--out", default=str(DEFAULT_OUT), help=f"输出目录（默认 {DEFAULT_OUT}）")
    p.add_argument("--voice", default=DEFAULT_VOICE, help=f"edge-tts 音色（默认 {DEFAULT_VOICE}）")
    p.add_argument("--rate", default=DEFAULT_RATE, help=f"语速（默认 {DEFAULT_RATE}，如 -10% / +0% / +20%）")
    p.add_argument("--name", default=None, help="输出文件名（不含后缀），默认用输入 stem")
    return p.parse_args()


def strip_timestamp(line: str) -> str:
    return TS_PREFIX_RE.sub("", line)


async def synthesize_async(text: str, voice: str, rate: str, out_path: Path) -> None:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(str(out_path))


def main() -> None:
    args = parse_args()

    txt_path = Path(args.txt).resolve()
    if not txt_path.exists():
        print(f"✗ 文本不存在: {txt_path}")
        sys.exit(1)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or txt_path.stem
    out_mp3 = out_dir / f"{stem}.mp3"

    print(f"文本文件: {txt_path}")
    print(f"输出    : {out_mp3}")
    print(f"音色    : {args.voice}  语速: {args.rate}\n")

    raw_lines = txt_path.read_text(encoding="utf-8").splitlines()
    cleaned = [strip_timestamp(ln).rstrip() for ln in raw_lines if ln.strip()]
    text = "\n".join(cleaned)

    if not text.strip():
        print("✗ 文本为空")
        sys.exit(1)

    print(f"剥离时间戳后 {len(cleaned)} 段，长度 {len(text)} 字符")
    print("开始合成...\n")

    asyncio.run(synthesize_async(text, args.voice, args.rate, out_mp3))

    size_kb = out_mp3.stat().st_size / 1024
    print(f"\n✓ 已写入: {out_mp3}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()