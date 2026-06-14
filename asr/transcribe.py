#!/usr/local/bin/python3.13
# -*- coding: utf-8 -*-
"""
用本地 faster-whisper 模型把音频转成文本。

用法:
  python3.13 transcribe.py path/to/audio.mp3
  python3.13 transcribe.py path/to/audio.mp3 --out ./output
  python3.13 transcribe.py path/to/audio.mp3 --json          # 同时输出段级+词级时间戳
  python3.13 transcribe.py path/to/audio.mp3 --lang zh

输出（写到 --out 目录，默认 ./output）:
  <stem>.txt     每行一段识别文本
  <stem>.json    段级 + 词级时间戳（仅 --json 时生成）

环境变量:
  WHISPER_MODEL  模型路径或 HF 模型名，默认 ./models/faster-whisper-small
"""

import argparse
import json
import os
import sys
import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = os.environ.get(
    "WHISPER_MODEL",
    str(SCRIPT_DIR / "models" / "faster-whisper-small"),
)
DEFAULT_OUT = SCRIPT_DIR / "output"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="音频 → 文本（faster-whisper 本地推理）")
    p.add_argument("audio", help="mp3 / wav 路径")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"模型路径或 HF 名（默认 {DEFAULT_MODEL}）")
    p.add_argument("--out", default=str(DEFAULT_OUT), help=f"输出目录（默认 {DEFAULT_OUT}）")
    p.add_argument("--lang", default=None, help="语言代码（默认自动检测；可指定 zh / en 等）")
    p.add_argument("--json", action="store_true", help="同时输出 json 时间戳")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="推理设备（默认 cpu）")
    p.add_argument("--compute", default="int8", help="compute_type（默认 int8；GPU 可改 float16）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    audio = Path(args.audio).resolve()
    if not audio.exists():
        print(f"✗ 音频不存在: {audio}")
        sys.exit(1)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio.stem
    out_txt = out_dir / f"{stem}.txt"
    out_json = out_dir / f"{stem}.json"

    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] transcribe 开始")
    print(f"音频文件: {audio}")
    print(f"模型    : {args.model}")
    print(f"输出    : {out_txt}\n")

    from faster_whisper import WhisperModel
    print("加载模型...")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute)
    print("开始识别...\n")

    segments_iter, info = model.transcribe(
        str(audio),
        language=args.lang,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )

    print(f"检测语言: {info.language}  时长: {info.duration:.1f}s\n")

    lines: list[str] = []
    json_segs: list[dict] = []
    for seg in segments_iter:
        text = seg.text.strip()
        lines.append(f"{seg.start:.2f}s → {seg.end:.2f}s | {text}")
        print(f"  {seg.start:7.2f}s → {seg.end:7.2f}s | {text}")
        if args.json:
            json_segs.append({
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": text,
                "words": [
                    {
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                    }
                    for w in seg.words
                ],
            })

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✓ 文本已写入: {out_txt}  ({len(lines)} 段)")

    if args.json:
        out_json.write_text(
            json.dumps(json_segs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"✓ 时间戳已写入: {out_json}")

    print(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] transcribe 完成")


if __name__ == "__main__":
    main()