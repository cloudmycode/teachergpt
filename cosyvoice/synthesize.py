#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本 → mp3（基于 CosyVoice 2.0，本地离线推理）。

支持情感标签：<happy> <sad> <calm> <angry> <fearful> <disgusted> <surprised> <neutral>

用法:
  python3 synthesize.py path/to/text.txt
  python3 synthesize.py path/to/text.txt --name my_clip
  python3 synthesize.py path/to/text.txt --device cpu

环境变量:
  COSYVOICE_DEVICE  推理设备（默认 cpu）
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / "output"
DEFAULT_DEVICE = os.environ.get("COSYVOICE_DEVICE", "cpu")

# CosyVoice 源码路径（需先 git clone --recursive）
_COSYVOICE_HOME = Path(
    os.environ.get("COSYVOICE_HOME", Path.home() / "CosyVoice")
)
if _COSYVOICE_HOME.is_dir():
    sys.path.insert(0, str(_COSYVOICE_HOME))

# 本地模型目录（与 download_model.py 的 --to 默认值一致）
MODEL_DIR = SCRIPT_DIR / "models" / "CosyVoice2-0.5B"
SAMPLE_RATE = 24000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="文本 → mp3（CosyVoice 2.0）")
    p.add_argument("txt", help="文本文件路径")
    p.add_argument("--out", default=str(DEFAULT_OUT), help=f"输出目录（默认 {DEFAULT_OUT}）")
    p.add_argument("--name", default=None, help="输出文件名（不含后缀），默认用输入 stem")
    p.add_argument("--device", default=DEFAULT_DEVICE, help=f"推理设备（默认 {DEFAULT_DEVICE}）")
    p.add_argument("--model", default=str(MODEL_DIR), help=f"模型目录（默认 {MODEL_DIR}）")
    return p.parse_args()


def synthesize_text(text: str, out_path: str, device: str = "cpu",
                    model_dir: str | None = None) -> None:
    """合成单段文本到音频文件（wav/mp3）。"""
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2
    except ModuleNotFoundError:
        print(
            "\n✗ CosyVoice 未正确安装。请按以下步骤操作：\n"
            "  cd ~\n"
            "  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git\n"
            "  cd CosyVoice\n"
            "  pip install -r requirements.txt\n"
            "\n然后设置环境变量:\n"
            "  export COSYVOICE_HOME=~/CosyVoice\n"
        )
        sys.exit(1)

    import soundfile as sf

    mdir = model_dir or str(MODEL_DIR)
    if not Path(mdir).exists():
        print(f"✗ 模型目录不存在: {mdir}")
        print(f"  请先运行: python3 cosyvoice/download_model.py --download")
        sys.exit(1)

    model = CosyVoice2(mdir)
    output = model.inference(text, stream=False)

    # 拼接所有段
    import numpy as np
    chunks = []
    for seg in output:
        chunks.append(seg["tts_speech"])
    audio = np.concatenate(chunks) if chunks else np.array([])

    if len(audio) == 0:
        print("✗ 合成失败：无音频输出")
        sys.exit(1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix == ".mp3":
        # soundfile 不直接写 mp3，先写 wav 再 ffmpeg 转
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        sf.write(tmp_wav, audio, SAMPLE_RATE)
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_wav, "-acodec", "libmp3lame", "-q:a", "2", str(out_path)],
            capture_output=True, check=True,
        )
        os.unlink(tmp_wav)
    else:
        sf.write(str(out_path), audio, SAMPLE_RATE)

    size_kb = out_path.stat().st_size / 1024
    print(f"  ✓ {out_path}  ({size_kb:.1f} KB)")


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
    print(f"设备    : {args.device}")
    print()

    text = txt_path.read_text(encoding="utf-8").strip()
    if not text:
        print("✗ 文本为空")
        sys.exit(1)

    print(f"文本长度 {len(text)} 字符，开始合成...")
    synthesize_text(text, str(out_mp3), device=args.device, model_dir=args.model)


if __name__ == "__main__":
    main()
