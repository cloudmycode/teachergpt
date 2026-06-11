#!/usr/local/bin/python3.13
# -*- coding: utf-8 -*-
"""
遍历 chinese_mp3 下所有 mp3，调用 transcribe.py 转写，
结果按同样子目录结构保存到 chinese_text。

用法:
  python3.13 tts/script/transcribe_dir.py
  python3.13 tts/script/transcribe_dir.py --src /path/to/mp3 --dst /path/to/text
"""

import argparse
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SRC = PROJECT_ROOT / "data" / "chinese_mp3"
DEFAULT_DST = PROJECT_ROOT / "data" / "chinese_text"
TRANSCRIBER = PROJECT_ROOT / "asr" / "transcribe.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批量转写 mp3 目录")
    p.add_argument("--src", type=Path, default=DEFAULT_SRC, help="mp3 根目录")
    p.add_argument("--dst", type=Path, default=DEFAULT_DST, help="文本输出根目录")
    p.add_argument("--lang", default="zh", help="语言（默认 zh）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()

    if not src.exists():
        print(f"✗ 源目录不存在: {src}")
        sys.exit(1)

    if not TRANSCRIBER.exists():
        print(f"✗ 转写脚本不存在: {TRANSCRIBER}")
        sys.exit(1)

    # 收集所有 mp3
    mp3_files = sorted(src.rglob("*.mp3"))
    if not mp3_files:
        print(f"未找到 mp3 文件: {src}")
        sys.exit(0)

    print(f"源目录  : {src}")
    print(f"目标目录: {dst}")
    print(f"共找到  : {len(mp3_files)} 个 mp3 文件\n")

    # 计算已完成的（目标目录下已存在同名 txt）
    done: set[str] = set()
    for txt in dst.rglob("*.txt"):
        rel = txt.relative_to(dst)
        # 用 stem 匹配，允许源文件扩展名不同
        done.add(rel.with_suffix("").name)

    todo = [f for f in mp3_files if f.stem not in done]
    skip = len(mp3_files) - len(todo)
    print(f"已完成: {skip}, 待处理: {len(todo)}\n")

    if not todo:
        print("全部已完成，无需处理。")
        return

    start_all = time.time()
    ok = 0
    fail = 0
    for i, mp3 in enumerate(todo, 1):
        rel = mp3.relative_to(src)
        out_dir = dst / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        print(f"[{i}/{len(todo)}] {rel}  开始 {time.strftime('%H:%M:%S')}")
        ret = subprocess.run(
            [sys.executable, str(TRANSCRIBER), str(mp3),
             "--out", str(out_dir), "--lang", args.lang],
            capture_output=True,
            text=True,
        )
        dt = time.time() - t0
        print(f"  耗时 {timedelta(seconds=int(dt))}")
        if ret.returncode == 0:
            ok += 1
        else:
            fail += 1
            print(f"  ✗ 失败: {ret.stderr.strip() or ret.stdout.strip()}")

    total = time.time() - start_all
    print(f"\n完成: 成功 {ok}, 失败 {fail}, 总耗时 {timedelta(seconds=int(total))}")


if __name__ == "__main__":
    main()
