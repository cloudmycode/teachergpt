#!/usr/local/bin/python3.13
# -*- coding: utf-8 -*-
"""
遍历 chinese_mp3 下所有 mp3，调用 transcribe.py 转写，
结果按同样子目录结构保存到 chinese_text。

支持多进程并行：
  python3.13 script/transcribe_dir.py --workers 4

原理：启动时扫描所有 mp3，过滤掉已有 txt 的；
对剩余的用 multiprocessing.Pool 并行调用 transcribe.py，
每个子进程开始处理时先创建空目标文件占位，避免重复执行。
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import timedelta
from multiprocessing import Pool, cpu_count
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SRC = PROJECT_ROOT / "data" / "chinese_mp3"
DEFAULT_DST = PROJECT_ROOT / "data" / "chinese_text"
TRANSCRIBER = PROJECT_ROOT / "asr" / "transcribe.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批量转写 mp3 目录（支持多进程）")
    p.add_argument("--src", type=Path, default=DEFAULT_SRC, help="mp3 根目录")
    p.add_argument("--dst", type=Path, default=DEFAULT_DST, help="文本输出根目录")
    p.add_argument("--lang", default="zh", help="语言（默认 zh）")
    p.add_argument("--workers", type=int, default=1, help=f"并行进程数（默认 1，最大 {cpu_count()}）")
    return p.parse_args()


def process_one(args_tuple: tuple) -> tuple[str, str, str]:
    """单个 mp3 的处理函数（在子进程中执行）。

    参数用 tuple 打包是因为 Pool.map 只接受单参数函数。
    返回: (mp3 相对路径, 状态(ok/fail/skip), 消息)
    """
    mp3, src, dst, out_dir, out_txt, lang, python_exe = args_tuple
    rel = mp3.relative_to(src)

    # 确保输出子目录存在（否则 touch 会抛 FileNotFoundError）
    out_dir.mkdir(parents=True, exist_ok=True)

    # 严格占位：目标已存在（其他进程已占位/已完成）则跳过
    if out_txt.exists():
        return (str(rel), "skip", "")
    try:
        # 原子创建空占位（底层 O_EXCL），抢到才处理
        out_txt.touch(exist_ok=False)
    except FileExistsError:
        # 另一个进程刚抢先创建
        return (str(rel), "skip", "")

    t0 = time.time()
    # 开始日志：在子进程内直接打印，flush=True 保证实时显示（主进程拿不到“开始”事件）
    print(f"  ▶ [{datetime.now():%H:%M:%S}] [pid {os.getpid()}] {rel}  开始", flush=True)
    ret = subprocess.run(
        [python_exe, str(TRANSCRIBER), str(mp3),
         "--out", str(out_dir), "--lang", lang],
        capture_output=True,
        text=True,
    )
    dt = time.time() - t0
    if ret.returncode == 0:
        return (str(rel), "ok", f"耗时 {timedelta(seconds=int(dt))}")
    else:
        err = ret.stderr.strip() or ret.stdout.strip() or "未知错误"
        # 失败时删除占位文件，允许重试
        try:
            out_txt.unlink()
        except FileNotFoundError:
            pass
        return (str(rel), "fail", f"耗时 {timedelta(seconds=int(dt))} | {err}")





def build_tasks(src: Path, dst: Path, lang: str, python_exe: str) -> list[tuple]:
    """收集待处理任务，过滤已完成的。"""
    mp3_files = sorted(src.rglob("*.mp3"))
    if not mp3_files:
        return []

    tasks = []
    for mp3 in mp3_files:
        rel = mp3.relative_to(src)
        out_dir = dst / rel.parent
        out_txt = out_dir / f"{mp3.stem}.txt"
        # 按实际目标路径判断（避免不同子目录同名文件互相误判）
        if out_txt.exists():
            if out_txt.stat().st_size > 0:
                continue  # 已完成
            # 0 字节占位：上次中断遗留（此刻是单进程启动阶段，无 worker 在跑，安全删除重处理）
            out_txt.unlink()
        tasks.append((mp3, src, dst, out_dir, out_txt, lang, python_exe))
    return tasks


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    python_exe = sys.executable

    if not src.exists():
        print(f"✗ 源目录不存在: {src}")
        sys.exit(1)

    if not TRANSCRIBER.exists():
        print(f"✗ 转写脚本不存在: {TRANSCRIBER}")
        sys.exit(1)

    tasks = build_tasks(src, dst, args.lang, python_exe)
    skip_init = len([f for f in src.rglob("*.mp3")]) - len(tasks)
    workers = min(args.workers, cpu_count(), len(tasks) if tasks else 1)

    print(f"源目录  : {src}")
    print(f"目标目录: {dst}")
    print(f"已完成: {skip_init}, 待处理: {len(tasks)}, 并行: {workers} 进程\n")

    if not tasks:
        print("全部已完成，无需处理。")
        return

    start_all = time.time()
    ok = 0
    fail = 0
    skip = 0
    done_count = 0

    def on_done(result: tuple):
        """imap_unordered 结果回调风格的包装。"""
        nonlocal ok, fail, skip, done_count
        rel_str, status, msg = result
        done_count += 1
        if status == "ok":
            ok += 1
            print(f"  ✓ [{done_count}/{len(tasks)}] {rel_str}  {msg}")
        elif status == "fail":
            fail += 1
            print(f"  ✗ [{done_count}/{len(tasks)}] {rel_str}  {msg}")
        else:
            skip += 1
            print(f"  ⊘ [{done_count}/{len(tasks)}] {rel_str}  已被其他进程处理")

    with Pool(processes=workers) as pool:
        results = pool.imap_unordered(process_one, tasks)
        for result in results:
            on_done(result)

    total = time.time() - start_all
    print(f"\n完成: 成功 {ok}, 失败 {fail}, 跳过 {skip}, "
          f"总耗时 {timedelta(seconds=int(total))}")


if __name__ == "__main__":
    main()
