#!/usr/local/bin/python3.13
# -*- coding: utf-8 -*-
"""
管理 BGE embedding 模型：检查、下载 BAAI/bge-large-zh-v1.5。

默认目标：bge/models/bge-large-zh-v1.5/（相对本脚本）。
index_units.py 用 --model-dir ./bge/models 即可指向该目录。

用法:
  python3 bge/download_model.py                       # 检查是否有更新
  python3 bge/download_model.py --download            # 增量下载/更新
  python3 bge/download_model.py --download --force    # 清空重下

环境变量:
  HF_ENDPOINT  HuggingFace 镜像，国内推荐 https://hf-mirror.com
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

DEFAULT_REPO = "BAAI/bge-large-zh-v1.5"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET = SCRIPT_DIR / "models" / "bge-large-zh-v1.5"
REVISION_FILE = ".revision"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="管理 BGE 模型（默认查更新，加 --download 才下载）"
    )
    p.add_argument("--repo", default=DEFAULT_REPO, help=f"HF 仓库（默认 {DEFAULT_REPO}）")
    p.add_argument("--to", default=str(DEFAULT_TARGET), help=f"目标目录（默认 {DEFAULT_TARGET}）")
    p.add_argument("--download", action="store_true", help="实际下载/更新")
    p.add_argument("--force", action="store_true", help="清空后重下（与 --download 配合）")
    return p.parse_args()


# ------------------------------------------------------------------- helpers

def is_complete(model_dir: Path) -> bool:
    """BGE 必需的模型文件"""
    return (
        (model_dir / "config.json").exists()
        and (model_dir / "model.safetensors").exists()
    )


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def fmt_size(n: int) -> str:
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def get_remote_sha(repo_id: str) -> str:
    from huggingface_hub import HfApi
    return HfApi().repo_info(repo_id).sha


def get_local_sha(target: Path) -> Optional[str]:
    rev_file = target / REVISION_FILE
    return rev_file.read_text().strip() if rev_file.exists() else None


def write_local_sha(target: Path, sha: str) -> None:
    (target / REVISION_FILE).write_text(sha)


def download_from_hub(repo_id: str, target: Path) -> None:
    from huggingface_hub import snapshot_download
    print(f"从 HuggingFace 下载: {repo_id}")
    print(f"目标目录: {target}")
    print("(网络不通可设 HF_ENDPOINT=https://hf-mirror.com)\n")
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )


# ------------------------------------------------------------------- commands

def cmd_check(args: argparse.Namespace, target: Path) -> None:
    print(f"源仓库: {args.repo}")
    print(f"目标目录: {target}\n")

    try:
        remote_sha = get_remote_sha(args.repo)
    except Exception as e:
        print(f"✗ 获取远端 commit 失败: {e}")
        print(f"  检查 HF_ENDPOINT / 网络是否可用")
        sys.exit(1)
    remote_short = remote_sha[:12]
    local_sha = get_local_sha(target)
    print(f"远端 commit: {remote_short}")

    script_name = Path(sys.argv[0]).name
    download_cmd = f"python3 {script_name} --download"

    if local_sha is None:
        if not target.exists():
            print(f"本地: 未下载\n")
            print(f"✗ 模型未下载")
        elif is_complete(target):
            print(f"本地: 无 .revision 记录（目录完整）\n")
            print(f"⚠ 状态未知")
        else:
            print(f"本地: 目录不完整\n")
            print(f"✗ 需要补齐")
        print(f"\n执行以下命令下载:")
        print(f"  {download_cmd}")
        return

    print(f"本地 commit: {local_sha[:12]}\n")
    if local_sha == remote_sha:
        print(f"✓ 已是最新")
    else:
        print(f"⬆ 有更新: {local_sha[:12]} → {remote_short}")
        print(f"\n执行以下命令更新:")
        print(f"  {download_cmd}")


def cmd_download(args: argparse.Namespace, target: Path) -> None:
    print(f"源仓库: {args.repo}")
    print(f"目标目录: {target}\n")

    if target.exists():
        if is_complete(target):
            if args.force:
                print(f"--force: 清空 {target} 后重新下载\n")
                shutil.rmtree(target)
            else:
                print(f"目录已完整，保留现有文件做增量更新\n")
        else:
            print(f"目录不完整，补齐缺失文件\n")
    else:
        print(f"目标不存在，全量下载\n")

    download_from_hub(args.repo, target)

    try:
        sha = get_remote_sha(args.repo)
        write_local_sha(target, sha)
        print(f"已记录远端 commit: {sha[:12]} -> {target / REVISION_FILE}")
    except Exception as e:
        print(f"⚠ 记录 .revision 失败: {e}")

    if target.exists():
        print(f"\n目录大小: {fmt_size(dir_size(target))}")
        files = sorted(p.name for p in target.iterdir() if p.is_file())
        print(f"文件清单 ({len(files)} 个):")
        for f in files:
            print(f"  - {f}")
    print(f"\n✓ 完成。使用: ")
    print(f"  python3 bge/index_units.py --model-dir {target.parent}")


# ------------------------------------------------------------------- main

def main() -> None:
    args = parse_args()
    target = Path(args.to).resolve()

    if args.force and not args.download:
        print("✗ --force 必须与 --download 一起使用")
        sys.exit(1)

    if args.download:
        cmd_download(args, target)
    else:
        cmd_check(args, target)


if __name__ == "__main__":
    main()
