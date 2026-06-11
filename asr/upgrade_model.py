#!/usr/local/bin/python3.13
# -*- coding: utf-8 -*-
"""
管理 faster-whisper 模型：检查更新、下载、覆盖。

默认目标：../models/faster-whisper-medium/（相对脚本位置）。
支持任意 Systran/faster-whisper-* 模型。

用法:
  # 默认：只查远端 commit hash 跟本地对比，告诉你有没有更新及更新命令
  python3.13 upgrade_model.py

  # 执行下载/更新（不清空目录，snapshot_download 会拉远端最新）
  python3.13 upgrade_model.py --download

  # 从 ~/.cache/huggingface 复制（需本机已下载过，最快）
  python3.13 upgrade_model.py --download --from-cache

  # 清空目录后重新下载（覆盖式更新）
  python3.13 upgrade_model.py --download --force

  # 切其他模型
  python3.13 upgrade_model.py --repo Systran/faster-whisper-small

环境变量:
  HF_ENDPOINT  HuggingFace 镜像，国内推荐 https://hf-mirror.com
  HF_TOKEN     私有模型所需 token
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_REPO = "Systran/faster-whisper-medium"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET = SCRIPT_DIR / "models" / "faster-whisper-medium"
REVISION_FILE = ".revision"  # 记录本地上次拉取的远端 commit sha


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="管理 faster-whisper 模型（默认检查更新，加 --download 才下载）"
    )
    p.add_argument("--repo", default=DEFAULT_REPO, help=f"HF 仓库 id（默认 {DEFAULT_REPO}）")
    p.add_argument("--to", default=str(DEFAULT_TARGET), help=f"目标目录（默认 {DEFAULT_TARGET}）")
    p.add_argument(
        "--download",
        action="store_true",
        help="实际下载/更新到目标目录（默认不下载，只查更新）",
    )
    p.add_argument(
        "--from-cache",
        action="store_true",
        help="从 ~/.cache/huggingface 复制（与 --download 配合）",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="清空目录后重下（与 --download 配合；单独使用无意义）",
    )
    return p.parse_args()


def is_complete(model_dir: Path) -> bool:
    """faster-whisper 模型必需的权重文件"""
    return (model_dir / "model.bin").exists()


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


def get_local_sha(target: Path) -> str | None:
    rev_file = target / REVISION_FILE
    return rev_file.read_text().strip() if rev_file.exists() else None


def write_local_sha(target: Path, sha: str) -> None:
    (target / REVISION_FILE).write_text(sha)


def copy_from_cache(repo_id: str, target: Path) -> None:
    from huggingface_hub import scan_cache_dir
    cache_info = scan_cache_dir()
    snapshot_path = None
    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            rev = max(repo.revisions, key=lambda r: r.size)
            snapshot_path = Path(rev.snapshot_path)
            break
    if snapshot_path is None or not snapshot_path.exists():
        print(f"✗ 本地 HF 缓存未找到 {repo_id}，去掉 --from-cache 走下载流程")
        sys.exit(1)
    print(f"从本地缓存复制: {snapshot_path}")
    shutil.copytree(snapshot_path, target, dirs_exist_ok=True)
    print(f"✓ 已复制到: {target}")


def download_from_hub(repo_id: str, target: Path) -> None:
    from huggingface_hub import snapshot_download
    print(f"从 HuggingFace 下载: {repo_id}")
    print(f"目标目录: {target}")
    print("(网络不通可设 HF_ENDPOINT=https://hf-mirror.com 走镜像)\n")
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )


def cmd_check(args: argparse.Namespace, target: Path) -> None:
    """默认：对比远端 vs 本地 commit，告知更新状态。"""
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
    update_cmd = f"python3.13 {script_name} --download"

    if local_sha is None:
        if not target.exists():
            print(f"本地: 未下载\n")
            print(f"✗ 本地无目录")
            print(f"\n执行以下命令下载:")
            print(f"  {update_cmd}")
        elif is_complete(target):
            print(f"本地: 无 .revision 记录（目录完整，可能手动拷贝）\n")
            print(f"⚠ 状态未知")
            print(f"\n执行以下命令对齐 commit 记录并按需更新:")
            print(f"  {update_cmd}")
        else:
            print(f"本地: 目录不完整（缺 model.bin 等权重）\n")
            print(f"✗ 需要补齐")
            print(f"\n执行以下命令补齐:")
            print(f"  {update_cmd}")
        return

    print(f"本地 commit: {local_sha[:12]}\n")
    if local_sha == remote_sha:
        print(f"✓ 已是最新")
    else:
        print(f"⬆ 有更新: {local_sha[:12]} → {remote_short}")
        print(f"\n执行以下命令更新:")
        print(f"  {update_cmd}")


def cmd_download(args: argparse.Namespace, target: Path) -> None:
    """实际下载/更新到目标目录。"""
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
            print(f"目录不完整，补齐缺失权重文件\n")
    else:
        print(f"目标不存在，全量下载\n")

    if args.from_cache:
        copy_from_cache(args.repo, target)
    else:
        download_from_hub(args.repo, target)

    # 写入本次 commit 记录
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
    print(f"\n✓ 完成。模型路径: {target}")


def main() -> None:
    args = parse_args()
    target = Path(args.to).resolve()

    if args.force and not args.download:
        print("✗ --force 必须与 --download 一起使用（强制覆盖重下）")
        sys.exit(1)

    if args.download:
        cmd_download(args, target)
    else:
        cmd_check(args, target)


if __name__ == "__main__":
    main()