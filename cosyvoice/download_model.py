#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
管理 CosyVoice 2.0 模型：检查、下载 iic/CosyVoice2-0.5B。

默认目标：cosyvoice/models/CosyVoice2-0.5B/（相对本脚本）。
synthesize.py 默认读同一目录。

用法:
  python3 cosyvoice/download_model.py                       # 检查状态
  python3 cosyvoice/download_model.py --download            # 下载/更新
  python3 cosyvoice/download_model.py --download --force    # 清空重下
  python3 cosyvoice/download_model.py --download --test     # 下载后跑测试合成

环境变量:
  MODELSCOPE_CACHE  ModelScope 下载缓存（下载完会拷到 --to 目录）
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

MODEL_REPO = "iic/CosyVoice2-0.5B"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET = SCRIPT_DIR / "models" / "CosyVoice2-0.5B"
REVISION_FILE = ".ms_revision"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="管理 CosyVoice 2.0 模型（默认只检查，加 --download 才下载）"
    )
    p.add_argument("--repo", default=MODEL_REPO, help=f"ModelScope 仓库（默认 {MODEL_REPO}）")
    p.add_argument("--to", default=str(DEFAULT_TARGET), help=f"目标目录（默认 {DEFAULT_TARGET}）")
    p.add_argument("--download", action="store_true", help="实际下载/更新模型")
    p.add_argument("--force", action="store_true", help="清空后重下（与 --download 配合）")
    p.add_argument("--test", action="store_true", help="下载后跑测试合成（需 --download）")
    return p.parse_args()


# ------------------------------------------------------------------- helpers

def _ensure_modelscope():
    try:
        import modelscope  # noqa: F401
    except ImportError:
        print("✗ 未安装 modelscope")
        print("  执行: pip3 install modelscope")
        sys.exit(1)


def is_complete(model_dir: Path) -> bool:
    required = ["cosyvoice.yaml", "flow.pt", "hift.pt"]
    return all((model_dir / f).exists() for f in required)


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def fmt_size(n: int) -> str:
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def get_remote_sha(repo: str) -> Optional[str]:
    _ensure_modelscope()
    try:
        from modelscope.hub.api import HubApi
        commits = HubApi().get_model_commits(model_id=repo, limit=1)
        return commits[0]["Revision"] if commits else None
    except Exception:
        return None


def get_local_sha(target: Path) -> Optional[str]:
    rev_file = target / REVISION_FILE
    return rev_file.read_text().strip() if rev_file.exists() else None


def write_local_sha(target: Path, sha: str) -> None:
    (target / REVISION_FILE).write_text(sha)


# ------------------------------------------------------------------- commands

def cmd_check(args: argparse.Namespace, target: Path) -> None:
    print(f"仓库: {args.repo}")
    print(f"目标目录: {target}\n")

    try:
        remote_sha = get_remote_sha(args.repo)
    except Exception as e:
        print(f"✗ 获取远端版本失败: {e}")
        sys.exit(1)

    if remote_sha:
        print(f"远端版本: {remote_sha[:12]}")
    local_sha = get_local_sha(target)

    script = Path(sys.argv[0]).name
    download_cmd = f"python3 {script} --download"

    if local_sha is None:
        if not target.exists():
            print("本地: 未下载\n")
        elif is_complete(target):
            print("本地: 无版本记录（目录完整）\n")
            print("⚠ 状态未知")
        else:
            print("本地: 目录不完整\n")
            print("✗ 需要补齐")
        print(f"\n执行以下命令下载:")
        print(f"  {download_cmd}")
        return

    print(f"本地版本: {local_sha[:12]}\n")
    if remote_sha and local_sha != remote_sha:
        print(f"⬆ 有更新: {local_sha[:12]} → {remote_sha[:12]}")
        print(f"\n执行以下命令更新:")
        print(f"  {download_cmd}")
    else:
        print("✓ 已是最新")


def cmd_download(args: argparse.Namespace, target: Path) -> None:
    print(f"仓库: {args.repo}")
    print(f"目标目录: {target}\n")

    _ensure_modelscope()
    from modelscope import snapshot_download

    if target.exists():
        if is_complete(target):
            if args.force:
                print("--force: 清空后重新下载\n")
                shutil.rmtree(target)
            else:
                print("目录已完整，增量更新\n")
        else:
            print("目录不完整，补齐缺失文件\n")
    else:
        print("目标不存在，全量下载\n")

    print("下载中…（首次约 2GB，请耐心等待）")
    try:
        snapshot_download(
            args.repo,
            local_dir=str(target),
        )
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        print("  检查网络 / 是否需要代理")
        sys.exit(1)

    # 记录版本
    try:
        sha = get_remote_sha(args.repo)
        if sha:
            write_local_sha(target, sha)
            print(f"已记录版本: {sha[:12]} -> {target / REVISION_FILE}")
    except Exception:
        pass

    if target.exists():
        print(f"\n目录大小: {fmt_size(dir_size(target))}")
        files = sorted(p.name for p in target.iterdir() if p.is_file())
        print(f"文件清单 ({len(files)} 个):")
        for f in files[:10]:
            print(f"  - {f}")
        if len(files) > 10:
            print(f"  ... 共 {len(files)} 个文件")

    print(f"\n✓ 完成")

    if args.test:
        print("\n测试合成…")
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2
            import soundfile as sf
            import tempfile

            model = CosyVoice2(str(target))
            output = model.inference("<calm>静以修身。</calm>", stream=False)
            tmp = Path(tempfile.gettempdir()) / "cosyvoice_test.wav"
            for seg in output:
                sf.write(str(tmp), seg["tts_speech"], 24000)
                break
            print(f"✓ 测试通过 → {tmp}")
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            sys.exit(1)


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
