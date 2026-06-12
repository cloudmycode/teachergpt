#!/usr/local/bin/python3.13
# -*- coding: utf-8 -*-
"""
风格画像 · 第二条腿：大模型归纳（统计算不出的"套路"靠它）。

输入：
  - data/style/style_stats.json（第一条腿产出，程序统计数据）
  - data/chinese_clean/<课程>/*.jsonl（A.1 清洗段落）

输出：data/style/style_profile.json  结构化风格档案

密钥：复用 script/config.toml 的 DeepSeek 配置。

用法：
  # 单批归纳
  python3 script/style_profile.py

  # 多批：每批输出独立 batch 文件，不自动合并
  python3 script/style_profile.py --batches 3 --samples 50

  # 确认后手动合并
  python3 script/style_profile.py --merge
  python3 script/style_profile.py --merge --threshold 0.6

  python3 script/style_profile.py --dry-run          # 仅打印 prompt，不调模型
"""

import argparse
import json
import random
import re
from collections import Counter
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SRC = PROJECT_ROOT / "data" / "chinese_clean"
DEFAULT_STATS = PROJECT_ROOT / "data" / "style" / "style_stats.json"
DEFAULT_OUT = PROJECT_ROOT / "data" / "style" / "style_profile.json"
CONFIG_FILE = SCRIPT_DIR / "config.toml"

SYSTEM_PROMPT = "你是教学风格分析师，输出严格遵守 JSON 格式。"


# ------------------------------------------------------------------- 配置加载（复用）

def load_config() -> dict:
    cfg = {"api_key": "", "base_url": "https://api.deepseek.com", "model": "deepseek-chat"}
    if CONFIG_FILE.exists():
        ds = _read_toml_deepseek(CONFIG_FILE)
        for k in cfg:
            if ds.get(k):
                cfg[k] = ds[k]
    env_key = __import__("os").environ.get("DEEPSEEK_API_KEY")
    if env_key:
        cfg["api_key"] = env_key
    return cfg


def _read_toml_deepseek(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = None
    if tomllib is not None:
        with path.open("rb") as f:
            return tomllib.load(f).get("deepseek", {})
    out: dict = {}
    in_section = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_section = line[1:-1].strip() == "deepseek"
            continue
        if in_section and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


# ------------------------------------------------------------------- 模型调用

def call_deepseek(cfg: dict, user_prompt: str) -> str:
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['api_key']}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


# ------------------------------------------------------------------- 构建 Prompt

def build_prompt(stats: dict, samples: list[str]) -> str:
    """组装归纳风格的用户 prompt。"""
    stats_summary = _format_stats(stats)
    sample_text = "\n\n".join(
        f"[片段{i + 1}] {s}" for i, s in enumerate(samples)
    )
    return f"""你是教学风格分析师。下面是某语文老师的课堂转录统计和若干真实讲解片段。
请归纳他的教学风格，严格按给定 JSON 结构输出，每条结论必须有依据，不要编造。

【程序统计】
{stats_summary}

【真实片段】
{sample_text}

【输出 JSON 结构（只输出 JSON，不要任何解释）】
{{
  "persona": "一句话概括人格化风格（如：亲切、重情感体验、爱追问）",
  "口头禅": ["短语1", "短语2"],
  "开场套路": "开场白常见模式",
  "讲解结构": ["步骤1", "步骤2", "步骤3"],
  "提问方式": "提问特征概括",
  "举例偏好": "举例素材来源与方式",
  "情感表达": "情感类讲解的语言特征",
  "句式特征": "句子节奏、用词倾向",
  "禁忌": ["不会做的事1", "不会做的事2"]
}}
"""


def _format_stats(stats: dict) -> str:
    """把 stats JSON 转成简洁摘要。"""
    lines = []
    # 口头禅
    phrases = stats.get("高頻口頭禪", [])[:10]
    if phrases:
        tops = "、".join(
            f"{p['phrase']}({p['count']}次)" for p in phrases
        )
        lines.append(f"高频口头禅: {tops}")

    # 提问
    qs = stats.get("提问句式", {})
    lines.append(
        f"问句数: {qs.get('question_count', 'N/A')}  (覆盖率 {qs.get('question_ratio', 'N/A')})"
    )

    # 句长
    sl = stats.get("句长节奏", {})
    lines.append(
        f"句长: 平均{sl.get('avg_len', 'N/A')}字, "
        f"短句(≤20字){sl.get('short_ratio', 'N/A')}, "
        f"中句{sl.get('medium_ratio', 'N/A')}, "
        f"长句{sl.get('long_ratio', 'N/A')}"
    )

    # 互动
    id_ = stats.get("互动密度", {})
    lines.append(
        f"互动: 总{id_.get('interaction_count', 'N/A')}次, "
        f"每分钟{id_.get('per_minute', 'N/A')}次"
    )

    # 标签
    tags = stats.get("标签分布", [])[:8]
    if tags:
        ts = "、".join(f"{t['tag']}({t.get('ratio', 'N/A')})" for t in tags)
        lines.append(f"标签分布: {ts}")

    # 开场收尾
    oc = stats.get("开场收尾", {})
    if oc.get("openers"):
        lines.append("典型开场: " + "; ".join(oc["openers"][:5]))
    if oc.get("closers"):
        lines.append("典型收尾: " + "; ".join(oc["closers"][:5]))

    return "\n".join(lines)


# ------------------------------------------------------------------- 抽样

def load_samples(src: Path, stats: dict, n: int = 30) -> list[str]:
    """从 A.1 中随机抽 N 段真实讲解。优先覆盖不同标签。"""
    segs = []
    for f in sorted(src.rglob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            segs.append(json.loads(line))

    if len(segs) <= n:
        return [s["text"] for s in segs]

    # 按标签分层抽样，保证各类都有覆盖
    tag_groups: dict[str, list[str]] = {}
    for s in segs:
        for t in s.get("tags", []):
            tag_groups.setdefault(t, []).append(s["text"])
        tag_groups.setdefault("_other", []).append(s["text"])

    pool: list[str] = []
    for tag, texts in sorted(tag_groups.items()):
        k = max(1, int(n * len(texts) / len(segs)))
        pool.extend(random.sample(texts, min(k, len(texts))))

    random.shuffle(pool)
    return pool[:n]


# ------------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="风格画像 · 大模型归纳")
    p.add_argument("--stats", type=Path, default=DEFAULT_STATS,
                   help="第一条腿统计 JSON")
    p.add_argument("--src", type=Path, default=DEFAULT_SRC,
                   help="A.1 清洗 JSONL 目录")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="输出风格档案 JSON")
    p.add_argument("--samples", type=int, default=30,
                   help="每批抽样段数")
    p.add_argument("--batches", type=int, default=1,
                   help="分批归纳批数（每批输出独立文件）")
    p.add_argument("--merge", action="store_true",
                   help="读取已生成的 batch 文件做合并（不调模型）")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="合并阈值：条目至少在 N 比例的批次中出现才保留（默认0.5）")
    p.add_argument("--dry-run", action="store_true",
                   help="仅打印 prompt，不调模型")
    return p.parse_args()


# ------------------------------------------------------------------- 合并多批次结果

MERGE_KEYS_LIST = ["口头禅", "讲解结构", "禁忌"]
MERGE_KEYS_STR = ["persona", "开场套路", "提问方式", "举例偏好",
                  "情感表达", "句式特征"]


def merge_profiles(profiles: list[dict], threshold: float = 0.5) -> dict:
    """多批次结果合并：列表项取高频，字符串取众数。"""
    if not profiles:
        return {}
    if len(profiles) == 1:
        return profiles[0]

    min_count = max(1, int(len(profiles) * threshold))
    merged: dict = {}

    for key in MERGE_KEYS_LIST:
        counter: Counter = Counter()
        for p in profiles:
            items = p.get(key, [])
            if isinstance(items, list):
                for item in items:
                    counter[str(item)] += 1
        merged[key] = [
            item for item, cnt in counter.most_common() if cnt >= min_count
        ]

    for key in MERGE_KEYS_STR:
        counter: Counter = Counter()
        for p in profiles:
            val = p.get(key, "")
            if isinstance(val, str) and val.strip():
                counter[val.strip()] += 1
        merged[key] = counter.most_common(1)[0][0] if counter else ""

    return merged


def find_batch_files(out: Path) -> list[Path]:
    """找 data/style_profile_batch_*.json 文件。"""
    pattern = out.stem + "_batch_"
    parent = out.parent
    files = sorted(
        f for f in parent.glob(pattern + "*.json")
        if re.match(rf"^{re.escape(pattern)}\d+\.json$", f.name)
    )
    return files


def main() -> None:
    args = parse_args()

    # --- 合并模式：只读已有 batch 文件，不调模型 ---
    if args.merge:
        batch_files = find_batch_files(args.out)
        if not batch_files:
            print(f"✗ 未找到 batch 文件（{args.out.parent}/"
                  f"{args.out.stem}_batch_*.json）")
            sys.exit(1)
        profiles = [json.loads(f.read_text(encoding="utf-8")) for f in batch_files]
        print(f"读取 {len(profiles)} 个 batch 文件:")
        for f in batch_files:
            print(f"  - {f.name}")
        merged = merge_profiles(profiles, args.threshold)
        args.out.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n✓ 合并输出: {args.out}")
        return

    # --- 归纳模式：逐批调模型 ---
    if not args.stats.exists():
        print(f"✗ 统计文件不存在: {args.stats}")
        print(f"  请先运行: python3 script/style_stats.py")
        sys.exit(1)

    stats = json.loads(args.stats.read_text(encoding="utf-8"))

    if args.dry_run:
        samples = load_samples(args.src, stats, args.samples)
        prompt = build_prompt(stats, samples)
        print(f"=== PROMPT ({len(samples)}段样本) ===")
        print(prompt[:3000], "\n... (截断)")
        return

    cfg = load_config()
    if not cfg["api_key"]:
        print("✗ 未找到 DeepSeek api_key。")
        sys.exit(1)

    print(f"统计来源: {args.stats}")
    print(f"抽样段数: {args.samples} × {args.batches} 批")
    print(f"调用 DeepSeek ({cfg['model']}) ...\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ok = 0

    for bi in range(args.batches):
        samples = load_samples(args.src, stats, args.samples)
        prompt = build_prompt(stats, samples)
        print(f"  [{bi + 1}/{args.batches}] 模型归纳中 ...", flush=True)
        try:
            content = call_deepseek(cfg, prompt)
        except Exception as e:
            print(f"  ✗ 第{bi + 1}批调用失败: {e}")
            continue
        content = content.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            profile = json.loads(content)
        except json.JSONDecodeError:
            print(f"  ⚠ 第{bi + 1}批输出非 JSON，保存原文")
            profile = {"_raw": content}

        batch_out = args.out.parent / f"{args.out.stem}_batch_{bi + 1:03d}.json"
        batch_out.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        ok += 1

    if ok == 0:
        print("✗ 所有批次均失败")
        sys.exit(1)

    dt = int(time.time() - t0)
    print(f"\n✓ {ok}/{args.batches} 批完成，输出到 "
          f"{args.out.stem}_batch_*.json  ({dt}s)")

    if args.batches > 1:
        merge_cmd = f"python3 script/style_profile.py --merge"
        if args.threshold != 0.5:
            merge_cmd += f" --threshold {args.threshold}"
        print(f"确认后执行合并:")
        print(f"  {merge_cmd}")


if __name__ == "__main__":
    main()
