#!/usr/local/bin/python3.13
# -*- coding: utf-8 -*-
"""
风格画像 · 第一条腿：程序统计（不调模型，纯代码跑）。

输入：data/chinese_clean/<课程>/*.jsonl
      每行 A.1 段落 JSON（segment_id, t_start, t_end, text, tags）

输出：data/style/style_stats.json  结构化统计数据

依赖：
  pip3 install jieba

用法：
  python3 script/style_stats.py                    # 统计全部课程
  python3 script/style_stats.py --out stats.md     # 输出 Markdown
  python3 script/style_stats.py --top-k 30         # 口头禅取前30
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jieba

# 语气词/虚词类停用词（分词后过滤）
STOP_WORDS = set("的了呢吗吧啊呀哦嗯呃哈嘛哇嘿还又也才就都却".split())
# 标点
PUNCT_RE = re.compile(r"[，。！？；：、\n\r\s]")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SRC = PROJECT_ROOT / "data" / "chinese_clean"
DEFAULT_OUT = PROJECT_ROOT / "data" / "style" / "style_stats.json"


# ------------------------------------------------------------------- 加载数据

def load_segments(src: Path) -> list[dict]:
    """读所有 A.1 JSONL，返回 segment 列表。"""
    segs: list[dict] = []
    for f in sorted(src.rglob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            segs.append(json.loads(line))
    return segs


# ------------------------------------------------------------------- 1. 高频口头禅

def count_phrases(texts: list[str], top_k: int = 20) -> list[dict]:
    """jieba 分词 → 单字词 + 2~3词短语频次，过滤虚词类停用词。"""
    word_freq: Counter = Counter()
    bigram_freq: Counter = Counter()

    for t in texts:
        for sent in PUNCT_RE.split(t):
            sent = sent.strip()
            if not sent:
                continue
            # jieba 分词
            words = [w.strip() for w in jieba.lcut(sent) if w.strip()]
            # 单字词频率（过滤停用词）
            for w in words:
                if w not in STOP_WORDS:
                    word_freq[w] += 1
            # 连续两词搭为口头禅短语（如"我们/来看/一下"）
            for i in range(len(words) - 1):
                bi = words[i] + words[i + 1]
                if len(bi) >= 3:
                    bigram_freq[bi] += 1

    total_segs = len(texts)
    # 取词 + 短语合并，按频次混合排列
    combined: list[tuple[str, int]] = []
    combined.extend(bigram_freq.most_common(top_k * 2))
    combined.extend(word_freq.most_common(top_k))
    # 去重（短语优先），保留 top_k
    seen: set[str] = set()
    result: list[dict] = []
    for phrase, count in sorted(combined, key=lambda x: -x[1]):
        if phrase in seen or len(phrase) < 2:
            continue
        seen.add(phrase)
        result.append({
            "phrase": phrase, "count": count,
            "ratio": round(count / total_segs, 3),
        })
        if len(result) >= top_k:
            break

    return result


# ------------------------------------------------------------------- 2. 开场白 / 收尾句

def extract_openers_closers(segs: list[dict]) -> dict:
    """每节课首段首句、末段末句。"""
    # 按 lesson_slug（从 segment_id 提取）分组
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in segs:
        slug = s["segment_id"].rsplit("_", 1)[0]
        groups[slug].append(s)
    groups = dict(sorted(groups.items()))

    openers: list[str] = []
    closers: list[str] = []
    for slug, group in groups.items():
        if group:
            openers.append(_first_sent(group[0]["text"]))
            closers.append(_last_sent(group[-1]["text"]))

    return {
        "openers": openers[:30],
        "closers": closers[:30],
        "sample_count": len(openers),
    }


def _first_sent(text: str) -> str:
    m = re.search(r"^.*?[。！？，]", text)
    return m.group(0).strip() if m else text[:40]


def _last_sent(text: str) -> str:
    # 去掉末尾标点后取最后一句
    text = text.rstrip("。！？")
    m = re.search(r"[。！？]([^。！？]*)$", text)
    return (m.group(1) + "。").strip() if m else text[-40:]


# ------------------------------------------------------------------- 3. 提问句式

QUESTION_INDICATORS = re.compile(r"[吗呢吧]|什么|怎么|为什么|如何|哪[个些]|谁|几[个次]")


def count_questions(texts: list[str]) -> dict:
    """统计带问号的句子，取高频句式模板。"""
    questions: list[str] = []
    for t in texts:
        for sent in re.split(r"[。！\n]", t):
            sent = sent.strip()
            if "？" not in sent and "?" not in sent:
                continue
            if not QUESTION_INDICATORS.search(sent):
                continue
            questions.append(sent)

    # 句式模板：截取前 N 个字做聚类
    patterns: Counter = Counter()
    for q in questions:
        # 取问句开头关键部分做模板
        pattern = q[:15].rstrip("，。！？")
        patterns[pattern] += 1

    total = len(texts)
    return {
        "question_count": len(questions),
        "question_ratio": round(len(questions) / total, 3) if total else 0,
        "top_patterns": [
            {"pattern": p, "count": c}
            for p, c in patterns.most_common(15)
        ],
        "samples": questions[:20],
    }


# ------------------------------------------------------------------- 4. 句长 / 节奏

def sentence_stats(texts: list[str]) -> dict:
    """句子长度分布、短中长句比例。"""
    lengths: list[int] = []
    for t in texts:
        for sent in re.split(r"[。！？\n]", t):
            sent = sent.strip()
            if sent:
                lengths.append(len(sent))

    if not lengths:
        return {}

    def percentile(lst: list[int], p: float) -> float:
        s = sorted(lst)
        return s[int(len(s) * p)]

    return {
        "avg_len": round(sum(lengths) / len(lengths), 1),
        "median_len": percentile(lengths, 0.5),
        "p25_len": percentile(lengths, 0.25),
        "p75_len": percentile(lengths, 0.75),
        "short_ratio": round(
            sum(1 for l in lengths if l <= 20) / len(lengths), 3
        ),  # ≤20字
        "medium_ratio": round(
            sum(1 for l in lengths if 20 < l <= 50) / len(lengths), 3
        ),
        "long_ratio": round(
            sum(1 for l in lengths if l > 50) / len(lengths), 3
        ),
        "total_sentences": len(lengths),
    }


# ------------------------------------------------------------------- 5. 互动密度

def interaction_density(segs: list[dict]) -> dict:
    """按时间戳算每分钟互动（提问/呼唤学生）次数。"""
    # 互动关键词
    interactive_re = re.compile(
        r"[？?]|同学们|大家|来[，,]?我们|你们觉得|你说[说看]|想一想|体会一下"
    )

    total_duration = 0.0
    interactions = 0
    for s in segs:
        dur = s["t_end"] - s["t_start"]
        if dur > 0:
            total_duration += dur
        if interactive_re.search(s["text"]):
            interactions += 1

    minutes = total_duration / 60.0 if total_duration > 0 else 0
    return {
        "total_duration_min": round(minutes, 1),
        "interaction_count": interactions,
        "per_minute": round(interactions / minutes, 1) if minutes > 0 else 0,
        "total_segments": len(segs),
    }


# ------------------------------------------------------------------- 6. 标签分布

def tag_distribution(segs: list[dict]) -> list[dict]:
    """各标签出现频次。"""
    counter: Counter = Counter()
    for s in segs:
        for t in s.get("tags", []):
            counter[t] += 1
    total = max(len(segs), 1)
    return [
        {"tag": t, "count": c, "ratio": round(c / total, 3)}
        for t, c in counter.most_common()
    ]


# ------------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="风格画像 · 程序统计")
    p.add_argument("--src", type=Path, default=DEFAULT_SRC,
                   help="A.1 清洗 JSONL 目录")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="输出文件（json/md）")
    p.add_argument("--top-k", type=int, default=20,
                   help="口头禅取前K条")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = args.src.resolve()

    if not src.exists():
        print(f"✗ 源目录不存在: {src}")
        sys.exit(1)

    segs = load_segments(src)
    if not segs:
        print("没有找到 A.1 清洗数据。")
        return
    print(f"已加载 {len(segs)} 段\n")

    texts = [s["text"] for s in segs]

    result = {
        "source": str(src),
        "segment_count": len(segs),
        "高頻口頭禪": count_phrases(texts, args.top_k),
        "开场收尾": extract_openers_closers(segs),
        "提问句式": count_questions(texts),
        "句长节奏": sentence_stats(texts),
        "互动密度": interaction_density(segs),
        "标签分布": tag_distribution(segs),
    }

    if args.out.suffix == ".md":
        _write_md(result, args.out)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"✓ 输出: {args.out}")


def _write_md(result: dict, out: Path) -> None:
    """输出 Markdown 格式（方便阅读）。"""
    lines = [
        f"# 风格画像 · 程序统计",
        f"来源: {result['source']}",
        f"段落数: {result['segment_count']}\n",
        "---\n",
        "## 高頻口頭禪",
        "| 短语 | 次数 | 覆盖率 |",
        "|------|------|--------|",
    ]
    for p in result["高頻口頭禪"]:
        lines.append(f"| {p['phrase']} | {p['count']} | {p['ratio']} |")

    oc = result["开场收尾"]
    lines.extend([
        "\n## 开场白",
        *(f"- {o}" for o in oc["openers"][:15]),
        "\n## 收尾句",
        *(f"- {c}" for c in oc["closers"][:15]),
        "\n## 提问句式",
        f"问句总数: {result['提问句式']['question_count']}  (覆盖率 {result['提问句式']['question_ratio']})",
        "| 句式模板 | 次数 |",
        "|----------|------|",
    ])
    for p in result["提问句式"]["top_patterns"]:
        lines.append(f"| {p['pattern']} | {p['count']} |")

    sl = result["句长节奏"]
    lines.extend([
        "\n## 句长节奏",
        f"平均句长: {sl.get('avg_len', 'N/A')}字",
        f"中位数: {sl.get('median_len', 'N/A')}字",
        f"短句(≤20字): {sl.get('short_ratio', 'N/A')}",
        f"中句(20~50字): {sl.get('medium_ratio', 'N/A')}",
        f"长句(>50字): {sl.get('long_ratio', 'N/A')}",
    ])

    id_ = result["互动密度"]
    lines.extend([
        "\n## 互动密度",
        f"总时长: {id_.get('total_duration_min', 'N/A')} 分钟",
        f"互动次数: {id_.get('interaction_count', 'N/A')}",
        f"每分钟互动: {id_.get('per_minute', 'N/A')} 次",
    ])

    lines.extend(["\n## 标签分布", "| 标签 | 次数 | 覆盖率 |", "|------|------|--------|"])
    for t in result["标签分布"]:
        lines.append(f"| {t['tag']} | {t['count']} | {t['ratio']} |")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
