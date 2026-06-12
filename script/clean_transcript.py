#!/usr/local/bin/python3.13
# -*- coding: utf-8 -*-
"""
转录清洗 pipeline：纠错 + 断句合并 + 结构化标签（一次大模型调用完成）。

输入：data/chinese_text/<课程>/<NNN-标题>.txt
      每行格式  start_time → end_time | 文本
      例：16.34s → 17.64s | 大家好,我是少新

输出：data/chinese_clean/<课程>/<NNN-标题>.jsonl
      每行一个 JSON 段落对象：
      {"lesson","segment_id","t_start","t_end","text","tags"}

核心：序号锚定法。送给模型的是「行号 + 纯文本」，模型只决定
「哪几行合成一段 + 段内纠错后的文字 + 该段标签」，并标出覆盖的行号区间 [a-b]；
时间戳由代码按行号区间从原始数据还原，模型绝不碰时间戳。
对每个分块校验区间「连续、不重叠、全覆盖」，不通过则重试，
重试仍失败则降级（该块保留原始行不合并），绝不静默吞行。

密钥：从 script/config.toml 读取（已 .gitignore），或环境变量
      DEEPSEEK_API_KEY 覆盖。复制 config.example.toml 起步。

用法：
  python3 script/clean_transcript.py                 # 清洗全部
  python3 script/clean_transcript.py --file xxx.txt  # 只清洗单个文件
  python3 script/clean_transcript.py --dry-run       # 不调模型，仅检查解析/分块
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # 低版本解释器回退到极简解析
    tomllib = None
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SRC = PROJECT_ROOT / "data" / "chinese_text"
DEFAULT_DST = PROJECT_ROOT / "data" / "chinese_clean"
CONFIG_FILE = SCRIPT_DIR / "config.toml"

# 一节课可能上千行，单块行数上限（控制上下文长度）
CHUNK_LINES = 120
# 在 CHUNK_LINES 之后最多再延伸这么多行，以便在自然停顿处切块
CHUNK_LOOKAHEAD = 20
# 认为是“自然停顿”的最小间隔（秒）：上一行 end 到下一行 start
PAUSE_GAP = 0.8
# 失败重试次数
MAX_RETRY = 2

# 步骤4 可选标签集合（design.md A.1 步骤4）
TAG_SET = ("导入", "背景", "字词", "句析", "提问", "互动", "情感", "总结", "其他")

# 常见错别字/专名词表（可按课程扩充）——放进 prompt 提示模型
HINT_VOCAB = (
    "世说新语 刘义庆 阮籍 刘伶 咏雪 陈太丘与友期 志人小说 "
    "南朝宋 魏晋 竹林七贤 嵇康 王羲之 谢安"
)


# ---------------------------------------------------------------- 数据结构


@dataclass
class Line:
    """原始转录的一行。"""

    start: float
    end: float
    text: str


@dataclass
class Segment:
    """合并纠错后的一段。"""

    t_start: float
    t_end: float
    text: str
    tags: list[str]


# ---------------------------------------------------------------- 配置加载


def load_config() -> dict:
    """读取 DeepSeek 配置。环境变量 DEEPSEEK_API_KEY 优先于配置文件。"""
    cfg = {
        "api_key": "",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    }
    if CONFIG_FILE.exists():
        ds = _read_toml_deepseek(CONFIG_FILE)
        for k in cfg:
            if ds.get(k):
                cfg[k] = ds[k]
    env_key = os.environ.get("DEEPSEEK_API_KEY")
    if env_key:
        cfg["api_key"] = env_key
    return cfg


def _read_toml_deepseek(path: Path) -> dict:
    """读取配置文件的 [deepseek] 段。优先用 tomllib，缺失时极简回退解析。"""
    if tomllib is not None:
        with path.open("rb") as f:
            return tomllib.load(f).get("deepseek", {})
    # 回退：只识别 key = "value" 行，够用于本配置
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


# ---------------------------------------------------------------- 解析

LINE_RE = re.compile(r"^\s*([\d.]+)\s*s\s*→\s*([\d.]+)\s*s\s*\|\s*(.*)$")


def parse_txt(path: Path) -> list[Line]:
    """解析转录 txt，返回 Line 列表（跳过空行/不合规行）。"""
    lines: list[Line] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        m = LINE_RE.match(raw)
        if not m:
            continue
        start, end, text = float(m.group(1)), float(m.group(2)), m.group(3).strip()
        if text:
            lines.append(Line(start, end, text))
    return lines


# ---------------------------------------------------------------- 分块


def split_chunks(lines: list[Line]) -> list[tuple[int, int]]:
    """把行切成若干块，返回每块的 [start_idx, end_idx]（闭区间，全局连续）。

    优先在自然停顿处断块：到达 CHUNK_LINES 后，向后看 CHUNK_LOOKAHEAD 行，
    在第一个间隔 >= PAUSE_GAP 的位置断开；找不到就在硬上限处断。
    """
    n = len(lines)
    chunks: list[tuple[int, int]] = []
    i = 0
    while i < n:
        soft_end = min(i + CHUNK_LINES - 1, n - 1)
        if soft_end >= n - 1:
            chunks.append((i, n - 1))
            break
        # 在 [soft_end, soft_end+lookahead] 找自然停顿断点
        cut = soft_end
        hard_end = min(soft_end + CHUNK_LOOKAHEAD, n - 1)
        for j in range(soft_end, hard_end):
            gap = lines[j + 1].start - lines[j].end
            if gap >= PAUSE_GAP:
                cut = j
                break
        else:
            cut = hard_end
        chunks.append((i, cut))
        i = cut + 1
    return chunks


# ---------------------------------------------------------------- 大模型


SYSTEM_PROMPT = "你是语文课转录整理员，输出严格遵守格式，不要任何额外解释。"


def build_user_prompt(lines: list[Line], base: int) -> str:
    """构造发给模型的用户消息：行号 + 纯文本（不含时间戳）。

    base 是该块首行在全局的行号；块内对模型呈现的行号用全局行号，
    便于解析时直接对回原始数据。
    """
    body = "\n".join(f"[{base + k}] {ln.text}" for k, ln in enumerate(lines))
    lo, hi = base, base + len(lines) - 1
    return (
        f"下面是《世说新语》精读课的语音识别结果，每行前是行号 [{lo}] 到 [{hi}]。\n"
        "请完成三件事：\n"
        "1. 纠正同音错别字，重点是书名/人名/典故，例如："
        "“诗说新语”→“世说新语”、“刘逸庆”→“刘义庆”、“软鸡”→“阮籍”、"
        "“流灵”→“刘伶”、“永雪”→“咏雪”、“智人小说”→“志人小说”。\n"
        "2. 把相邻碎句合并成通顺的段落，每段约 200~400 字，对应一个完整讲解动作。\n"
        "3. 给每段打 1~3 个标签，从这些里选："
        + "/".join(TAG_SET)
        + "，多个用逗号分隔。\n"
        "规则：\n"
        "- 保留口头禅和语气（“大家好”“那首先呢”“对吧”），只删无意义的“嗯/呃”。\n"
        "- 每段必须以 [起-止] 开头标出覆盖的行号区间，区间要连续、不重叠、"
        f"且完整覆盖 [{lo}] 到 [{hi}] 的全部行，不得遗漏。\n"
        "- 单独一行也写成 [n-n]。\n"
        "- 输出格式：每段一行，形如  [起-止] (标签1,标签2) 整理后的段落文字\n"
        f"【常见词表】{HINT_VOCAB}\n"
        "【转录】\n"
        f"{body}"
    )


def call_deepseek(cfg: dict, user_prompt: str) -> str:
    """调用 DeepSeek Chat（OpenAI 兼容）。返回模型文本内容。"""
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


# ---------------------------------------------------------------- 解析+校验模型输出

SEG_RE = re.compile(
    r"^\s*\[\s*(\d+)\s*-\s*(\d+)\s*\]\s*(?:[（(]([^）)]*)[）)])?\s*(.*)$"
)


def _norm_tags(raw: str) -> list[str]:
    """切分标签串，只保留 TAG_SET 内的合法标签，去重保序。"""
    out: list[str] = []
    for t in re.split(r"[,，、/\s]+", raw.strip()):
        t = t.strip()
        if t in TAG_SET and t not in out:
            out.append(t)
    return out or ["其他"]


def parse_model_output(content: str) -> list[tuple[int, int, list[str], str]]:
    """解析模型输出为 (a, b, tags, text) 列表。"""
    out: list[tuple[int, int, list[str], str]] = []
    for raw in content.splitlines():
        m = SEG_RE.match(raw)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        tags = _norm_tags(m.group(3) or "")
        text = m.group(4).strip()
        if text:
            out.append((a, b, tags, text))
    return out


def validate_coverage(
    segs: list[tuple[int, int, list[str], str]], lo: int, hi: int
) -> bool:
    """校验区间连续、不重叠、完整覆盖 [lo, hi]。"""
    if not segs:
        return False
    ordered = sorted(segs, key=lambda x: x[0])
    cursor = lo
    for a, b, *_ in ordered:
        if a != cursor or b < a:
            return False
        cursor = b + 1
    return cursor == hi + 1


def to_segments(
    parsed: list[tuple[int, int, list[str], str]], lines: list[Line], base: int
) -> list[Segment]:
    """用行号区间从原始 lines 还原时间戳，生成 Segment。"""
    segs: list[Segment] = []
    for a, b, tags, text in sorted(parsed, key=lambda x: x[0]):
        la, lb = a - base, b - base
        segs.append(Segment(lines[la].start, lines[lb].end, text, tags))
    return segs


def fallback_segments(lines: list[Line]) -> list[Segment]:
    """降级：不合并，原始行逐行（只去掉时间戳，文本不纠错，标签留空）。"""
    return [Segment(ln.start, ln.end, ln.text, []) for ln in lines]


# ---------------------------------------------------------------- 单文件处理


def clean_one(src_txt: Path, out_jsonl: Path, cfg: dict, dry_run: bool) -> str:
    """清洗单个文件。返回状态字符串。"""
    lines = parse_txt(src_txt)
    if not lines:
        return "empty"

    chunks = split_chunks(lines)
    lesson = src_txt.parent.name
    slug = src_txt.stem

    if dry_run:
        return f"dry: {len(lines)} 行 → {len(chunks)} 块"

    all_segs: list[Segment] = []
    degraded = 0
    for ci, (a, b) in enumerate(chunks):
        block = lines[a : b + 1]
        prompt = build_user_prompt(block, a)
        segs: list[Segment] | None = None
        for attempt in range(MAX_RETRY + 1):
            try:
                content = call_deepseek(cfg, prompt)
            except (urllib.error.URLError, KeyError, TimeoutError) as e:
                print(f"      块{ci} 调用失败(尝试{attempt + 1}): {e}", flush=True)
                time.sleep(2 * (attempt + 1))
                continue
            parsed = parse_model_output(content)
            if validate_coverage(parsed, a, b):
                segs = to_segments(parsed, lines, a)
                break
            print(f"      块{ci} 区间校验未通过(尝试{attempt + 1})，重试", flush=True)
        if segs is None:
            # 重试用尽，降级保留原始行
            segs = fallback_segments(block)
            degraded += 1
        all_segs.extend(segs)

    # 写 JSONL（先写临时文件再原子替换，避免半截文件被当成已完成）
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_jsonl.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for idx, seg in enumerate(all_segs):
            obj = {
                "lesson": lesson,
                "segment_id": f"{slug}_{idx:04d}",
                "t_start": round(seg.t_start, 2),
                "t_end": round(seg.t_end, 2),
                "text": seg.text,
                "tags": seg.tags,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(out_jsonl)

    msg = f"{len(lines)} 行 → {len(all_segs)} 段"
    if degraded:
        msg += f"（{degraded}/{len(chunks)} 块降级）"
    return msg


# ---------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="转录清洗：纠错 + 断句合并")
    p.add_argument("--src", type=Path, default=DEFAULT_SRC, help="转录 txt 根目录")
    p.add_argument("--dst", type=Path, default=DEFAULT_DST, help="清洗输出根目录")
    p.add_argument("--file", type=str, default=None,
                   help="只处理匹配该文件名/相对路径的单个文件")
    p.add_argument("--dry-run", action="store_true",
                   help="不调模型，仅检查解析与分块")
    p.add_argument("--force", action="store_true",
                   help="即使输出已存在也重新处理")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()

    if not src.exists():
        print(f"✗ 源目录不存在: {src}")
        sys.exit(1)

    cfg = load_config()
    if not args.dry_run and not cfg["api_key"]:
        print("✗ 未找到 DeepSeek api_key。请复制 script/config.example.toml "
              "为 script/config.toml 填入，或设置环境变量 DEEPSEEK_API_KEY。")
        sys.exit(1)

    txt_files = sorted(src.rglob("*.txt"))
    if args.file:
        txt_files = [f for f in txt_files
                     if f.name == args.file or str(f).endswith(args.file)]
    if not txt_files:
        print("没有匹配的 txt 文件。")
        return

    print(f"源目录  : {src}")
    print(f"目标目录: {dst}")
    print(f"待处理  : {len(txt_files)} 个文件"
          f"{'（dry-run）' if args.dry_run else ''}\n")

    ok = skip = fail = 0
    t0 = time.time()
    from datetime import datetime as _dt
    print(f"开始时间: {_dt.now().isoformat(timespec='seconds')}")
    for i, txt in enumerate(txt_files, 1):
        rel = txt.relative_to(src)
        out_jsonl = dst / rel.with_suffix(".jsonl")
        if not args.force and not args.dry_run \
                and out_jsonl.exists() and out_jsonl.stat().st_size > 0:
            skip += 1
            print(f"  ⊘ [{i}/{len(txt_files)}] {rel} 已存在，跳过")
            continue
        print(f"  ▶ [{i}/{len(txt_files)}] {rel} 开始 @ {_dt.now().isoformat(timespec='seconds')}", flush=True)
        t_file = time.time()
        try:
            msg = clean_one(txt, out_jsonl, cfg, args.dry_run)
            dt_file = int(time.time() - t_file)
            ok += 1
            print(f"  ✓ [{i}/{len(txt_files)}] {rel}  {msg}  ({dt_file}s)")
        except Exception as e:  # noqa: BLE001  顶层兜底，单文件失败不影响整体
            fail += 1
            print(f"  ✗ [{i}/{len(txt_files)}] {rel}  {type(e).__name__}: {e}")

    dt = int(time.time() - t0)
    print(f"结束时间: {_dt.now().isoformat(timespec='seconds')}")
    print(f"完成: 成功 {ok}, 跳过 {skip}, 失败 {fail}, 耗时 {dt}s")


if __name__ == "__main__":
    main()
