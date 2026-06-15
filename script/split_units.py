#!/usr/local/bin/python3.13
# -*- coding: utf-8 -*-
"""
切分成检索单元：读 A.1 JSONL → 按 tags 主题切换切分 → 补 summary/entities → 输出。

输入：data/chinese_clean/<课程>/<NNN-标题>.jsonl
      每行一个 JSON 段落对象：
      {"lesson","segment_id","t_start","t_end","text","tags"}

输出：data/chinese_units/<课程>/<NNN-标题>.jsonl
      每行一个检索单元 JSON 对象：
      {"unit_id","segment_ids","lesson","lesson_slug","t_start",
       "t_end","text","tags","summary","entities","source_refs"}

切分规则：
1. 主规则：前后两段 tags 有交集 → 合并；无交集 → 切一刀。
2. 字数微调：<200 字的单元尝试与后一单元合并（放宽交集规则）；
   >1000 字的单元在句号处再切（~800 字处）。
3. source_refs 依赖课文原文库，未就位时默认 []。

密钥：从 script/config.toml 读取（复用 clean_transcript 的配置）。

用法：
  python3 script/split_units.py                      # 处理全部
  python3 script/split_units.py --file xxx.jsonl     # 只处理单个文件
  python3 script/split_units.py --dry-run            # 不调模型，仅切分
  python3 script/split_units.py --sample-check 0.1   # 切分后随机抽检
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SRC = PROJECT_ROOT / "data" / "chinese_clean"
DEFAULT_DST = PROJECT_ROOT / "data" / "chinese_units"
CONFIG_FILE = SCRIPT_DIR / "config.toml"

# 字数控制
TARGET_MIN_CHARS = 200
TARGET_MAX_CHARS = 800
HARD_MAX_CHARS = 1000

# ------------------------------------------------------------------- 配置加载（复用 clean_transcript）

def load_config() -> dict:
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

SYSTEM_PROMPT = "你是语文课转录整理员，输出严格 JSON 格式，不要任何额外解释。"

META_PROMPT = """下面是语文老师一段讲解的文字。

请做三件事：
1. 写一句摘要（≤50字）。
2. 提取关键实体：人名、地名、课文名、术语。
3. 这段讲解对应了课文的哪些原文句子？把原文原句摘出来，标段落号（凭记忆即可）。

输出严格 JSON（不要 markdown 代码块，只输出一行 JSON）：
{"summary": "……", "entities": ["……", "……"], "source_refs": [{"para": 6, "quote": "他蹒跚地走到铁道边……"}]}

讲解原文：
"""


def call_deepseek(cfg: dict, user_prompt: str) -> str:
    import ssl
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
    # 跳过 SSL 验证（解决证书问题）
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=300, context=context) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def fill_meta(cfg: dict, text: str) -> dict:
    """调用模型补 summary + entities + source_refs。失败时返回占位值。"""
    try:
        content = call_deepseek(cfg, META_PROMPT + text)
        content = content.strip().removeprefix("```json").removesuffix("```").strip()
        obj = json.loads(content)
        src_refs = obj.get("source_refs", [])
        if not isinstance(src_refs, list):
            src_refs = []
        return {
            "summary": str(obj.get("summary", "")).strip()[:100],
            "entities": obj.get("entities", []) if isinstance(obj.get("entities"), list) else [],
            "source_refs": src_refs,
        }
    except Exception:
        return {"summary": text[:50] + "……", "entities": []}


# ------------------------------------------------------------------- 数据结构

class SegmentRow:
    """从 A.1 JSONL 读入的一个段落。"""

    __slots__ = ("lesson", "segment_id", "t_start", "t_end", "text", "tags")

    def __init__(self, obj: dict):
        self.lesson = obj["lesson"]
        self.segment_id = obj["segment_id"]
        self.t_start = obj["t_start"]
        self.t_end = obj["t_end"]
        self.text = obj["text"]
        self.tags = obj.get("tags", [])


class Unit:
    """一个检索单元。"""

    __slots__ = ("segments", "text", "t_start", "t_end",
                 "tags", "summary", "entities", "source_refs")

    def __init__(self, segs: list[SegmentRow]):
        self.segments = segs
        self.text = "\n".join(s.text for s in segs)
        self.t_start = segs[0].t_start
        self.t_end = segs[-1].t_end
        # 合并 tags（并集，去重，保留首次出现顺序）
        seen: set[str] = set()
        tags: list[str] = []
        for s in segs:
            for t in s.tags:
                if t not in seen:
                    seen.add(t)
                    tags.append(t)
        self.tags = tags
        self.summary = ""
        self.entities: list[str] = []
        self.source_refs: list[dict] = []

    @property
    def chars(self) -> int:
        return len(self.text)


# ------------------------------------------------------------------- 切分

def tags_overlap(a: list[str], b: list[str]) -> bool:
    """两段 tags 是否有交集。"""
    return bool(set(a) & set(b))


def split_by_tags(segs: list[SegmentRow]) -> list[list[SegmentRow]]:
    """按 tags 主题切换初次切分。"""
    if not segs:
        return []
    groups: list[list[SegmentRow]] = [[segs[0]]]
    for s in segs[1:]:
        if tags_overlap(groups[-1][-1].tags, s.tags):
            groups[-1].append(s)
        else:
            groups.append([s])
    return groups


def adjust_length(groups: list[list[SegmentRow]]) -> list[list[SegmentRow]]:
    """字数微调：太短合并，太长再切。"""
    # 1. 先处理太长的——在 ~800 字处的句号切
    expanded: list[list[SegmentRow]] = []
    SENT_BOUNDARY = re.compile(r"[。！？]")
    for g in groups:
        total = sum(len(s.text) for s in g)
        if total <= HARD_MAX_CHARS:
            expanded.append(g)
            continue
        # 把 g 拼成完整文本，在 ~800 字处的句号切分
        full = "\n".join(s.text for s in g)
        parts = _split_long_text(full, g)
        expanded.extend(parts)
    if not expanded:
        return expanded

    # 2. 太短的与后一单元合并（放宽：即使 tags 无交集也合，优先保证字数）
    merged: list[list[SegmentRow]] = [expanded[0]]
    for g in expanded[1:]:
        prev = merged[-1]
        prev_chars = sum(len(s.text) for s in prev)
        if prev_chars < TARGET_MIN_CHARS:
            # 合并到前一个，tags 合并
            prev.extend(g)
        else:
            merged.append(g)
    return merged


def _split_long_text(full_text: str, group: list[SegmentRow]) -> list[list[SegmentRow]]:
    """把超长单元按 ~800 字处的句号切成多块。
    需要用 segment 粒度重新组织（不能直接按字符切段）。"""
    # 简化：在 segment 粒度累计字数，找到合适断点
    result: list[list[SegmentRow]] = []
    current: list[SegmentRow] = []
    chars = 0
    for s in group:
        if current and chars + len(s.text) > TARGET_MAX_CHARS:
            result.append(current)
            current = [s]
            chars = len(s.text)
        else:
            current.append(s)
            chars += len(s.text)
    if current:
        result.append(current)
    return result


def collect_units(groups: list[list[SegmentRow]]) -> list[Unit]:
    """把分组转为 Unit 对象。跳过空单元。"""
    return [Unit(g) for g in groups if g]


# ------------------------------------------------------------------- 单文件处理

def split_one(src_jsonl: Path, out_jsonl: Path, cfg: dict, dry_run: bool) -> str:
    """切分单个 JSONL 文件。返回状态字符串。"""
    # 读入所有段落
    segs: list[SegmentRow] = []
    with src_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            segs.append(SegmentRow(json.loads(line)))
    if not segs:
        return "empty"

    lesson = segs[0].lesson
    # lesson_slug 从第一个 segment_id 提取（格式：NNN-Title_0003）
    slug = segs[0].segment_id.rsplit("_", 1)[0]

    # 切分
    groups = split_by_tags(segs)
    groups = adjust_length(groups)
    units = collect_units(groups)

    if dry_run:
        return f"dry: {len(segs)} 段 → {len(units)} 单元"

    # 补元数据（模型调用）
    for i, u in enumerate(units):
        meta = fill_meta(cfg, u.text)
        u.summary = meta["summary"]
        u.entities = meta["entities"]
        u.source_refs = meta.get("source_refs", [])

    # 写输出
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_jsonl.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for idx, u in enumerate(units):
            obj = {
                "unit_id": f"{slug}_u{idx:03d}",
                "segment_ids": [s.segment_id for s in u.segments],
                "lesson": lesson,
                "lesson_slug": slug,
                "t_start": round(u.t_start, 2),
                "t_end": round(u.t_end, 2),
                "text": u.text,
                "tags": u.tags,
                "summary": u.summary,
                "entities": u.entities,
                "source_refs": u.source_refs,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(out_jsonl)

    return f"{len(segs)} 段 → {len(units)} 单元"


# ------------------------------------------------------------------- 抽检

def sample_check(src_jsonl: Path, out_jsonl: Path, ratio: float = 0.1) -> None:
    """抽检切分结果：打印抽中的单元摘要，供人工查看切点是否合理。"""
    units: list[dict] = []
    with out_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                units.append(json.loads(line))
    if not units:
        return
    n = max(1, int(len(units) * ratio))
    sample = random.sample(units, min(n, len(units)))
    print(f"\n  [抽检] 从 {len(units)} 个单元中抽 {len(sample)} 个：")
    for u in sample:
        tags_str = ", ".join(u["tags"])
        preview = u["text"][:120].replace("\n", " ")
        print(f"    {u['unit_id']}  [{tags_str}]  {preview}……")
        if u["source_refs"]:
            for ref in u["source_refs"]:
                print(f"      ↳ 原文段落{ref.get('para','?')}: {ref.get('quote','')}")
    print()


# ------------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="切分检索单元：按 tags 切分 + 补元数据")
    p.add_argument("--src", type=Path, default=DEFAULT_SRC, help="A.1 清洗根目录")
    p.add_argument("--dst", type=Path, default=DEFAULT_DST, help="检索单元输出根目录")
    p.add_argument("--file", type=str, default=None,
                   help="只处理匹配该文件名的单个文件")
    p.add_argument("--dry-run", action="store_true",
                   help="不调模型，仅切分")
    p.add_argument("--force", action="store_true",
                   help="即使输出已存在也重新处理")
    p.add_argument("--sample-check", type=float, default=0.0,
                   help="抽检比例（0.0~1.0），切分完成后随机抽检")
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
        print("✗ 未找到 DeepSeek api_key。"
              "复制 script/config.example.toml 为 config.toml 填入，"
              "或设 DEEPSEEK_API_KEY。")
        sys.exit(1)

    jsonl_files = sorted(src.rglob("*.jsonl"))
    if args.file:
        jsonl_files = [f for f in jsonl_files
                       if f.name == args.file or str(f).endswith(args.file)]
    if not jsonl_files:
        print("没有匹配的 jsonl 文件。")
        return

    print(f"源目录  : {src}")
    print(f"目标目录: {dst}")
    print(f"待处理  : {len(jsonl_files)} 个文件"
          f"{'（dry-run）' if args.dry_run else ''}\n")

    ok = skip = fail = 0
    t0 = time.time()
    from datetime import datetime as _dt
    print(f"开始时间: {_dt.now().isoformat(timespec='seconds')}")
    for i, src_f in enumerate(jsonl_files, 1):
        rel = src_f.relative_to(src)
        out_f = dst / rel
        if not args.force and not args.dry_run \
                and out_f.exists() and out_f.stat().st_size > 0:
            skip += 1
            print(f"  ⊘ [{i}/{len(jsonl_files)}] {rel} 已存在，跳过")
            continue
        print(f"  ▶ [{i}/{len(jsonl_files)}] {rel} 开始"
              f" @ {_dt.now().isoformat(timespec='seconds')}", flush=True)
        t_file = time.time()
        try:
            msg = split_one(src_f, out_f, cfg, args.dry_run)
            dt_file = int(time.time() - t_file)
            ok += 1
            print(f"  ✓ [{i}/{len(jsonl_files)}] {rel}  {msg}  ({dt_file}s)")
            if args.sample_check > 0 and not args.dry_run:
                sample_check(src_f, out_f, args.sample_check)
        except Exception as e:
            fail += 1
            print(f"  ✗ [{i}/{len(jsonl_files)}] {rel}  {type(e).__name__}: {e}")

    dt = int(time.time() - t0)
    print(f"结束时间: {_dt.now().isoformat(timespec='seconds')}")
    print(f"完成: 成功 {ok}, 跳过 {skip}, 失败 {fail}, 耗时 {dt}s")


if __name__ == "__main__":
    main()
