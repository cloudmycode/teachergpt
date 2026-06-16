#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按段落分批生成讲解时间轴（设计文档 §11 核心管线）。

输入：课文查询（如"诫子书 全文"）
输出：
  data/timelines/{课文名}/script.json         讲稿（narration）
  data/timelines/{课文名}/timeline.json        时间轴（narration + audio + keyword_timings）

流程：
  1. 复用 generate.py 的意图解析 + 课文事实获取 → sentences / keywords
  2. 按段落分批：每批处理一个段落（多句），段落间传递上下文
  3. 对每句调 DeepSeek 生成 narration（带上下文和上一句结尾）
  4. edge-tts 合成讲解音频（免费云端 TTS）
  5. 线性估时对齐 → keyword_timings
  6. 输出完整 timeline.json

段落分批策略：
  - 空行分隔段落
  - 空行不足时，按句数分批（默认每批最多 5 句）
  - 段落间传递上一段最后一句的 narration 作为上下文

用法：
  python3 script/build_timeline.py "诫子书 全文"
  python3 script/build_timeline.py "木兰词 全文" --batch-size 5
  python3 script/build_timeline.py "诫子书" --skip-tts
  python3 script/build_timeline.py "诫子书" --skip-align
  python3 script/build_timeline.py "诫子书" --skip-tts --skip-align
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import generate  # 复用意图解析、课文事实、检索、风格加载

DATA_DIR = PROJECT_ROOT / "data"
TIMELINE_DIR = DATA_DIR / "timelines"

# ------------------------------------------------------------------- 逐句生成

PER_INTRO_USER = """{few_shot}
{context}

请为这节课写一段**开头导入语**（50-100字）。

要求：
1. 用第一人称"我"的课堂口吻。
2. 自然引入本课主题，可以抛出一个有趣的问题或悬念。
3. 体现风格设定中的口头禅、提问方式。
4. 让学生想继续听下去。

请直接输出导入语，不要加标题，不要输出 JSON。"""

PER_SENTENCE_USER = """{few_shot}
{context}

现在请你讲解以下这句：

原句：{sentence_text}
{keyword_info}

要求：
1. 用第一人称"我"的课堂口吻。
2. 如本句有重点词，逐个解读：读音、本义、语境义、用法。释义以背景信息为准。
3. 然后把这句话串讲一遍，点出它好在哪、要体会什么。
4. 上一句结尾为参考，开头自然过渡衔接，可以用不同的过渡方式：
   - 直接承接上文："刚才我们讲了...接下来"
   - 提出问题："那这个字是什么意思呢？"
   - 简单过渡："接下来看""来看看这句"
   - 也可以不用过渡语，直接开始讲解
   **注意：不要每次都用"好"开头，要变化多样。**
5. 不限字数，该展开就展开。
6. **重要**：以下是本句所有的重点词，你必须逐个讲解，并且在讲到该词时用方括号 [ ] 标记。
   例如讲解"静"字时写成 [静]。注意：仅在讲解该词含义时标记，
   开头引用原文（如"静以修身"）时不要标记。必须标记所有重点词，一个都不能漏。

请直接输出讲解文本，不要加标题，不要输出 JSON。"""


def _build_system(style: dict) -> str:
    lines = ["你是语文老师的课堂克隆，请严格遵循以下风格设定："]
    if style.get("persona"):
        lines.append(f"人格：{style['persona']}")
    for key in [
        "口头禅", "开场套路", "提问方式", "举例偏好",
        "句式特征", "禁忌", "讲解结构",
    ]:
        val = style.get(key)
        if val:
            val = "、".join(val) if isinstance(val, list) else val
            lines.append(f"{key}：{val}")
    return "\n".join(lines)


def _build_context(
    sentence_idx: int,
    total: int,
    facts: dict,
    prev_narration: str,
) -> str:
    parts = []

    author = facts.get("author", "")
    source = facts.get("source", "")
    dynasty = facts.get("dynasty", "")
    parts.append(
        f"你正在讲{f'《{source}》' if source else '课文'}（{author}，{dynasty}）。"
    )
    if facts.get("synopsis"):
        parts.append(f"全文大意：{facts['synopsis']}")

    sentences = facts.get("sentences") or []
    if sentences:
        parts.append("\n课文全文（共 {} 句）：".format(len(sentences)))
        for i, s in enumerate(sentences, 1):
            marker = " ← 当前要讲" if i == sentence_idx else ""
            parts.append(f"  {i}. {s.get('text', '')}{marker}")

    if prev_narration and sentence_idx > 1:
        tail = prev_narration[-120:]
        parts.append(
            f"\n上一句你讲完时的最后一段话：\"{tail}\"\n"
            f"请从这里自然往下接，开头用老师的过渡语。"
        )

    return "\n".join(parts)


def _build_intro_context(facts: dict) -> str:
    """构建开头导入语的上下文。"""
    parts = []
    
    author = facts.get("author", "")
    source = facts.get("source", "")
    dynasty = facts.get("dynasty", "")
    parts.append(
        f"你正在开始讲{f'《{source}》' if source else '课文'}（{author}，{dynasty}）。"
    )
    if facts.get("synopsis"):
        parts.append(f"全文大意：{facts['synopsis']}")
    
    # 添加前几句原文作为参考
    sentences = facts.get("sentences") or []
    if sentences:
        parts.append(f"\n课文开头几句：")
        for i, s in enumerate(sentences[:3], 1):
            parts.append(f"  {i}. {s.get('text', '')}")
    
    return "\n".join(parts)


def build_per_sentence_prompt(
    sentence: dict,
    sentence_idx: int,
    total: int,
    facts: dict,
    style: dict,
    segments: list[dict],
    prev_narration: str,
) -> tuple[str, str]:
    system = _build_system(style)
    context = _build_context(sentence_idx, total, facts, prev_narration)

    kws = sentence.get("keywords") or []
    keyword_info = ""
    if kws:
        keyword_info = "重点词：" + "、".join(
            f"{k['word']}({k['note']})" for k in kws
        )

    few_shot = ""
    if segments:
        samples = "\n---\n".join(
            f"片段{i + 1}: {s['text'][:200]}"
            for i, s in enumerate(segments[:3])
        )
        few_shot = (
            "以下是这位老师讲过的真实片段（参考风格，不要照抄）：\n"
            "---\n"
            f"{samples}\n"
            "---\n\n"
        )

    user = PER_SENTENCE_USER.format(
        few_shot=few_shot,
        context=context,
        sentence_text=sentence.get("text", ""),
        keyword_info=keyword_info,
    )
    return system, user


def build_intro_prompt(
    facts: dict,
    style: dict,
    segments: list[dict],
) -> tuple[str, str]:
    """构建开头导入语的 prompt。"""
    system = _build_system(style)
    context = _build_intro_context(facts)

    few_shot = ""
    if segments:
        samples = "\n---\n".join(
            f"片段{i + 1}: {s['text'][:200]}"
            for i, s in enumerate(segments[:3])
        )
        few_shot = (
            "以下是这位老师讲过的真实片段（参考风格，不要照抄）：\n"
            "---\n"
            f"{samples}\n"
            "---\n\n"
        )

    user = PER_INTRO_USER.format(
        few_shot=few_shot,
        context=context,
    )
    return system, user


# ------------------------------------------------------------------- TTS

async def synthesize_audio_async(narration: str, out_path: Path) -> None:
    """edge-tts 合成音频到 mp3。

    注：edge-tts 逆向的微软 Read-Aloud 接口对中文不返回 WordBoundary 事件
    （实测中文文本边界数恒为 0），故无法用 TTS 词边界做对齐，
    关键词时间统一走 align_by_ratio（字符位置线性估时）。
    """
    import edge_tts

    voice = "zh-CN-YunjianNeural"
    comm = edge_tts.Communicate(narration, voice)
    await comm.save(str(out_path))


def synthesize_audio(narration: str, out_path: Path) -> None:
    """edge-tts 合成（同步封装）。"""
    narration = narration.strip()
    if not narration:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(synthesize_audio_async(narration, out_path))


def read_mp3_duration(path: Path) -> float:
    """mutagen 读 MP3 时长（秒）。"""
    from mutagen.mp3 import MP3
    return MP3(path).info.length


# ------------------------------------------------------------------- 标记解析

def parse_markers(narration: str) -> tuple[str, dict[str, int]]:
    """解析 [关键词] 标记，返回 (纯净文本, {词: 字符位置})。

    注：DeepSeek 可能在同一句里多次标记同一词（如开头引用原句时 + 讲解时）。
    这里会解析出首次标记的位置。但最终用于高亮的是按字符占比 × 音频时长的
    线性估时——该位置占 narration 长度的比例，估算音频读到该位置的时刻。
    若 DeepSeek 在原句引用处标记而非讲解处标记，估时仍会偏早，但比启发式
    猜测（取中部出现）更可控。
    """
    import re

    positions = {}
    clean = []
    offset = 0
    for m in re.finditer(r"\[(.+?)\]", narration):
        word = m.group(1)
        if word not in positions:
            positions[word] = m.start() - offset
        clean.append(narration[offset:m.start()])
        clean.append(word)
        offset = m.end()
    clean.append(narration[offset:])
    return "".join(clean), positions


# ------------------------------------------------------------------- 关键词定位（已废弃，保留兼容）
# locate_keywords_in_narration 已移除，改用 DeepSeek 生成时直接标 [关键词]


# ------------------------------------------------------------------- 对齐

def _narration_pos(narration: str, word: str,
                   positions: Optional[dict]) -> Optional[int]:
    """关键词在 narration 原文中的字符位置。

    优先找讲解引导词（"这个X"/"再看X"/"所谓X"），
    这是中文讲解的自然语序，比 [词] 标记更可靠。
    引导词匹配不到再退回用 [词] 标记；都没有就返回 None。
    """
    import re
    # 引导词 + 可选引号 + 关键词
    guides = r"(?:这个|再看|所谓|叫做|注意这个|先看|重点看|就是)"
    quote = r"(?:[\u201c\u201d\u2018\u2019\u300c\u300d\"'']?)"
    pattern = re.compile(f"({guides})\\s*{quote}({re.escape(word)})")
    m = pattern.search(narration)
    if m:
        return m.start(2)  # 关键词组的起始位置
    if positions and word in positions:
        return positions[word]
    return None


def align_by_ratio(narration: str, keywords: list, duration: float,
                   positions: Optional[dict] = None) -> list:
    """按字符位置线性估时定位关键词（默认对齐路径，无需 ASR）。

    edge-tts 中文语速均匀，关键词在讲解文本中的字符位置占比 × 音频时长
    ≈ 读到该词的时刻，误差通常 1~2 秒内，对"讲到词→句中高亮"足够，
    且零 ASR 误差、零依赖、秒级完成。

    定位仍优先用 DeepSeek 打的 [词] 标记(positions)，否则取该词在原文
    中部出现处（_narration_pos）。
    """
    n = len(narration)
    if n == 0 or duration <= 0:
        return []
    timings = []
    for kw in keywords:
        word = kw.get("word", "")
        if not word:
            continue
        npos = _narration_pos(narration, word, positions)
        if npos is None:
            continue
        t = npos / n * duration
        timings.append({
            "word": word,
            "note": kw.get("note", ""),
            "time": round(t, 2),
            "start": round(max(0, t - 0.5), 2),
        })
    return sorted(timings, key=lambda x: x["time"])


# ------------------------------------------------------------------- 课程/章节拆分

def _split_course_chapter(lesson: str) -> tuple[str, str]:
    """从 lesson 名提取课程名和章节名。
    
    规则：
      - "德行篇第25则" → ("世说新语精读", "德行篇")
      - "言语篇" → ("世说新语精读", "言语篇")
      - "木兰诗" → ("木兰诗", "木兰诗")
      - "背影" → ("背影", "背影")
    
    Returns:
        (课程名, 章节名)
    """
    # 世说新语各篇 → 课程=世说新语精读
    shishuo_chapters = {
        "德行篇", "言语篇", "政事篇", "文学篇", "方正篇", "雅量篇",
        "识鉴篇", "赏誉篇", "品藻篇", "规箴篇", "捷悟篇", "夙慧篇",
        "豪爽篇", "容止篇", "自新篇", "俭啬篇", "汰侈篇", "忿狷篇",
        "情礼篇", "黜免篇", "俭吝篇", "惑溺篇", "仇隙篇", "任诞篇",
        "伤逝篇", "栖逸篇", "贤媛篇", "术解篇", "巧艺篇", "知惧篇",
        "企羡篇",
    }
    
    for chapter in shishuo_chapters:
        if chapter in lesson:
            return ("世说新语精读", chapter)
    
    # 其他课文：课程=章节=lesson
    return (lesson, lesson)


# ------------------------------------------------------------------- 保存

def _save_json(out_file: Path, lesson_name: str, facts: dict,
               results: list, intro_narration: str = "") -> None:
    """写入 JSON。

    保留 _positions（[词] 标记解析出的原文位置），
    供重跑对齐时精确去歧义，否则只能退回"取中部出现"的粗略猜测。
    
    Args:
        out_file: 输出文件路径
        lesson_name: 课文名
        facts: 课文事实
        results: 句子讲解列表
        intro_narration: 开头导入语
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)

    obj = {
        "title": f"《{lesson_name}》精讲",
        "author": facts.get("author", ""),
        "dynasty": facts.get("dynasty", ""),
        "source": facts.get("source", ""),
        "intro": intro_narration,
        "sentences": results,
    }
    out_file.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ------------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="按段落分批生成讲解时间轴"
    )
    p.add_argument("query", type=str, help="课文查询，如'诫子书 全文'")
    p.add_argument("--lesson", type=str, default=None,
                   help="手动指定课文名")
    p.add_argument("--max-sentences", type=int, default=0,
                   help="最多处理前 N 句（0=全量），调试用")
    p.add_argument("--batch-size", type=int, default=5,
                   help="每批最大句数（默认 5），控制单次 LLM 请求大小")
    p.add_argument("--skip-tts", action="store_true",
                   help="跳过 TTS 合成")
    p.add_argument("--skip-align", action="store_true",
                   help="跳过对齐（不生成 keyword_timings）")
    p.add_argument("--rerank", action="store_true",
                   help="启用 reranker 精排")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="打印讲解内容摘要")
    return p.parse_args()


def split_sentences_by_paragraph(sentences: list, batch_size: int = 5) -> list:
    """按段落分批句子。

    策略：
    1. 如果句子带有 paragraph 标记，按段落分组
    2. 否则按 batch_size 句一批

    Args:
        sentences: 句子列表，每句可含可选的 "paragraph" 字段
        batch_size: 每批最大句数

    Returns:
        分批后的句子列表，每批是一个句子列表
    """
    # 检查是否有段落标记
    has_paragraph = any(s.get("paragraph") is not None for s in sentences)

    if has_paragraph:
        # 按段落分组
        paragraphs = {}
        for s in sentences:
            para = s.get("paragraph", 0)
            if para not in paragraphs:
                paragraphs[para] = []
            paragraphs[para].append(s)

        # 合并小段落，拆分大段落
        batches = []
        for para_idx in sorted(paragraphs.keys()):
            para_sentences = paragraphs[para_idx]
            if len(para_sentences) <= batch_size:
                batches.append(para_sentences)
            else:
                # 大段落按 batch_size 拆分
                for i in range(0, len(para_sentences), batch_size):
                    batches.append(para_sentences[i:i + batch_size])
        return batches
    else:
        # 没有段落标记，按 batch_size 分批
        batches = []
        for i in range(0, len(sentences), batch_size):
            batches.append(sentences[i:i + batch_size])
        return batches


def main() -> None:
    args = parse_args()

    cfg = generate.load_config()
    if not cfg["api_key"]:
        print("✗ 未找到 DeepSeek api_key。")
        sys.exit(1)

    # 步骤 1: 意图解析（先拿到真实课程名）
    intent = generate.parse_intent(args.query, cfg)
    intent["intent"] = "精读讲解"
    if args.lesson:
        intent["lesson"] = args.lesson
    print(f"意图: {json.dumps(intent, ensure_ascii=False)}")

    # 确定输出目录
    # 从 lesson 提取课程名和章节名
    # lesson 格式可能是：
    #   - "木兰诗" → 课程=章节=木兰诗
    #   - "世说新语精读" → 课程=世说新语精读, 章节=世说新语精读
    #   - "德行篇第25则" → 课程=世说新语精读, 章节=德行篇
    lesson_name = intent.get("lesson", "") or generate._extract_lesson(args.query)
    if args.lesson:
        lesson_name = args.lesson
    
    course_name, chapter_name = _split_course_chapter(lesson_name)
    course_dir = generate.lesson_name_to_pinyin(course_name)
    chapter_dir = generate.lesson_name_to_pinyin(chapter_name)
    lesson_dir = TIMELINE_DIR / course_dir / chapter_dir
    script_file = lesson_dir / "script.json"
    timeline_file = lesson_dir / "timeline.json"

    if script_file.exists():
        # 已有讲稿，跳过 DeepSeek，直接跑 TTS + 对齐
        print(f"讲稿已存在，跳过生成: {script_file}")
        timeline = json.loads(script_file.read_text(encoding="utf-8"))
        results = timeline.get("sentences", [])
        total = len(results)
        facts = {
            "title": timeline.get("title", ""),
            "author": timeline.get("author", ""),
            "source": timeline.get("source", ""),
            "dynasty": timeline.get("dynasty", ""),
        }
        intro_narration = timeline.get("intro", "")
    else:
        intro_narration = ""

        # 步骤 2: 获取课文事实
        print("获取课文事实...")
        facts = generate.fetch_lesson_facts(cfg, args.query, intent)
        if not facts or not facts.get("sentences"):
            print("✗ 未获取到课文 sentences。")
            sys.exit(1)

        sentences = facts["sentences"]
        total = len(sentences)
        if args.max_sentences > 0:
            sentences = sentences[:args.max_sentences]
            total = len(sentences)
        print(
            f"课文: {facts.get('source', '')}  {facts.get('author', '')}"
            f"（{facts.get('dynasty', '')}）  共 {total} 句"
        )

        # 步骤 3: 检索相关语料
        print("检索相关语料...")
        enc = generate.Encoder()
        segments = generate.retrieve(args.query, intent, enc, rerank=args.rerank)

        # 步骤 4: 加载风格
        style = generate.load_style()

        # 步骤 5.0: 生成开头导入语
        print("\n生成开头导入语...")
        intro_system, intro_user = build_intro_prompt(facts, style, segments)
        try:
            intro_narration = generate.call_deepseek(cfg, intro_system, intro_user).strip()
            print(f"  ✓ 导入语: {intro_narration[:80]}...")
        except Exception as e:
            print(f"  ✗ 导入语生成失败: {e}")
            intro_narration = ""

        # 步骤 5.1: 按段落分批生成 narration
        batch_size = args.batch_size
        batches = split_sentences_by_paragraph(sentences, batch_size)
        num_batches = len(batches)
        print(f"\n分 {num_batches} 批生成讲解（每批最多 {batch_size} 句）：")

        results = []
        prev_narration = intro_narration  # 用导入语作为第一句的上下文

        for batch_idx, batch_sentences in enumerate(batches):
            batch_start = sum(len(b) for b in batches[:batch_idx])
            batch_sentences_in_results = [
                sentences[batch_start + i] for i in range(len(batch_sentences))
            ]

            print(f"\n--- 批次 {batch_idx + 1}/{num_batches}（{len(batch_sentences)} 句）---")

            for i, s in enumerate(batch_sentences_in_results):
                idx = batch_start + i + 1
                text_preview = s.get("text", "")[:30]
                print(f"  [{idx}/{total}] {text_preview}...", end=" ", flush=True)

                t0 = time.time()
                system, user = build_per_sentence_prompt(
                    s, idx, total, facts, style, segments, prev_narration,
                )
                try:
                    raw = generate.call_deepseek(cfg, system, user).strip()
                except Exception as e:
                    print(f"✗ {e}")
                    raw = ""

                # clean narration（去除 [关键词] 标记——暂不用于对齐，仅保留纯净文本）
                narration, _ = parse_markers(raw)

                dt = time.time() - t0
                print(f"({dt:.1f}s)")

                if args.verbose and narration:
                    print(f"    {narration[:150]}...")

                entry = {
                    "id": idx,
                    "text": s.get("text", ""),
                    "translation": s.get("translation", ""),
                    "keywords": s.get("keywords", []),
                    "narration": narration,
                }
                results.append(entry)
                prev_narration = narration  # 更新为当前句的 narration

    # 先生成别忘了存——万一 TTS 崩了稿子还在
    if not script_file.exists():
        _save_json(script_file, lesson_name, facts, results, intro_narration)
        print(f"✓ 讲稿已存: {script_file}")

    # 步骤 6: TTS + 对齐
    if not args.skip_tts:
        audio_dir = lesson_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # 先合成导入语音频（id=0）
        if intro_narration:
            print("\nTTS 合成导入语...")
            intro_mp3 = audio_dir / "00_intro.mp3"
            if not intro_mp3.exists():
                synthesize_audio(intro_narration, intro_mp3)
                intro_dur = read_mp3_duration(intro_mp3)
                print(f"  [0] 导入语 ({intro_dur:.1f}s)")
            else:
                intro_dur = read_mp3_duration(intro_mp3)
                print(f"  [0] 导入语已存在 ({intro_dur:.1f}s)")

        print("\nTTS 合成讲解...")
        for entry in results:
            idx = entry["id"]
            narration = entry.get("narration", "")
            if not narration:
                print(f"  [{idx}/{total}] 无讲解，跳过")
                continue

            mp3_path = audio_dir / f"{idx:02d}.mp3"
            entry["audio"] = str(mp3_path.relative_to(lesson_dir))

            if mp3_path.exists():
                dur = read_mp3_duration(mp3_path)
                entry["duration"] = round(dur, 1)
                print(f"  [{idx}/{total}] 已存在，跳过 ({dur:.1f}s)")
                continue

            print(f"  [{idx}/{total}] 合成...", end=" ", flush=True)
            synthesize_audio(narration, mp3_path)
            dur = read_mp3_duration(mp3_path)
            entry["duration"] = round(dur, 1)
            print(f"{dur:.1f}s")

        # 关键词对齐（TODO: 待攻坚）
        for entry in results:
            entry["keyword_timings"] = []

    # 步骤 7: 输出 timeline.json
    _save_json(timeline_file, lesson_name, facts, results, intro_narration)
    print(f"✓ timeline: {timeline_file}")

    ok = sum(1 for r in results if r["narration"])
    chars = sum(len(r["narration"]) for r in results)
    dur = sum(r.get("duration", 0) for r in results)
    print(f"  有效讲解: {ok}/{len(results)} 句  {chars} 字  {dur:.0f}s")


if __name__ == "__main__":
    main()
