#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按句生成讲解时间轴（设计文档 §11 核心管线）。

输入：课文查询（如"诫子书 全文"）
输出：data/timelines/{课文名}/timeline.json

流程：
  1. 复用 generate.py 的意图解析 + 课文事实获取 → sentences / keywords
  2. 对每句调 DeepSeek 生成 narration（带上下文和上一句结尾）
  3. CosyVoice 2.0 合成讲解音频
  4. WhisperX 强制对齐 → keyword_timings
  5. 输出完整 timeline JSON

用法：
  python3 script/build_timeline.py "诫子书 全文"
  python3 script/build_timeline.py "诫子书 全文" --max-sentences 3 --verbose
  python3 script/build_timeline.py "诫子书" --skip-tts        # 只生成讲解文本，不合成音频
  python3 script/build_timeline.py "诫子书" --skip-align      # 跳过 WhisperX 对齐
  python3 script/build_timeline.py "诫子书" --skip-tts --skip-align  # 仅生成 narration
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# CosyVoice 源码路径（需先 git clone --recursive）
# 默认 ~/CosyVoice，可通过 COSYVOICE_HOME 覆盖
_COSYVOICE_HOME = Path(
    os.environ.get("COSYVOICE_HOME", Path.home() / "CosyVoice")
)
if _COSYVOICE_HOME.is_dir():
    sys.path.insert(0, str(_COSYVOICE_HOME))

import generate  # 复用意图解析、课文事实、检索、风格加载

DATA_DIR = PROJECT_ROOT / "data"
TIMELINE_DIR = DATA_DIR / "timelines"

# CosyVoice 本地模型目录
COSYVOICE_MODEL = PROJECT_ROOT / "cosyvoice" / "models" / "CosyVoice2-0.5B"

# ------------------------------------------------------------------- 逐句生成

PER_SENTENCE_USER = """{few_shot}
{context}

现在请你讲解以下这句：

原句：{sentence_text}
{keyword_info}

要求：
1. 用第一人称"我"的课堂口吻。
2. 如本句有重点词，逐个解读：读音、本义、语境义、用法。释义以背景信息为准。
3. 然后把这句话串讲一遍，点出它好在哪、要体会什么。
4. 上一句结尾为参考，开头的过渡语自然衔接（如"接着看下一句""好，我们再来看"），
   但不要重复复述上一句内容。
5. 不限字数，该展开就展开。

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


# ------------------------------------------------------------------- TTS

def synthesize_audio(narration: str, out_path: Path) -> float:
    """CosyVoice 2.0 合成音频，返回时长（秒）。"""
    narration = narration.strip()
    if not narration:
        return 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2
    except ModuleNotFoundError:
        print(
            "\n✗ CosyVoice 未正确安装。请按以下步骤操作：\n"
            "  cd ~\n"
            "  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git\n"
            "  cd CosyVoice\n"
            "  pip install -r requirements.txt\n"
            "\n然后设置环境变量:\n"
            "  export COSYVOICE_HOME=~/CosyVoice\n"
        )
        sys.exit(1)

    import numpy as np
    import soundfile as sf

    model = CosyVoice2(str(COSYVOICE_MODEL))
    output = model.inference(narration, stream=False)
    chunks = [seg["tts_speech"] for seg in output]
    audio = np.concatenate(chunks) if chunks else np.array([])

    if len(audio) == 0:
        return 0.0

    sf.write(str(out_path), audio, 24000)
    return round(len(audio) / 24000, 1)


def read_mp3_duration(path: Path) -> float:
    """mutagen 读 MP3 时长（秒）。"""
    from mutagen.mp3 import MP3
    return MP3(path).info.length


# ------------------------------------------------------------------- 强制对齐

def align_keywords(audio_path: Path, narration: str, keywords: list,
                   device: str = "cpu") -> list[dict]:
    """WhisperX 强制对齐 → 关键词时间戳。"""
    import whisperx

    model_a, meta = whisperx.load_align_model(language_code="zh", device=device)
    audio = whisperx.load_audio(str(audio_path))
    segments = [{"text": narration, "start": 0.0, "end": None}]
    result = whisperx.align(segments, model_a, meta, audio, device,
                            return_char_alignments=True)

    char_times = []
    for seg in result["segments"]:
        for ch in seg.get("chars", []):
            if ch.get("start") is not None:
                char_times.append((ch["char"], ch["start"]))

    full = "".join(c for c, _ in char_times)
    timings = []
    for kw in keywords:
        word = kw.get("word", "")
        if not word:
            continue
        idx = full.find(word)
        if idx < 0:
            continue
        t = char_times[idx][1]
        timings.append({
            "word": word,
            "note": kw.get("note", ""),
            "time": round(t, 2),
            "start": round(max(0, t - 0.5), 2),
        })
    return sorted(timings, key=lambda x: x["time"])


# ------------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="按句生成讲解时间轴（CosyVoice + WhisperX）"
    )
    p.add_argument("query", type=str, help="课文查询，如'诫子书 全文'")
    p.add_argument("--lesson", type=str, default=None,
                   help="手动指定课文名")
    p.add_argument("--max-sentences", type=int, default=0,
                   help="最多处理前 N 句（0=全量），调试用")
    p.add_argument("--skip-tts", action="store_true",
                   help="跳过 TTS 合成")
    p.add_argument("--skip-align", action="store_true",
                   help="跳过 WhisperX 对齐（不生成 keyword_timings）")
    p.add_argument("--rerank", action="store_true",
                   help="启用 reranker 精排")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="打印讲解内容摘要")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = generate.load_config()
    if not cfg["api_key"]:
        print("✗ 未找到 DeepSeek api_key。")
        sys.exit(1)

    # 确定输出目录
    lesson_name = generate._extract_lesson(args.query)
    if args.lesson:
        lesson_name = args.lesson
    lesson_dir = TIMELINE_DIR / lesson_name.replace("/", "_")
    out_file = lesson_dir / "timeline.json"

    if out_file.exists():
        # JSON 已存在，跳过 DeepSeek，直接跑 TTS + 对齐
        print(f"JSON 已存在，跳过生成: {out_file}")
        timeline = json.loads(out_file.read_text(encoding="utf-8"))
        results = timeline.get("sentences", [])
        total = len(results)
        facts = {
            "title": timeline.get("title", ""),
            "author": timeline.get("author", ""),
            "source": timeline.get("source", ""),
            "dynasty": timeline.get("dynasty", ""),
        }
    else:

        # 步骤 1: 意图解析
        intent = generate.parse_intent(args.query, cfg)
        intent["intent"] = "精读讲解"
        if args.lesson:
            intent["lesson"] = args.lesson
        print(f"意图: {json.dumps(intent, ensure_ascii=False)}")

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

        # 步骤 5: 逐句生成 narration
        print(f"\n逐句生成讲解 ({cfg['model']})：")
        results = []
        prev_narration = ""

        for i, s in enumerate(sentences):
            idx = i + 1
            text_preview = s.get("text", "")[:30]
            print(f"  [{idx}/{total}] {text_preview}...", end=" ", flush=True)

            t0 = time.time()
            system, user = build_per_sentence_prompt(
                s, idx, total, facts, style, segments, prev_narration,
            )
            try:
                narration = generate.call_deepseek(cfg, system, user).strip()
            except Exception as e:
                print(f"✗ {e}")
                narration = ""

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
            prev_narration = narration

    # 步骤 6: TTS + 对齐
    if not args.skip_tts:
        audio_dir = lesson_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        print("\nTTS 合成（CosyVoice 2.0）...")
        for entry in results:
            idx = entry["id"]
            narration = entry.get("narration", "")
            if not narration:
                print(f"  [{idx}/{total}] 无讲解，跳过")
                continue

            mp3_path = audio_dir / f"{idx:02d}.mp3"
            print(f"  [{idx}/{total}] 合成...", end=" ", flush=True)

            dur = synthesize_audio(narration, mp3_path)
            entry["audio"] = str(mp3_path.relative_to(lesson_dir))
            entry["duration"] = dur
            print(f"{dur:.1f}s")

        # 对齐
        if not args.skip_align:
            print("\n强制对齐（WhisperX）...")
            for entry in results:
                idx = entry["id"]
                narration = entry.get("narration", "")
                keywords = entry.get("keywords", [])
                if not narration or not keywords:
                    entry["keyword_timings"] = []
                    continue

                mp3_path = lesson_dir / entry["audio"]
                print(f"  [{idx}/{total}] 对齐...", end=" ", flush=True)
                timings = align_keywords(mp3_path, narration, keywords)
                entry["keyword_timings"] = timings
                print(f"{len(timings)} 个关键词")
        else:
            for entry in results:
                entry["keyword_timings"] = []

    # 步骤 7: 输出 JSON
    lesson_dir.mkdir(parents=True, exist_ok=True)

    timeline = {
        "title": f"《{lesson_name}》精讲",
        "author": facts.get("author", ""),
        "dynasty": facts.get("dynasty", ""),
        "source": facts.get("source", ""),
        "sentences": results,
    }

    out_file.write_text(
        json.dumps(timeline, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n✓ 已输出 {out_file}")

    ok = sum(1 for r in results if r["narration"])
    chars = sum(len(r["narration"]) for r in results)
    dur = sum(r.get("duration", 0) for r in results)
    print(f"  有效讲解: {ok}/{len(results)} 句  {chars} 字  {dur:.0f}s")


if __name__ == "__main__":
    main()
