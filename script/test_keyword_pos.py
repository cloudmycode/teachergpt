#!/usr/bin/env python3
"""独立测试/调试 关键词定位 + whisper 对齐。

两套对齐方案对比：
  A) 线性估时——按字符占比 × 音频时长
  B) whisper 对齐——whisper-small 逐字时间戳 + difflib 对齐 narration → 时间戳

用法：
  python3 script/test_keyword_pos.py            # 仅线性估时
  python3 script/test_keyword_pos.py --whisper  # 加 whisper 对齐对比
"""

import re
import json
import sys
from pathlib import Path


# ---- 从 build_timeline.py 提取 ----


def parse_markers(narration: str) -> tuple[str, dict[str, int]]:
    """解析 [关键词] 标记，返回 (纯净文本, {词: 字符位置})。"""
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


def _narration_pos(narration: str, word: str,
                   positions: dict | None = None) -> int | None:
    """关键词在 narration 原文中的字符位置。

    1) 引导词匹配——找"这个X"/"再看X"/"所谓X"等讲解语序
    2) [词] 标记(positions)
    3) 都没有 → None
    """
    guides = r"(?:这个|再看|所谓|叫做|注意这个|先看|重点看|就是)"
    quote = r"(?:[\u201c\u201d\u2018\u2019\u300c\u300d\"'']?)"
    pattern = re.compile(f"({guides})\\s*{quote}({re.escape(word)})")
    m = pattern.search(narration)
    if m:
        return m.start(2), "引导词"
    if positions and word in positions:
        return positions[word], "[词]标记"
    return None, "未找到"


def align_by_ratio(narration: str, keywords: list, duration: float,
                   positions: dict | None = None) -> list[dict]:
    """按字符位置线性估时。"""
    n = len(narration)
    if n == 0 or duration <= 0:
        return []
    timings = []
    for kw in keywords:
        word = kw.get("word", "")
        if not word:
            continue
        npos, _ = _narration_pos(narration, word, positions)
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


# ---- whisper 对齐（方案 B）----

def _align_text(src: str, dst: str) -> dict[int, int]:
    """difflib 字符级对齐 {narration下标: whisper下标}。

    相等块逐字符映射；差异块（同音错字/增删）按区间线性插值兜底。
    """
    import difflib
    sm = difflib.SequenceMatcher(None, src, dst, autojunk=False)
    mapping: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        span_s = i2 - i1
        span_d = j2 - j1
        if tag == "equal":
            for k in range(span_s):
                mapping[i1 + k] = j1 + k
        else:
            for k in range(span_s):
                if span_d == 0:
                    mapping[i1 + k] = max(0, j1 - 1)
                else:
                    mapping[i1 + k] = j1 + min(span_d - 1, k * span_d // span_s)
    return mapping


def align_keywords_whisper(
    audio_path: Path,
    narration: str,
    keywords: list,
    positions: dict | None = None,
    model_dir: str | None = None,
) -> tuple[list[dict], list[dict] | None]:
    """faster-whisper-small → 段落级对齐 → 关键词时间。

    不用全局 difflib（同音字错误会逐字传播），改为：
      1. narration 按 \\n 段落切分
      2. 每段在 whisper segments 中找最匹配的段（SequenceMatcher ratio）
      3. 关键词落在哪个段落 → 用该段 start/end + 段内字符占比估时
    同音字错误只影响所在段落，不跨段传播。
    """
    import difflib as _difflib
    from faster_whisper import WhisperModel

    SCRIPT_DIR = Path(__file__).resolve().parent
    if model_dir is None:
        model_path = SCRIPT_DIR.parent / "asr" / "models" / "faster-whisper-small"
    else:
        model_path = Path(model_dir)
    if not (model_path / "model.bin").exists():
        print("  ✗ 本地模型不存在: " + str(model_path))
        return [], None

    print(f"  加载 whisper 模型: {model_path}")
    model = WhisperModel(str(model_path), device="cpu", compute_type="int8")
    segments_iter, _ = model.transcribe(
        str(audio_path), language="zh", word_timestamps=True,
    )

    raw_segments = []
    for seg in segments_iter:
        raw_segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "words": [{"word": w.word.strip(), "start": round(w.start, 3),
                        "end": round(w.end, 3)} for w in seg.words],
        })

    if not raw_segments:
        _save_whisper_debug(raw_segments, [], [], narration, [])
        return [], raw_segments

    # ---- 段落切分 & 偏移量 ----
    paras = [p.strip() for p in narration.split("\n") if p.strip()]
    if not paras:
        paras = [narration]
    para_offsets = []
    off = 0
    for p in paras:
        idx = narration.index(p, off)
        para_offsets.append(idx)
        off = idx + len(p)

    # ---- 段落 → whisper segment 匹配 ----
    para_to_seg = {}
    seg_used = set()
    for pi, para in enumerate(paras):
        if len(para) < 6:
            continue
        best_ratio, best_si = 0, -1
        for si, seg in enumerate(raw_segments):
            if si in seg_used:
                continue
            r = _difflib.SequenceMatcher(None, para, seg["text"]).ratio()
            if r > best_ratio:
                best_ratio, best_si = r, si
        if best_ratio > 0.35 and best_si >= 0:
            para_to_seg[pi] = best_si
            seg_used.add(best_si)

    # ---- 关键词 → 时间 ----
    timings = []
    for kw in keywords:
        word = kw.get("word", "")
        if not word:
            continue
        npos, _ = _narration_pos(narration, word, positions)
        if npos is None:
            continue

        para_idx = -1
        for pi, p_start in enumerate(para_offsets):
            if p_start <= npos < p_start + len(paras[pi]):
                para_idx = pi
                break
        if para_idx < 0 or para_idx not in para_to_seg:
            continue

        seg = raw_segments[para_to_seg[para_idx]]
        p_start = para_offsets[para_idx]
        rel = (npos - p_start) / max(len(paras[para_idx]), 1)
        t = seg["start"] + rel * (seg["end"] - seg["start"])
        timings.append({
            "word": word,
            "note": kw.get("note", ""),
            "time": round(t, 2),
            "start": round(max(0, t - 0.5), 2),
        })

    _save_whisper_debug(raw_segments, [], [], narration, timings)
    return sorted(timings, key=lambda x: x["time"]), raw_segments


def _save_whisper_debug(segments, whisper_chars, char_times, narration, timings):
    """把 whisper 原始输出 + 对齐信息写入调试文件。"""
    out_dir = Path(__file__).resolve().parent / "_debug_whisper"
    out_dir.mkdir(exist_ok=True)

    # 1) 原始 segment 数据
    (out_dir / "segments.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) whisper 完整转写文本
    whisper_text = "".join(whisper_chars)
    (out_dir / "whisper_text.txt").write_text(whisper_text, encoding="utf-8")

    # 3) narration 原文（对照用）
    (out_dir / "narration.txt").write_text(narration, encoding="utf-8")

    # 4) 字级时间戳摘要（前 20 和后 20 字符）
    timeline_lines = []
    for i, (ch, t) in enumerate(zip(whisper_chars, char_times)):
        timeline_lines.append(f"{i:4d}  {t:7.3f}s  {ch}")
    (out_dir / "char_timeline.txt").write_text("\n".join(timeline_lines), encoding="utf-8")

    # 5) 关键词对齐结果
    (out_dir / "keyword_timings.json").write_text(
        json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  → 调试文件: {out_dir}/")


# ---- 调试辅助 ----


def show_context(narration: str, pos: int, word: str, ctx: int = 30) -> str:
    """打印关键词周围的上下文片段。"""
    start = max(0, pos - ctx)
    end = min(len(narration), pos + len(word) + ctx)
    before = narration[start:pos]
    after = narration[pos + len(word):end]
    marker = "^" * len(word)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(narration) else ""
    return f"    {prefix}{before}「{word}」{after}{suffix}\n    {' ' * (len(prefix) + len(before))}{marker}"


# ---- 测试 ----


# 注意：以下 narration 对关键词有 [词] 标记（讲解处标记），也存在引导词变体
NAR_RAW = '''好，我们接着来看这句。这句话是全文的开篇，也是诸葛亮给儿子立下的一个总纲，所以非常重要。大家看这句——\u201c夫君子之行，静以修身，俭以养德。\u201d 这个\u201c夫\u201d字，我们读作 夫，它是一个句首发语词，没有实际的意义，就是用来提起话题的，相当于\u201c啊\u201d、\u201c要说这个\u201d的意思，对不对？所以一上来，诸葛亮就说：\u201c啊，一个君子的行为……\u201d 那么，什么是\u201c君子\u201d呢？这个大家应该不陌生，就是品德高尚、有修养的人。诸葛亮希望儿子诸葛瞻成为一个君子，所以一开头就点明了这个目标。

接着看这个\u201c静\u201d字，[静]字可太关键了。它的本义就是安静、平静，但在语境里，它指的是屏除杂念和干扰，达到一种宁静专一的状态。大家想想，一个人心浮气躁的时候，能读得进书吗？能想明白道理吗？不能见得着吧？所以这个\u201c静\u201d，就是修身养性的一个前提。诸葛亮说\u201c静以修身\u201d，这个\u201c以\u201d字，它是一个连词，表示目的，可以翻译成\u201c来\u201d，所以就是\u201c用宁静专一来修养身心\u201d。这个 修身 啊，就是修养自己的身心，让自己的言行举止都合乎规范。

然后下半句，\u201c俭以养德\u201d。这个\u201c俭\u201d字，读作 俭，它的本义就是节俭、不浪费。在语境里，它不仅仅指生活上的朴素，更是一种对欲望的克制。一个人如果奢侈浪费，他的心思就会被物质享受所牵绊，哪还有精力去培养品德呢？所以诸葛亮说，要用\u201c俭\u201d来\u201c养德\u201d。这个 养德 就是培养品德的意思。你看，一个\u201c静\u201d，一个\u201c俭\u201d，一内一外，一个是内心的专注，一个是外在的克制，这两者结合起来，才能成就一个君子的德行。

所以，我们把这句话串讲一下：一个君子的行为，是用宁静专一来修养身心，用节俭来培养品德。这句话好在哪呢？好在它开门见山，直接点出了修身养德的两个核心要素。我们要体会的是，诸葛亮作为一个父亲，他对儿子的期望是多么殷切，他把自己一生最宝贵的经验——就是\u201c静\u201d和\u201c俭\u201d——毫无保留地传授给了儿子。这就像我们常说的，一个人要想有所成就，首先得沉得下心来，还得管得住自己，对不对？这个道理，放到今天，对我们每个人来说，依然是非常有启发的。'''

NAR_CLEAN = '''好，我们接着来看这句。这句话是全文的开篇，也是诸葛亮给儿子立下的一个总纲，所以非常重要。大家看这句——\u201c夫君子之行，静以修身，俭以养德。\u201d 这个\u201c夫\u201d字，我们读作 夫，它是一个句首发语词，没有实际的意义，就是用来提起话题的，相当于\u201c啊\u201d、\u201c要说这个\u201d的意思，对不对？所以一上来，诸葛亮就说：\u201c啊，一个君子的行为……\u201d 那么，什么是\u201c君子\u201d呢？这个大家应该不陌生，就是品德高尚、有修养的人。诸葛亮希望儿子诸葛瞻成为一个君子，所以一开头就点明了这个目标。

接着看这个\u201c静\u201d字，静字可太关键了。它的本义就是安静、平静，但在语境里，它指的是屏除杂念和干扰，达到一种宁静专一的状态。大家想想，一个人心浮气躁的时候，能读得进书吗？能想明白道理吗？不能见得着吧？所以这个\u201c静\u201d，就是修身养性的一个前提。诸葛亮说\u201c静以修身\u201d，这个\u201c以\u201d字，它是一个连词，表示目的，可以翻译成\u201c来\u201d，所以就是\u201c用宁静专一来修养身心\u201d。这个 修身 啊，就是修养自己的身心，让自己的言行举止都合乎规范。

然后下半句，\u201c俭以养德\u201d。这个\u201c俭\u201d字，读作 俭，它的本义就是节俭、不浪费。在语境里，它不仅仅指生活上的朴素，更是一种对欲望的克制。一个人如果奢侈浪费，他的心思就会被物质享受所牵绊，哪还有精力去培养品德呢？所以诸葛亮说，要用\u201c俭\u201d来\u201c养德\u201d。这个 养德 就是培养品德的意思。你看，一个\u201c静\u201d，一个\u201c俭\u201d，一内一外，一个是内心的专注，一个是外在的克制，这两者结合起来，才能成就一个君子的德行。

所以，我们把这句话串讲一下：一个君子的行为，是用宁静专一来修养身心，用节俭来培养品德。这句话好在哪呢？好在它开门见山，直接点出了修身养德的两个核心要素。我们要体会的是，诸葛亮作为一个父亲，他对儿子的期望是多么殷切，他把自己一生最宝贵的经验——就是\u201c静\u201d和\u201c俭\u201d——毫无保留地传授给了儿子。这就像我们常说的，一个人要想有所成就，首先得沉得下心来，还得管得住自己，对不对？这个道理，放到今天，对我们每个人来说，依然是非常有启发的。'''

# 测试用：带 [词] 标记的（加了 [静] 标记在讲解处）
NAR_WITH_MARKERS = NAR_RAW

# 测试用：不带标记的（模拟 DeepSeek 漏标）
NAR_NO_MARKERS = NAR_CLEAN

KEYWORDS = [
    {"word": "夫", "note": "句首发语词，无实义"},
    {"word": "静", "note": "屏除杂念和干扰，宁静专一"},
    {"word": "修身", "note": "修养身心"},
    {"word": "俭", "note": "节俭"},
    {"word": "养德", "note": "培养品德"},
]

DURATION = 180.0  # 假设音频时长


MP3_PATH = Path(__file__).resolve().parent.parent / "data" / "timelines" / "诫子书" / "audio" / "01.mp3"


def main():
    use_whisper = "--whisper" in sys.argv

    # 实际时长（不用估算值，走真实 mp3 读取）
    from mutagen.mp3 import MP3
    if MP3_PATH.exists():
        real_dur = round(MP3(MP3_PATH).info.length, 1)
        print(f"音频: {MP3_PATH}  时长: {real_dur}s\n")
    else:
        real_dur = DURATION
        print(f"音频不存在，假设时长: {real_dur}s\n")

    for label, nar in [("有 [词] 标记（NAR_RAW）", NAR_WITH_MARKERS),
                       ("无 [词] 标记（NAR_CLEAN）", NAR_NO_MARKERS)]:
        print("=" * 60)
        print(f"场景：{label}")
        print("-" * 60)

        clean, positions = parse_markers(nar)

        # 逐个定位关键词
        keyword_positions = {}
        for kw in KEYWORDS:
            word = kw["word"]
            pos, method = _narration_pos(nar, word, positions)
            keyword_positions[word] = (pos, method)

        # A) 线性估时
        ratio_timings = align_by_ratio(nar, KEYWORDS, real_dur, positions)
        print("\n  [方案A] 线性估时:")
        for t in ratio_timings:
            print(f"    {t['word']:5s} {t['time']:6.1f}s  ({t['note']})")
        print(f"  命中: {len(ratio_timings)}/{len(KEYWORDS)}")

        # B) whisper 对齐
        if use_whisper:
            print("\n  [方案B] whisper-small 逐字对齐 ...")
            w_timings, _ = align_keywords_whisper(
                MP3_PATH, nar, KEYWORDS, positions,
            )
            for t in w_timings:
                print(f"    {t['word']:5s} {t['time']:6.1f}s  ({t['note']})")
            print(f"  命中: {len(w_timings)}/{len(KEYWORDS)}")

            # 对比
            if ratio_timings and w_timings:
                ratio_map = {t["word"]: t["time"] for t in ratio_timings}
                whisper_map = {t["word"]: t["time"] for t in w_timings}
                print("\n  方案A vs 方案B 时间差:")
                for kw in KEYWORDS:
                    w = kw["word"]
                    rt = ratio_map.get(w)
                    wt = whisper_map.get(w)
                    if rt is not None and wt is not None:
                        diff = abs(rt - wt)
                        marker = " ✓" if diff < 3 else " ✗" if diff > 5 else ""
                        print(f"    {w:5s}  A={rt:5.1f}s  B={wt:5.1f}s  Δ={diff:.1f}s{marker}")
                    elif rt is not None:
                        print(f"    {w:5s}  A={rt:5.1f}s  B=未命中")
                    elif wt is not None:
                        print(f"    {w:5s}  A=未命中  B={wt:5.1f}s")
        print()


if __name__ == "__main__":
    main()
