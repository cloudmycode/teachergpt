#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 timeline.json 生成 HTML 播放器。

用法:
  python3 script/build_player.py "诫子书 全文"
  python3 script/build_player.py "诫子书" --out player.html
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import generate

TIMELINE_DIR = PROJECT_ROOT / "data" / "timelines"

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: "PingFang SC", "Noto Sans SC", sans-serif; background: #f5f0e8; color: #3a2f1f; line-height: 1.8; min-height: 100vh; display: flex; flex-direction: column; }}
.header {{ text-align: center; padding: 32px 20px 12px; }}
.header h1 {{ font-size: 26px; font-weight: 700; margin-bottom: 4px; }}
.header .meta {{ font-size: 13px; color: #9c8b75; }}
.page {{ flex: 1; display: flex; flex-direction: column; align-items: center; padding: 16px 20px 40px; }}
.card {{ background: #fff; border-radius: 14px; padding: 28px 24px; max-width: 680px; width: 100%; box-shadow: 0 2px 12px rgba(0,0,0,.06); position: relative; }}
.sentence-num {{ font-size: 12px; color: #c4a97d; letter-spacing: 2px; margin-bottom: 8px; }}
.original-text {{ font-size: 24px; font-weight: 600; letter-spacing: 2px; line-height: 1.6; padding: 16px 0; border-bottom: 1px solid #e8e0d0; margin-bottom: 20px; }}
.original-text .translation {{ font-size: 14px; color: #9c8b75; font-weight: 400; margin-top: 8px; letter-spacing: 1px; }}
.keywords-title {{ font-size: 13px; color: #b5a48c; margin-bottom: 10px; }}
.keywords {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }}
.keyword-tag {{ background: #faf6ed; border: 1px solid #e0d6c2; border-radius: 8px; padding: 6px 14px; font-size: 14px; }}
.keyword-tag .kw {{ font-weight: 700; color: #5c3d2e; margin-right: 6px; }}
.keyword-tag .note {{ font-size: 13px; color: #8c7a6b; }}
.player-bar {{ display: flex; align-items: center; gap: 12px; padding: 12px 0; }}
.player-bar button {{ width: 40px; height: 40px; border: 1px solid #c4a97d; border-radius: 50%; background: #fff; color: #5c3d2e; font-size: 16px; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all .2s; flex-shrink: 0; }}
.player-bar button:hover {{ background: #5c3d2e; color: #fff; }}
.player-bar .btn-play {{ width: 48px; height: 48px; font-size: 20px; }}
.player-bar .time {{ font-size: 12px; color: #b5a48c; min-width: 45px; text-align: center; }}
.player-bar .progress-track {{ flex: 1; height: 4px; background: #e8e0d0; border-radius: 2px; cursor: pointer; position: relative; }}
.player-bar .progress-fill {{ height: 100%; background: #c4a97d; border-radius: 2px; transition: width .3s linear; width: 0%; }}
.nav {{ display: flex; justify-content: center; gap: 12px; margin-top: 20px; }}
.nav button {{ padding: 10px 28px; border: 1px solid #c4a97d; border-radius: 8px; background: #fff; color: #5c3d2e; font-size: 14px; cursor: pointer; transition: all .2s; }}
.nav button:hover {{ background: #5c3d2e; color: #fff; border-color: #5c3d2e; }}
.nav button:disabled {{ opacity: .35; cursor: default; pointer-events: none; }}
.footer {{ text-align: center; padding: 16px; font-size: 12px; color: #c4a97d; }}
</style>
</head>
<body>

<div class="header">
  <h1>{title}</h1>
  <p class="meta">{author} · {dynasty}</p>
</div>

<div class="page">
  <div class="card" id="card">
    <div class="sentence-num" id="sentenceNum"></div>
    <div class="original-text">
      <div id="originalText"></div>
      <div class="translation" id="translationText"></div>
    </div>
    <div class="keywords-title">重点词</div>
    <div class="keywords" id="keywordsList"></div>
    <div class="player-bar">
      <button onclick="prevSentence()" title="上一句">◀</button>
      <button class="btn-play" id="btnPlay" onclick="togglePlay()">▶</button>
      <button onclick="nextSentence()" title="下一句">▶</button>
      <span class="time" id="timeDisplay">--:--</span>
      <div class="progress-track" id="progressTrack" onclick="seek(event)">
        <div class="progress-fill" id="progressFill"></div>
      </div>
    </div>
  </div>
  <div class="nav">
    <button id="btnPrev" onclick="prevSentence()">◀ 上一句</button>
    <button id="btnNext" onclick="nextSentence()">下一句 ▶</button>
  </div>
</div>

<div class="footer">
  <span id="progressLabel"></span>
</div>

<audio id="audioPlayer" preload="none"></audio>

<script>
const DATA = {data_json};
let idx = 0;
let playing = false;

function init() {{
  renderSentence(0);
}}

function renderSentence(i) {{
  if (i < 0 || i >= DATA.sentences.length) return;
  idx = i;
  const s = DATA.sentences[i];
  document.getElementById("sentenceNum").textContent =
    `第 ${{i + 1}} 句 / 共 ${{DATA.sentences.length}} 句`;
  document.getElementById("originalText").textContent = s.text;
  document.getElementById("translationText").textContent = s.translation || "";
  document.getElementById("keywordsList").innerHTML = (s.keywords || []).map(k =>
    `<span class="keyword-tag"><span class="kw">${{k.word}}</span><span class="note">${{k.note}}</span></span>`
  ).join("");
  document.getElementById("btnPrev").disabled = i === 0;
  document.getElementById("btnNext").disabled = i === DATA.sentences.length - 1;
  document.getElementById("progressLabel").textContent =
    `${{i + 1}} / ${{DATA.sentences.length}}`;

  loadAudio(i);
}}

function loadAudio(i) {{
  const s = DATA.sentences[i];
  const a = document.getElementById("audioPlayer");
  a.src = s.audio;
  a.load();
  playing = false;
  document.getElementById("btnPlay").textContent = "▶";
  document.getElementById("progressFill").style.width = "0%";
  document.getElementById("timeDisplay").textContent = "--:--";
}}

function togglePlay() {{
  const a = document.getElementById("audioPlayer");
  if (playing) {{
    a.pause();
  }} else {{
    a.play();
  }}
}}

document.getElementById("audioPlayer").addEventListener("play", () => {{
  playing = true;
  document.getElementById("btnPlay").textContent = "⏸";
}});

document.getElementById("audioPlayer").addEventListener("pause", () => {{
  playing = false;
  document.getElementById("btnPlay").textContent = "▶";
}});

document.getElementById("audioPlayer").addEventListener("timeupdate", () => {{
  const a = document.getElementById("audioPlayer");
  const fmt = t => {{
    const m = Math.floor(t / 60), s = Math.floor(t % 60);
    return `${{m}}:${{String(s).padStart(2, "0")}}`;
  }};
  document.getElementById("timeDisplay").textContent =
    `${{fmt(a.currentTime)}}`;
  if (a.duration) {{
    const pct = a.currentTime / a.duration * 100;
    document.getElementById("progressFill").style.width = pct + "%";
  }}
}});

document.getElementById("audioPlayer").addEventListener("ended", () => {{
  playing = false;
  document.getElementById("btnPlay").textContent = "▶";
  document.getElementById("progressFill").style.width = "100%";
  if (idx < DATA.sentences.length - 1) {{
    setTimeout(() => nextSentence(), 600);
  }}
}});

function seek(e) {{
  const a = document.getElementById("audioPlayer");
  if (!a.duration) return;
  const rect = document.getElementById("progressTrack").getBoundingClientRect();
  const pct = (e.clientX - rect.left) / rect.width;
  a.currentTime = pct * a.duration;
}}

function nextSentence() {{
  if (idx < DATA.sentences.length - 1) {{
    const a = document.getElementById("audioPlayer");
    a.pause();
    renderSentence(idx + 1);
    setTimeout(() => a.play(), 100);
  }}
}}

function prevSentence() {{
  if (idx > 0) {{
    const a = document.getElementById("audioPlayer");
    a.pause();
    renderSentence(idx - 1);
    setTimeout(() => a.play(), 100);
  }}
}}

document.addEventListener("keydown", e => {{
  if (e.key === "ArrowRight") nextSentence();
  else if (e.key === "ArrowLeft") prevSentence();
  else if (e.key === " ") {{ e.preventDefault(); togglePlay(); }}
}});

init();
</script>
</body>
</html>'''


def build_player(timeline: dict, out_path: Path) -> None:
    html = HTML_TEMPLATE.format(
        title=timeline.get("title", "讲解"),
        author=timeline.get("author", ""),
        dynasty=timeline.get("dynasty", ""),
        data_json=json.dumps(timeline, ensure_ascii=False),
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"✓ {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="从 timeline.json 生成 HTML 播放器")
    p.add_argument("query", type=str, help="课文查询，如'诫子书 全文'")
    p.add_argument("--lesson", type=str, default=None, help="手动指定课文名")
    p.add_argument("--out", type=str, default=None, help="输出 HTML 路径")
    return p.parse_args()


def main():
    args = parse_args()
    lesson_name = generate._extract_lesson(args.query)
    if args.lesson:
        lesson_name = args.lesson

    # 目录名使用拼音
    lesson_dir_name = generate.lesson_name_to_pinyin(lesson_name)
    lesson_dir = TIMELINE_DIR / lesson_dir_name
    timeline_path = lesson_dir / "timeline.json"
    if not timeline_path.exists():
        print(f"✗ timeline.json 不存在: {timeline_path}")
        print("  请先运行: python3 script/build_timeline.py '{}'".format(args.query))
        sys.exit(1)

    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))

    out_path = Path(args.out) if args.out else lesson_dir / "player.html"
    build_player(timeline, out_path.resolve())


if __name__ == "__main__":
    main()
