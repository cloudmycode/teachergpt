#!/usr/bin/env python3
"""把 docs/design.md 转成格式化 HTML，输出 docs/index.html。"""

import markdown
from pathlib import Path

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI 仿真老师 — 技术方案</title>
<style>
  :root {
    --bg: #fafafa;
    --fg: #1f2937;
    --border: #e5e7eb;
    --accent: #2563eb;
    --code-bg: #f3f4f6;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    line-height: 1.7;
    padding: 2rem 1rem;
  }
  .container {
    max-width: 820px;
    margin: 0 auto;
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
    padding: 2.5rem 3rem;
  }
  h1 { font-size: 2rem; margin: 0 0 1rem; padding-bottom: .5rem; border-bottom: 2px solid var(--border); }
  h2 { font-size: 1.5rem; margin: 1.5rem 0 .8rem; color: #111827; }
  h3 { font-size: 1.2rem; margin: 1.2rem 0 .6rem; }
  h4, h5 { margin: 1rem 0 .4rem; }
  p { margin: .6rem 0; }
  ul, ol { margin: .6rem 0; padding-left: 1.6rem; }
  blockquote {
    margin: 1rem 0;
    padding: .6rem 1rem;
    border-left: 4px solid var(--accent);
    background: #eff6ff;
    color: #374151;
  }
  code {
    background: var(--code-bg);
    padding: .1em .4em;
    border-radius: 4px;
    font-size: .9em;
  }
  pre {
    background: #1f2937;
    color: #e5e7eb;
    padding: 1rem;
    border-radius: 6px;
    overflow-x: auto;
    margin: 1rem 0;
  }
  pre code { background: none; padding: 0; color: inherit; }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: .95rem;
  }
  th, td {
    border: 1px solid var(--border);
    padding: .5rem .7rem;
    text-align: left;
  }
  th { background: #f9fafb; font-weight: 600; }
  a { color: var(--accent); }
  hr { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }
  @media (max-width: 640px) {
    .container { padding: 1.5rem 1rem; }
  }
</style>
</head>
<body>
<div class="container">
{body}
</div>
</body>
</html>"""

def main():
    md_path = Path(__file__).parent / "design.md"
    html_path = Path(__file__).parent / "index.html"

    md_text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "codehilite", "toc"],
    )
    html = TEMPLATE.replace("{body}", html_body)
    html_path.write_text(html, encoding="utf-8")
    print(f"✓ 已生成: {html_path}")

if __name__ == "__main__":
    main()
