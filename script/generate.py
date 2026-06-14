#!/usr/local/bin/python3.13
# -*- coding: utf-8 -*-
"""
B.2 在线生成：解析意图 → 获取课文事实 → 检索相关讲解 → 组装 Prompt → LLM 生成。

输入：用户自然语言请求（如"讲《背影》第二段"）
输出：模拟老师风格的课堂讲解文字

流程：
  1. 意图解析（规则+模型）
  1.5 获取课文硬事实（讲解类意图时调模型获取作者/朝代/出处/概要，注入 system）
  2+3. 向量检索 + 可选 rerank
  4. 风格 Prompt 组装（风格档案 + 课文事实 + few-shot 片段）
  5. LLM 生成

依赖：data/vecdb/（向量库）、data/style/style_profile.json（风格档案）

用法：
  python3 script/generate.py "讲世说新语德行篇第25则"
  python3 script/generate.py "介绍一下顾荣" --lesson "世说新语精读"
  python3 script/generate.py "怎么讲蹒跚这个词" --rerank
  python3 script/generate.py "总结一下洛阳三俊" --verbose
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bge.encode import Encoder

CONFIG_FILE = SCRIPT_DIR / "config.toml"
STYLE_PROFILE = PROJECT_ROOT / "data" / "style" / "style_profile.json"
VECDB = PROJECT_ROOT / "data" / "vecdb"

INTENT_SYSTEM = "你是意图解析器，输出严格 JSON。"

FACTS_SYSTEM = (
    "你是语文课文信息查询助手。根据用户提供的课文名和范围，输出该课文的权威信息。"
    "必须严格基于你训练数据中的知识，禁止编造。如果某条信息不确定，对应字段留空字符串。"
    "输出严格 JSON，不要输出其他内容。"
)


# ------------------------------------------------------------------- 配置 & 模型调用

def load_config() -> dict:
    cfg = {"api_key": "", "base_url": "https://api.deepseek.com", "model": "deepseek-chat"}
    if CONFIG_FILE.exists():
        ds = _read_toml(CONFIG_FILE)
        for k in cfg:
            if ds.get(k):
                cfg[k] = ds[k]
    env_key = __import__("os").environ.get("DEEPSEEK_API_KEY")
    if env_key:
        cfg["api_key"] = env_key
    return cfg


def _read_toml(path: Path) -> dict:
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


def call_deepseek(cfg: dict, system: str, user: str) -> str:
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "stream": False,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['api_key']}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


# ------------------------------------------------------------------- 步骤 1: 意图解析

def parse_intent(query: str, cfg: "dict | None" = None) -> dict:
    """规则+模型双路径解析用户意图。"""
    # 先规则匹配
    lesson = _extract_lesson(query)
    scope = _extract_scope(query)
    intent = _extract_intent(query)

    # 规则没拿到 lesson 且有模型可用时，调模型
    if not lesson and cfg and cfg.get("api_key"):
        try:
            content = call_deepseek(cfg, INTENT_SYSTEM, _intent_prompt(query))
            content = content.strip().removeprefix("```json").removesuffix("```").strip()
            parsed = json.loads(content)
            lesson = parsed.get("lesson", "") or lesson
            scope = parsed.get("scope", "") or scope
            model_intent = parsed.get("intent", "")
            # 只接受已知意图，防止模型返回自由文本
            VALID_INTENTS = {"精读讲解", "字词分析", "背景导入", "概括大意"}
            intent = model_intent if model_intent in VALID_INTENTS else intent
        except Exception:
            pass

    return {
        "lesson": lesson,
        "scope": scope,
        "intent": intent or "精读讲解",
    }


# ------------------------------------------------------------------- 步骤 1.5: 获取课文事实信息

def fetch_lesson_facts(cfg: dict, query: str, intent: dict) -> "dict | None":
    """调用大模型获取课文作者、朝代、出处、原文概要等硬事实。

    只在意图为讲解类（精读/字词/背景）时触发，避免无关场景浪费调用。
    返回的 facts 将注入 system prompt 作为生成时的"已知事实"，禁止模型自行编造。
    """
    text_intents = {"精读讲解", "字词分析", "背景导入", "概括大意"}
    if intent.get("intent", "") not in text_intents:
        return None

    lesson = intent.get("lesson", "")
    scope = intent.get("scope", "")
    if not lesson:
        return None

    user = (
        f"用户想了解以下课文信息：\n"
        f"课文：{lesson}\n" + (f"范围：{scope}\n" if scope else "")
        + "\n请输出 JSON，其中 sentences 要把指定范围的原文**逐句拆开**，"
        "每句标出需要精讲的重点字词及其释义（实词/虚词/活用/古今异义/通假等优先）：\n"
        '{"author":"作者名","dynasty":"朝代","source":"出处/选自",'
        '"excerpt":"指定范围的完整课文原文（指定了范围只输出该范围；否则输出核心段落）",'
        '"sentences":[{"text":"原文一句","translation":"该句白话翻译",'
        '"keywords":[{"word":"重点字词","note":"释义/用法"}]}],'
        '"synopsis":"课文内容概要（100字内）","keyPoints":["关键知识点1","关键知识点2"]}\n'
        "要求：sentences 按原文顺序逐句排列，不要遗漏；keywords 是该句真正的考点字词，"
        "没有重点词的句子 keywords 可为空数组。如果某项不确定，留空字符串。只输出 JSON。"
    )

    try:
        content = call_deepseek(cfg, FACTS_SYSTEM, user)
        content = content.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(content)
    except Exception:
        return None


def _extract_lesson(q: str) -> str:
    """规则提取课文名。"""
    # 《XXX》格式
    m = re.search(r"《(.+?)》", q)
    if m:
        return m.group(1)
    # 已知课程名
    courses = ["世说新语", "背影", "滕王阁序", "诫子书", "爱莲说", "陋室铭", "桃花源记", "岳阳楼记", "醉翁亭记"]
    for c in courses:
        if c in q:
            return c
    return ""


def _extract_scope(q: str) -> str:
    m = re.search(r"第[一二三四五六七八九十\d]+[则节段篇讲]", q)
    return m.group(0) if m else ""


def _extract_intent(q: str) -> str:
    for kw, intent in [
        ("概括", "概括大意"), ("总结", "概括大意"), ("大意", "概括大意"),
        ("考试", "考试点拨"), ("考点", "考试点拨"), ("应试", "考试点拨"),
        ("介绍", "背景导入"), ("导入", "背景导入"), ("背景", "背景导入"),
        ("精读", "精读讲解"), ("讲解", "精读讲解"), ("分析", "精读讲解"),
        ("字词", "字词分析"),
    ]:
        if kw in q:
            return intent
    return ""


def _intent_prompt(q: str) -> str:
    return f"""从这句话提取：课文名、范围、意图。
输出 JSON：{{"lesson":"","scope":"","intent":""}}

用户输入：{q}"""


# ------------------------------------------------------------------- 步骤 2+3: 检索

def retrieve(
    query: str, intent: dict, enc: Encoder,
    top: int = 5, rerank: bool = False,
) -> list[dict]:
    """向量检索 + 可选 rerank，返回 top-N 段。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(VECDB))
    col = client.get_collection("teacher_units")

    # bge query 前缀
    q_text = "为这句话设计课堂讲解：" + query
    q_vec = enc.encode([q_text])[0]

    where = None
    if intent.get("lesson"):
        # 模糊匹配课程名
        where = {"lesson": intent["lesson"]}

    recall = 20 if rerank else top
    recall = min(recall, col.count())
    res = col.query(query_embeddings=[q_vec], n_results=recall, where=where)

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    ids = res["ids"][0]

    if rerank and len(docs) > top:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
        pairs = [[query, d] for d in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, ids, docs, metas), reverse=True)[:top]
        return [
            {"id": r_id, "text": r_doc, "meta": r_meta, "score": float(r_score)}
            for r_score, r_id, r_doc, r_meta in ranked
        ]

    return [
        {"id": did, "text": doc, "meta": meta}
        for did, doc, meta in zip(ids, docs, metas)
    ]


# ------------------------------------------------------------------- 步骤 4: 组装 Prompt

def load_style() -> dict:
    if STYLE_PROFILE.exists():
        return json.loads(STYLE_PROFILE.read_text(encoding="utf-8"))
    return {}


def build_prompt(
    query: str, intent: dict, segments: list[dict], style: dict,
    facts: "dict | None" = None,
) -> tuple[str, str]:
    """组装 system + user prompt。"""
    # System: 风格设定
    style_lines = ["你是语文老师的课堂克隆，请严格遵循以下风格设定："]
    if style.get("persona"):
        style_lines.append(f"人格：{style['persona']}")
    for key in ["口头禅", "开场套路", "提问方式", "举例偏好",
                "句式特征", "禁忌", "讲解结构"]:
        val = style.get(key)
        if val:
            if isinstance(val, list):
                style_lines.append(f"{key}：{'、'.join(val)}")
            else:
                style_lines.append(f"{key}：{val}")

    # 注入课文硬事实：作者、朝代、出处、原文等，作为不可违背的已知事实
    if facts:
        fact_lines = ["\n【课文硬事实——以下信息必须严格遵守，禁止自行编造】"]
        if facts.get("author"):
            fact_lines.append(f"作者：{facts['author']}")
        if facts.get("dynasty"):
            fact_lines.append(f"朝代：{facts['dynasty']}")
        if facts.get("source"):
            fact_lines.append(f"出处：{facts['source']}")
        if facts.get("excerpt"):
            fact_lines.append(f"原文：{facts['excerpt']}")
        if facts.get("synopsis"):
            fact_lines.append(f"内容概要：{facts['synopsis']}")
        if facts.get("keyPoints"):
            kp = "、".join(facts["keyPoints"])
            fact_lines.append(f"关键知识点：{kp}")

        # 逐句逐词讲解骨架：每句原文 + 翻译 + 重点字词释义
        sentences = facts.get("sentences") or []
        if sentences:
            fact_lines.append(
                "\n【逐句精讲骨架——讲解时必须逐句逐词覆盖，字词释义以此为准】"
            )
            for i, s in enumerate(sentences, 1):
                line = f"{i}. 原句：{s.get('text', '')}"
                if s.get("translation"):
                    line += f"\n   译：{s['translation']}"
                kws = s.get("keywords") or []
                if kws:
                    kw_str = "；".join(
                        f"{k.get('word', '')}={k.get('note', '')}" for k in kws
                    )
                    line += f"\n   重点词：{kw_str}"
                fact_lines.append(line)
        style_lines.extend(fact_lines)

    system = "\n".join(style_lines)

    # User: few-shot + task
    few_shot = ""
    if segments:
        samples = "\n---\n".join(
            f"片段{i + 1}: {s['text'][:300]}" for i, s in enumerate(segments)
        )
        few_shot = (
            "以下是这位老师讲过的真实片段（仅供参考风格，不要照抄）：\n"
            "---\n"
            f"{samples}\n"
            "---\n\n"
        )

    lesson = intent.get("lesson", "")
    scope = intent.get("scope", "")
    task_intent = intent.get("intent", "")
    has_skeleton = bool(facts and facts.get("sentences"))

    if has_skeleton:
        user = (
            f"{few_shot}"
            f"用户请求：{query}\n"
            + (f"课文：{lesson}，范围：{scope}，意图：{task_intent}\n\n" if lesson else "\n")
            + "请按以下格式输出一堂**逐句逐词精讲**的课：\n\n"
            "【课文信息】\n"
            "列出作者、朝代、出处（引用 system 硬事实，禁止修改）。\n\n"
            "【精讲正文】\n"
            "按 system 中【逐句精讲骨架】的顺序往下讲，覆盖每一句、每个重点词：\n"
            "- 篇幅短就**一句一句**地讲；篇幅长时可把意思连贯的几句**合成一段**、"
            "**逐段**地讲，先引这一段原文再展开，但段内仍要把每个重点词逐个落到，不能跳过。\n"
            "每一（句/段）都要做到：\n"
            "1. 先完整引用这一句（或这一段）原文（照抄骨架，不得改字）。\n"
            "2. 再**逐个**解读其中的重点字词——读音、本义、在这里的意思、词类活用/古今异义等，"
            "释义以骨架为准，不要自行发挥成别的意思。\n"
            "3. 然后把整句（整段）串讲一遍，点出它好在哪、要体会什么。\n"
            "4. 全程用第一人称\"我\"的课堂口吻，自然带出风格设定里的口头禅、提问、追问、举例，"
            "让解读听起来像真在上课，而不是查字典。\n\n"
            "要求：\n"
            "- **不限字数**，该展开就展开，宁可长也不要概括省略；每一句、每个重点词都要落到。\n"
            "- 字词读音、释义、出处必须严格依据 system 提供的事实，宁可不展开也不要编。\n"
            "- 句与句、段与段之间用老师的过渡语自然衔接。\n"
            "- 结尾可用提问或互动引导。\n"
        )
    else:
        user = (
            f"{few_shot}"
            f"用户请求：{query}\n"
            + (f"课文：{lesson}，范围：{scope}，意图：{task_intent}\n\n" if lesson else "\n")
            + "请按以下格式输出：\n\n"
            "【课文信息】\n"
            "列出课文的作者、朝代、出处、原文（直接引用 system 中【课文硬事实】的内容，禁止修改）。\n\n"
            "【课堂讲解】\n"
            "然后进入讲解，要求：\n"
            "1. 用第一人称\"我\"，课堂口吻。\n"
            "2. 体现风格设定中的口头禅、句式、提问方式。\n"
            "3. 可参考 few-shot 片段的讲解结构，但不要照搬。\n"
            "4. **不限字数**，把内容讲透，不要为了简短而概括省略。\n"
            "5. 结尾可用提问或互动引导。\n"
        )

    return system, user


# ------------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B.2 在线生成：RAG + 风格 Prompt")
    p.add_argument("query", type=str, help="用户请求")
    p.add_argument("--lesson", type=str, default=None,
                   help="手动指定课文名")
    p.add_argument("--top", type=int, default=5, help="检索段数")
    p.add_argument("--rerank", action="store_true", help="启用 reranker 精排")
    p.add_argument("--model-dir", type=str, default=None,
                   help="bge 模型目录")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="打印检索结果和 Prompt 详情")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config()
    if not cfg["api_key"]:
        print("✗ 未找到 DeepSeek api_key。")
        sys.exit(1)

    # 步骤 1: 意图解析
    intent = parse_intent(args.query, cfg)
    if args.lesson:
        intent["lesson"] = args.lesson

    print(f"意图解析: {json.dumps(intent, ensure_ascii=False)}")
    if args.verbose:
        print(f"原始查询: {args.query}\n")

    # 步骤 1.5: 获取课文硬事实（讲解类意图才触发）
    facts = fetch_lesson_facts(cfg, args.query, intent)
    if facts and args.verbose:
        print(f"课文事实: {json.dumps(facts, ensure_ascii=False)}\n")

    # 步骤 2+3: 检索
    enc = Encoder(args.model_dir)
    segments = retrieve(args.query, intent, enc, args.top, args.rerank)

    if args.verbose:
        print(f"检索到 {len(segments)} 段:")
        for s in segments:
            print(f"  {s['id']}  [{s['meta'].get('tags','')}]  "
                  f"{s['text'][:80].replace(chr(10),' ')}……")
        print()

    # 步骤 4: 组装 Prompt
    style = load_style()
    system, user = build_prompt(args.query, intent, segments, style, facts)

    if args.verbose:
        print("=== SYSTEM ===")
        print(system[:500], "\n...\n")
        print("=== USER ===")
        print(user[:1000], "\n...\n")

    # 步骤 5: 生成
    print(f"模型生成中 ({cfg['model']}) ...\n")
    t0 = time.time()
    try:
        result = call_deepseek(cfg, system, user)
    except Exception as e:
        print(f"✗ 生成失败: {e}")
        sys.exit(1)

    dt = int(time.time() - t0)
    print("=" * 50)
    print(result)
    print("=" * 50)
    print(f"\n生成耗时: {dt}s")


if __name__ == "__main__":
    main()
