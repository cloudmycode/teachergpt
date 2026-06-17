#!/usr/local/bin/python3.13
# -*- coding: utf-8 -*-
"""
B.2 在线生成库：意图解析 / 课文事实 / 检索 / 风格 Prompt 组装。

本模块是**库**（无 main），供 build_timeline.py 等编排脚本调用。
所有“和大模型交互 + prompt 构建”都集中在这里：

  - 配置 & 模型调用：load_config / call_deepseek
  - 步骤 1   意图解析：parse_intent
  - 步骤 1.5 课文事实：fetch_lesson_facts
  - 步骤 2+3 检索：    retrieve
  - 步骤 4   Prompt：  load_style / build_style_system / build_few_shot
                       / build_intro_prompt / build_per_sentence_prompt
  - 工具：            lesson_name_to_pinyin

依赖：data/vecdb/（向量库）、data/style/style_profile.json（风格档案）
"""

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

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

    返回的 sentences 中可包含可选的 "paragraph" 字段（段落索引，从 0 开始），
    用于按段落分批生成讲解。
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
        "每句标出需要精讲的重点字词及其释义（实词/虚词/活用/古今异义/通假等优先）。"
        "同时为每句标注 paragraph 字段（段落索引，从 0 开始），"
        "段落按原文自然段划分（空行分隔）。\n"
        '{"author":"作者名","dynasty":"朝代","source":"出处/选自",'
        '"excerpt":"指定范围的完整课文原文（指定了范围只输出该范围；否则输出核心段落）",'
        '"sentences":[{"text":"原文一句","translation":"该句白话翻译","paragraph":0,'
        '"keywords":[{"word":"重点字词","note":"释义/用法"}]}],'
        '"synopsis":"课文内容概要（100字内）","background":"写作背景（50字内，作者在什么境遇/动机下写的）",'
        '"keyPoints":["关键知识点1","关键知识点2"]}\n'
        "要求：sentences 按原文顺序逐句排列，不要遗漏；keywords 是该句真正的考点字词，"
        "没有重点词的句子 keywords 可为空数组。paragraph 用于按段落分批生成，"
        "没有段落信息时可不填。如果某项不确定，留空字符串。只输出 JSON。"
    )

    try:
        content = call_deepseek(cfg, FACTS_SYSTEM, user)
        content = content.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(content)
        # 注入篇名：用户实际要讲的文章名（如《卖油翁》），
        # 避免把出处（source，如《归田录》）当成标题来念。
        if isinstance(data, dict):
            data["title"] = lesson
        return data
    except Exception:
        return None


def lesson_name_to_pinyin(name: str) -> str:
    """将文章名转为拼音，用于目录名（服务器兼容性更好）。
    
    例：木兰词 -> mulan_ci, 诫子书 -> jiezi_shu
    """
    try:
        from pypinyin import pinyin, Style
        # 转拼音，使用 normal 风格（无声调）
        py_list = pinyin(name, style=Style.NORMAL)
        # 合并每个字的拼音
        result = "_".join("".join(py) for py in py_list)
        # 转小写，移除空格
        return result.lower().replace(" ", "")
    except ImportError:
        # pypinyin 未安装时，直接返回原名
        return name


def _extract_lesson(q: str) -> str:
    """规则提取课文名。"""
    # 《XXX》格式
    m = re.search(r"《(.+?)》", q)
    if m:
        return m.group(1)
    
    # 篇名到课程的映射（世说新语的篇名 → 世说新语精读）
    chapter_to_course = {
        "德行篇": "世说新语精读",
        "言语篇": "世说新语精读",
        "政事篇": "世说新语精读",
        "文学篇": "世说新语精读",
        "方正篇": "世说新语精读",
        "雅量篇": "世说新语精读",
        "识鉴篇": "世说新语精读",
        "赏誉篇": "世说新语精读",
        "品藻篇": "世说新语精读",
        "规箴篇": "世说新语精读",
        "捷悟篇": "世说新语精读",
        "夙慧篇": "世说新语精读",
        "豪爽篇": "世说新语精读",
        "容止篇": "世说新语精读",
        "自新篇": "世说新语精读",
        "俭啬篇": "世说新语精读",
        "汰侈篇": "世说新语精读",
        "忿狷篇": "世说新语精读",
        "情礼篇": "世说新语精读",
        "黜免篇": "世说新语精读",
        "俭吝篇": "世说新语精读",
        "惑溺篇": "世说新语精读",
        "仇隙篇": "世说新语精读",
        "任诞篇": "世说新语精读",
        "伤逝篇": "世说新语精读",
        "栖逸篇": "世说新语精读",
        "贤媛篇": "世说新语精读",
        "术解篇": "世说新语精读",
        "巧艺篇": "世说新语精读",
        "知惧篇": "世说新语精读",
        "企羡篇": "世说新语精读",
        "黜免篇": "世说新语精读",
    }
    for chapter, course in chapter_to_course.items():
        if chapter in q:
            return course
    
    # 已知课程名
    courses = ["世说新语", "背影", "滕王阁序", "诫子书", "爱莲说", "陋室铭", "桃花源记", "岳阳楼记", "醉翁亭记", "木兰诗", "木兰辞"]
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
    verbose: bool = False,
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
    distances = res["distances"][0] if res.get("distances") else [0.0] * len(ids)

    if verbose:
        print(f"\n{'='*60}")
        print(f"ChromaDB 查询详情")
        print(f"{'='*60}")
        print(f"  集合: teacher_units")
        print(f"  总单元数: {col.count()}")
        print(f"  查询文本: {q_text}")
        print(f"  过滤条件: {where or '无'}")
        print(f"  召回数量: {len(docs)}")
        print(f"  是否 rerank: {rerank}")
        print(f"{'='*60}")
        print(f"\n召回结果（按相似度排序）:")
        for i, (did, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances)):
            similarity = 1 - dist
            print(f"\n  [{i+1}] unit_id: {did}")
            print(f"      lesson: {meta.get('lesson', 'N/A')}")
            print(f"      tags: {meta.get('tags', 'N/A')}")
            print(f"      summary: {meta.get('summary', 'N/A')[:80]}...")
            print(f"      paras: {meta.get('paras', 'N/A')}")
            print(f"      相似度: {similarity:.4f} (距离: {dist:.4f})")
            text_preview = doc[:200].replace('\n', ' ')
            print(f"      文本: {text_preview}...")

    if rerank and len(docs) > top:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
        pairs = [[query, d] for d in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, ids, docs, metas), reverse=True)[:top]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Rerank 精排结果 (top-{top})")
            print(f"{'='*60}")
            for i, (r_score, r_id, r_doc, r_meta) in enumerate(ranked):
                print(f"\n  [{i+1}] unit_id: {r_id}")
                print(f"      rerank 分数: {r_score:.4f}")
                print(f"      lesson: {r_meta.get('lesson', 'N/A')}")
                print(f"      tags: {r_meta.get('tags', 'N/A')}")
                text_preview = r_doc[:150].replace('\n', ' ')
                print(f"      文本: {text_preview}...")
        
        return [
            {"id": r_id, "text": r_doc, "meta": r_meta, "score": float(r_score)}
            for r_score, r_id, r_doc, r_meta in ranked
        ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"最终检索结果 (top-{top})")
        print(f"{'='*60}")
        for i, (did, doc, meta, dist) in enumerate(zip(ids[:top], docs[:top], metas[:top], distances[:top])):
            print(f"\n  [{i+1}] unit_id: {did}")
            print(f"      lesson: {meta.get('lesson', 'N/A')}")
            print(f"      tags: {meta.get('tags', 'N/A')}")
            print(f"      相似度: {1-dist:.4f}")
            text_preview = doc[:200].replace('\n', ' ')
            print(f"      文本: {text_preview}...")

    return [
        {"id": did, "text": doc, "meta": meta}
        for did, doc, meta in zip(ids[:top], docs[:top], metas[:top])
    ]


# ------------------------------------------------------------------- 步骤 4: 组装 Prompt

def load_style() -> dict:
    if STYLE_PROFILE.exists():
        return json.loads(STYLE_PROFILE.read_text(encoding="utf-8"))
    return {}


def build_style_system(style: dict) -> str:
    """构建风格 system prompt（老师人格 + 口头禅/句式/禁忌等风格设定）。"""
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


def build_few_shot(segments: list[dict], limit: int = 3, maxlen: int = 200) -> str:
    """把检索到的真实讲解片段拼成 few-shot 提示块（不要照抄，仅参考风格）。"""
    if not segments:
        return ""
    samples = "\n---\n".join(
        f"片段{i + 1}: {s['text'][:maxlen]}"
        for i, s in enumerate(segments[:limit])
    )
    return (
        "以下是这位老师讲过的真实片段（参考风格，不要照抄）：\n"
        "---\n"
        f"{samples}\n"
        "---\n\n"
    )


PER_INTRO_USER = """{few_shot}
{context}

请为这节课写一段**开头导入语**（50-100字）。

要求：
1. 用第一人称"我"的课堂口吻。
2. 简要介绍作者、朝代和写作背景（从上下文获取），自然引入本课主题，可以抛出一个有趣的问题或悬念。
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
3. 然后**直接串讲**（不要说"我们把这句话串讲一下"之类的预告，直接开始）：
   先一字不差地念出文言原句，再用"就是说"接白话翻译，
   紧接着直接说出它的好处、学生要体会什么。**不要说"这句话好在哪呢","这个好处是什么呢","我们来分析一下"这类预设问和预告，直接说内容。**
   例如："初，权谓吕蒙曰：'卿今当涂掌事，不可不学！'
   就是说，当初孙权对吕蒙说：'你现在当权掌事，不能不学习啊！'"
   不要跳过文言原句直接说白话。
4. 上一句结尾为参考，开头自然过渡衔接，可以用不同的过渡方式：
   - 直接承接上文："这句"
   - 提出问题："这个字是什么意思呢？"
   - 简单过渡："接下来看""来看看这句"
   - 大部分情况可以不用过渡语，直接开始讲解
   **注意：不要每次都用"好"开头，要变化多样。**
5. 不限字数，需要展开时则展开讲。
6. **重要**：以下是本句所有的重点词，你必须逐个讲解，并且在讲到该词时用方括号 [ ] 标记。
   例如讲解"静"字时写成 [静]。注意：仅在讲解该词含义时标记，
   开头引用原文（如"静以修身"）时不要标记。必须标记所有重点词，一个都不能漏。

请直接输出讲解文本，不要加标题，不要输出 JSON。"""


def _wrap_title(name: str) -> str:
    """把篇名/出处规范化为带书名号形式；空值返回空串。"""
    name = (name or "").strip().strip("《》").strip()
    return f"《{name}》" if name else ""


def _lesson_intro_line(facts: dict, lead: str) -> str:
    """生成 "你正在讲《篇名》（作者，朝代，选自《出处》）。" 这类引导句。

    篇名（title）优先，出处（source）仅作"选自"补充，避免把出处当成标题。
    """
    author = facts.get("author", "")
    dynasty = facts.get("dynasty", "")
    title_disp = _wrap_title(facts.get("title", "")) or _wrap_title(
        facts.get("source", "")
    )
    src_disp = _wrap_title(facts.get("source", ""))
    extra = f"，选自{src_disp}" if src_disp and src_disp != title_disp else ""
    subject = title_disp or "课文"
    meta = "，".join(p for p in [author, dynasty] if p)
    if meta or extra:
        return f"{lead}{subject}（{meta}{extra}）。"
    return f"{lead}{subject}。"


def _build_context(
    sentence_idx: int,
    total: int,
    facts: dict,
    prev_narration: str,
) -> str:
    """逐句讲解的上下文：课文信息 + 全文句列表（标出当前句）+ 上一句结尾。"""
    parts = []

    parts.append(_lesson_intro_line(facts, "你正在讲"))
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

    parts.append(_lesson_intro_line(facts, "你正在开始讲"))
    if facts.get("synopsis"):
        parts.append(f"全文大意：{facts['synopsis']}")
    if facts.get("background"):
        parts.append(f"写作背景：{facts['background']}")

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
    """逐句讲解 prompt（system 风格 + user 逐句任务）。"""
    system = build_style_system(style)
    context = _build_context(sentence_idx, total, facts, prev_narration)

    kws = sentence.get("keywords") or []
    keyword_info = ""
    if kws:
        keyword_info = "重点词：" + "、".join(
            f"{k['word']}({k['note']})" for k in kws
        )

    user = PER_SENTENCE_USER.format(
        few_shot=build_few_shot(segments),
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
    """开头导入语 prompt（system 风格 + user 导入任务）。"""
    system = build_style_system(style)
    context = _build_intro_context(facts)

    user = PER_INTRO_USER.format(
        few_shot=build_few_shot(segments),
        context=context,
    )
    return system, user
