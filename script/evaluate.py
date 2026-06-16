#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 B 自动评估：用真实讲解数据作为 ground truth

用法：
  # 从真实讲解数据随机抽样，自动调用 generate 生成对比
  python3 script/evaluate.py --sample 5

  # 指定课程评估
  python3 script/evaluate.py --sample 5 --course "世说新语精读"

  # 对比评估（手动传入模型输出）
  python3 script/evaluate.py --compare "模型生成的讲解文本..." --query "讲德行篇"

  # 生成评估报告
  python3 script/evaluate.py --report

评估流程：
  1. 从 chinese_units 随机抽 N 个真实讲解 unit
  2. 用 unit 的 lesson_slug 构造查询 query（如"讲德行篇"）
  3. 自动调用 generate 模块生成模型输出
  4. 对比模型输出与真实讲解，计算各项指标
"""

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from collections import Counter
from typing import Optional, List, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import generate  # 导入 generate 模块用于自动评估

UNITS_DIR = PROJECT_ROOT / "data" / "chinese_units"
STYLE_STATS = PROJECT_ROOT / "data" / "style" / "style_stats.json"
CONFIG_FILE = SCRIPT_DIR / "config.toml"


def load_style_stats() -> Dict:
    """加载风格统计"""
    if not STYLE_STATS.exists():
        print(f"警告: {STYLE_STATS} 不存在")
        return {}
    with open(STYLE_STATS, "r", encoding="utf-8") as f:
        return json.load(f)


def load_units(course: str = "") -> List[Dict]:
    """加载所有检索单元"""
    units = []
    
    if course:
        course_dir = UNITS_DIR / course
        if course_dir.exists():
            for f in course_dir.glob("*.jsonl"):
                with open(f, "r", encoding="utf-8") as fp:
                    for line in fp:
                        if line.strip():
                            units.append(json.loads(line))
    else:
        for course_dir in UNITS_DIR.iterdir():
            if course_dir.is_dir():
                for f in course_dir.glob("*.jsonl"):
                    with open(f, "r", encoding="utf-8") as fp:
                        for line in fp:
                            if line.strip():
                                units.append(json.loads(line))
    
    return units


def sample_units(n: int, course: str = "") -> List[Dict]:
    """随机抽样 N 个 unit"""
    units = load_units(course)
    if n >= len(units):
        return units
    return random.sample(units, n)


def construct_query(unit: Dict) -> str:
    """从 unit 构造查询语句"""
    lesson = unit.get("lesson", "")
    lesson_slug = unit.get("lesson_slug", "")
    tags = unit.get("tags", [])
    summary = unit.get("summary", "")
    entities = unit.get("entities", [])
    
    # 优先用 lesson_slug 作为查询
    if lesson_slug:
        query = f"讲{lesson_slug}"
    elif lesson:
        query = f"讲{lesson}"
    else:
        query = "请讲解"
    
    # 如果有实体，添加到查询
    if entities:
        query += f"，涉及{entities[0]}"
    
    return query


def extract_sentences(text: str) -> List[str]:
    """提取句子列表"""
    text = text.replace("？", "?").replace("！", "!").replace("。", ".")
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]


def extract_question_sentences(text: str) -> List[str]:
    """提取问句/反问句"""
    questions = []
    for sent in extract_sentences(text):
        if sent.endswith("?") or sent.endswith("？"):
            questions.append(sent)
            continue
        question_words = ["什么", "怎么", "为什么", "是否", "吗", "呢", "是不是", "对吧", "对吗"]
        if any(w in sent for w in question_words):
            questions.append(sent)
    return questions


def count_phrase_hits(text: str, phrases: List[str]) -> Dict:
    """统计口头禅命中情况"""
    hits = {}
    for phrase in phrases:
        count = text.count(phrase)
        if count > 0:
            hits[phrase] = count
    return hits


def calculate_kl_divergence(dist1: List[float], dist2: List[float]) -> float:
    """计算 KL 散度"""
    if len(dist1) != len(dist2):
        max_len = max(len(dist1), len(dist2))
        dist1 = dist1 + [0] * (max_len - len(dist1))
        dist2 = dist2 + [0] * (max_len - len(dist2))
    
    kl = 0.0
    for p, q in zip(dist1, dist2):
        if p > 0 and q > 0:
            kl += p * math.log(p / q)
    return kl


def bucket_sentence_lengths(lengths: List[int], buckets: int = 10) -> List[int]:
    """将句长分桶"""
    if not lengths:
        return [0] * buckets
    
    max_len = 100
    bucket_size = max_len // buckets
    result = [0] * buckets
    
    for length in lengths:
        bucket_idx = min(length // bucket_size, buckets - 1)
        result[bucket_idx] += 1
    
    return result


def normalize_distribution(counts: List[int]) -> List[float]:
    """归一化分布"""
    total = sum(counts)
    if total == 0:
        return counts
    return [c / total for c in counts]


def calculate_text_similarity(text1: str, text2: str) -> Dict:
    """
    计算两段文本的相似度
    
    返回：
    - jaccard_word: 词级 Jaccard 相似度
    - jaccard_char: 字符级 Jaccard 相似度
    - ngram_overlap: n-gram 重叠率
    - topic_overlap: 主题词重叠（基于 entities 和 tags）
    """
    # 字符级 Jaccard
    chars1 = set(text1)
    chars2 = set(text2)
    char_intersection = chars1 & chars2
    char_union = chars1 | chars2
    jaccard_char = len(char_intersection) / len(char_union) if char_union else 0
    
    # 词级 Jaccard（简单分词）
    words1 = set(re.findall(r'[\u4e00-\u9fff]+', text1))
    words2 = set(re.findall(r'[\u4e00-\u9fff]+', text2))
    word_intersection = words1 & words2
    word_union = words1 | words2
    jaccard_word = len(word_intersection) / len(word_union) if word_union else 0
    
    # 2-gram 重叠
    def get_ngrams(text, n=2):
        return set(text[i:i+n] for i in range(len(text) - n + 1))
    
    ngrams1 = get_ngrams(text1, 2)
    ngrams2 = get_ngrams(text2, 2)
    ngram_overlap = len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2) if (ngrams1 | ngrams2) else 0
    
    return {
        "jaccard_char": jaccard_char,
        "jaccard_word": jaccard_word,
        "ngram_overlap": ngram_overlap,
    }


def evaluate_unit(model_output: str, real_text: str, style_stats: Dict) -> Dict:
    """
    评估单个 unit 的模型输出 vs 真实讲解
    
    返回各项指标
    """
    result = {
        "model_length": len(model_output),
        "real_length": len(real_text),
        "length_ratio": len(model_output) / len(real_text) if real_text else 1,
        
        # 文本相似度
        "similarity": calculate_text_similarity(model_output, real_text),
        
        # 模型输出的风格指标
        "model_sentences": extract_sentences(model_output),
        "model_sentence_count": len(extract_sentences(model_output)),
        "model_avg_sentence_length": sum(len(s) for s in extract_sentences(model_output)) / max(len(extract_sentences(model_output)), 1),
        "model_question_count": len(extract_question_sentences(model_output)),
        
        # 真实讲解的风格指标（作为参考）
        "real_sentences": extract_sentences(real_text),
        "real_sentence_count": len(extract_sentences(real_text)),
        "real_avg_sentence_length": sum(len(s) for s in extract_sentences(real_text)) / max(len(extract_sentences(real_text)), 1),
        "real_question_count": len(extract_question_sentences(real_text)),
        "real_text": real_text,  # 保存真实讲解文本
    }
    
    # 口头禅统计（如果有风格统计）
    if style_stats.get("高頻口頭禪"):
        phrases = [item["phrase"] for item in style_stats["高頻口頭禪"][:20]]
        
        model_hits = count_phrase_hits(model_output, phrases)
        real_hits = count_phrase_hits(real_text, phrases)
        
        result["model_phrase_hits"] = model_hits
        result["real_phrase_hits"] = real_hits
        
        # 口头禅命中率
        model_hit_count = len(model_hits)
        result["model_phrase_hit_rate"] = model_hit_count / len(phrases) if phrases else 0
        
        # 口头禅密度（每百字）
        model_density = sum(model_hits.values()) / (len(model_output) / 100) if model_output else 0
        real_density = sum(real_hits.values()) / (len(real_text) / 100) if real_text else 0
        result["model_phrase_density"] = model_density
        result["real_phrase_density"] = real_density
        
        # 口头禅过度使用率
        result["phrase_overuse_rate"] = model_density / real_density if real_density > 0 else 0
    
    return result


def evaluate_with_llm_as_judge(model_output: str, real_text: str, query: str, cfg: Dict) -> Dict:
    """
    使用 LLM-as-Judge 评估
    
    让大模型判断：
    1. 语义覆盖度：模型输出是否覆盖了真实讲解的核心内容
    2. 风格相似度：语气、提问方式是否相似
    3. 是否像真人讲解
    """
    import urllib.request
    
    system = """你是讲解质量评估专家。对比"模型输出"和"真实讲解"，评估模型输出的质量。

评分标准（1-5分）：
- 语义覆盖度：模型输出是否覆盖了真实讲解的核心知识点
- 风格相似度：语气、口头禅、提问方式是否像真实讲解
- 内容准确性：是否有事实错误或幻觉
- 整体质量：综合评分

输出严格 JSON 格式：
{"semantic_coverage": 0, "style_similarity": 0, "factuality": 0, "overall": 0, "comments": "简要评价"}"""

    user = f"""查询：{query}

模型输出：
{model_output[:2000]}

真实讲解：
{real_text[:2000]}

请评估模型输出的质量。"""

    try:
        url = cfg["base_url"].rstrip("/") + "/chat/completions"
        payload = {
            "model": cfg["model"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "stream": False,
        }
        
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg['api_key']}"
            },
            method="POST",
        )
        
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        
        content = data["choices"][0]["message"]["content"]
        content = content.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(content)
    
    except Exception as e:
        return {"error": str(e)}


class Evaluator:
    """评估器"""
    
    def __init__(self, style_stats: Optional[Dict] = None, use_llm_judge: bool = False):
        self.style_stats = style_stats or load_style_stats()
        self.use_llm_judge = use_llm_judge
        self.cfg = {}
        
        if use_llm_judge:
            self.cfg = self._load_config()
    
    def _load_config(self) -> Dict:
        cfg = {"api_key": "", "base_url": "https://api.deepseek.com", "model": "deepseek-chat"}
        if CONFIG_FILE.exists():
            ds = self._read_toml(CONFIG_FILE)
            for k in cfg:
                if ds.get(k):
                    cfg[k] = ds[k]
        return cfg
    
    def _read_toml(self, path: Path) -> Dict:
        try:
            import tomllib
        except ModuleNotFoundError:
            tomllib = None
        
        if tomllib is not None:
            with path.open("rb") as f:
                return tomllib.load(f).get("deepseek", {})
        
        out: Dict = {}
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
    
    def evaluate_sample(self, unit: Dict, model_output: Optional[str] = None) -> Dict:
        """
        评估单个样本
        
        如果不提供 model_output，则使用真实讲解作为 baseline
        """
        real_text = unit.get("text", "")
        
        if model_output is None:
            # 用真实讲解自己评估自己（baseline）
            model_output = real_text
        
        result = {
            "unit_id": unit.get("unit_id", ""),
            "lesson": unit.get("lesson", ""),
            "lesson_slug": unit.get("lesson_slug", ""),
            "tags": unit.get("tags", []),
            "summary": unit.get("summary", ""),
            "entities": unit.get("entities", []),
            "query": construct_query(unit),
        }
        
        # 基础评估
        eval_result = evaluate_unit(model_output, real_text, self.style_stats)
        result.update(eval_result)
        
        # LLM-as-Judge 评估
        if self.use_llm_judge and model_output != real_text:
            llm_result = evaluate_with_llm_as_judge(model_output, real_text, result["query"], self.cfg)
            result["llm_judge"] = llm_result
        
        return result
    
    def evaluate_batch(self, units: List[Dict], model_outputs: Optional[List[str]] = None) -> Dict:
        """批量评估"""
        results = []
        
        for i, unit in enumerate(units):
            model_output = model_outputs[i] if model_outputs and i < len(model_outputs) else None
            result = self.evaluate_sample(unit, model_output)
            results.append(result)
        
        # 聚合指标
        return self._aggregate(results)
    
    def _aggregate(self, results: List[Dict]) -> Dict:
        """聚合评估结果"""
        if not results:
            return {"count": 0}
        
        # 基础统计
        lengths = [r["model_length"] for r in results]
        real_lengths = [r["real_length"] for r in results]
        
        # 相似度统计
        jaccard_chars = [r["similarity"]["jaccard_char"] for r in results]
        jaccard_words = [r["similarity"]["jaccard_word"] for r in results]
        ngram_overlaps = [r["similarity"]["ngram_overlap"] for r in results]
        
        # 句长统计
        model_avg_lengths = [r["model_avg_sentence_length"] for r in results]
        real_avg_lengths = [r["real_avg_sentence_length"] for r in results]
        
        # 问句密度
        model_question_counts = [r["model_question_count"] for r in results]
        real_question_counts = [r["real_question_count"] for r in results]
        
        # 口头禅统计（如果有）
        phrase_hit_rates = [r.get("model_phrase_hit_rate", 0) for r in results if "model_phrase_hit_rate" in r]
        phrase_overuse_rates = [r.get("phrase_overuse_rate", 0) for r in results if "phrase_overuse_rate" in r]
        
        agg = {
            "count": len(results),
            "results": results,
            
            # 长度统计
            "avg_model_length": sum(lengths) / len(lengths),
            "avg_real_length": sum(real_lengths) / len(real_lengths),
            "avg_length_ratio": sum(r["length_ratio"] for r in results) / len(results),
            
            # 相似度统计
            "avg_jaccard_char": sum(jaccard_chars) / len(jaccard_chars),
            "avg_jaccard_word": sum(jaccard_words) / len(jaccard_words),
            "avg_ngram_overlap": sum(ngram_overlaps) / len(ngram_overlaps),
            
            # 句长统计
            "avg_model_sentence_length": sum(model_avg_lengths) / len(model_avg_lengths),
            "avg_real_sentence_length": sum(real_avg_lengths) / len(real_avg_lengths),
            
            # 问句统计
            "avg_model_question_count": sum(model_question_counts) / len(model_question_counts),
            "avg_real_question_count": sum(real_question_counts) / len(real_question_counts),
        }
        
        # 口头禅统计
        if phrase_hit_rates:
            agg["avg_phrase_hit_rate"] = sum(phrase_hit_rates) / len(phrase_hit_rates)
        if phrase_overuse_rates:
            agg["avg_phrase_overuse_rate"] = sum(phrase_overuse_rates) / len(phrase_overuse_rates)
        
        # LLM Judge 统计（如果有）
        llm_overalls = [r["llm_judge"]["overall"] for r in results if "llm_judge" in r and "overall" in r["llm_judge"]]
        if llm_overalls:
            agg["avg_llm_overall"] = sum(llm_overalls) / len(llm_overalls)
        
        return agg


def format_report(agg: Dict) -> str:
    """格式化评估报告"""
    report = []
    report.append("=" * 60)
    report.append("自动评估报告")
    report.append("=" * 60)
    report.append(f"\n样本数: {agg['count']}")
    
    # 长度对比
    report.append(f"\n【长度对比】")
    report.append(f"  模型输出平均长度: {agg['avg_model_length']:.0f} 字")
    report.append(f"  真实讲解平均长度: {agg['avg_real_length']:.0f} 字")
    report.append(f"  长度比: {agg['avg_length_ratio']:.2f}x")
    
    # 文本相似度
    report.append(f"\n【文本相似度】")
    report.append(f"  字符级 Jaccard: {agg['avg_jaccard_char']:.2%}")
    report.append(f"  词级 Jaccard: {agg['avg_jaccard_word']:.2%}")
    report.append(f"  2-gram 重叠: {agg['avg_ngram_overlap']:.2%}")
    
    # 句长对比
    report.append(f"\n【句长对比】")
    report.append(f"  模型输出平均句长: {agg['avg_model_sentence_length']:.1f} 字")
    report.append(f"  真实讲解平均句长: {agg['avg_real_sentence_length']:.1f} 字")
    
    # 问句对比
    report.append(f"\n【问句密度】")
    report.append(f"  模型输出平均问句数: {agg['avg_model_question_count']:.1f}")
    report.append(f"  真实讲解平均问句数: {agg['avg_real_question_count']:.1f}")
    
    # 口头禅（如果有）
    if "avg_phrase_hit_rate" in agg:
        report.append(f"\n【口头禅命中率】{agg['avg_phrase_hit_rate']:.2%}")
        report.append(f"  目标: ≥70%（真实语料均值）")
        report.append(f"  状态: {'✓ 达标' if agg['avg_phrase_hit_rate'] >= 0.7 else '✗ 不达标'}")
        
        if "avg_phrase_overuse_rate" in agg:
            report.append(f"\n【口头禅过度使用率】{agg['avg_phrase_overuse_rate']:.2f}x")
            report.append(f"  目标: ≤1.5x")
            report.append(f"  状态: {'✓ 达标' if agg['avg_phrase_overuse_rate'] <= 1.5 else '✗ 不达标'}")
    
    # LLM Judge（如果有）
    if "avg_llm_overall" in agg:
        report.append(f"\n【LLM-as-Judge 综合评分】{agg['avg_llm_overall']:.2f}/5")
        report.append(f"  目标: ≥4.0/5")
        report.append(f"  状态: {'✓ 达标' if agg['avg_llm_overall'] >= 4.0 else '✗ 不达标'}")
    
    report.append("\n" + "=" * 60)
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="阶段 B 自动评估")
    parser.add_argument("--sample", "-s", type=int, default=10, help="抽样数量（默认 10）")
    parser.add_argument("--course", "-c", type=str, default="", help="指定课程（默认全部）")
    parser.add_argument("--compare", type=str, default="", help="对比评估：传入模型输出文本")
    parser.add_argument("--query", "-q", type=str, default="", help="配合 --compare 使用，指定查询语句")
    parser.add_argument("--llm-judge", action="store_true", help="启用 LLM-as-Judge 评估（需要 API key）")
    parser.add_argument("--json", "-j", action="store_true", help="输出 JSON 格式")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # 加载风格统计
    style_stats = load_style_stats()
    if not style_stats:
        print("错误: 风格统计文件不存在，请先运行 style_stats.py")
        sys.exit(1)
    
    evaluator = Evaluator(style_stats, use_llm_judge=args.llm_judge)
    
    if args.compare:
        # 对比单条
        # 需要找到对应的 unit
        units = load_units(args.course)
        if not units:
            print("错误: 没有找到评估数据")
            sys.exit(1)
        
        # 随机选一个 unit 作为对比
        unit = random.choice(units)
        result = evaluator.evaluate_sample(unit, args.compare)
        
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"查询: {result['query']}")
            print(f"真实讲解: {result['real_text'][:200]}...")
            print(f"\n相似度: {result['similarity']}")
            if "llm_judge" in result:
                print(f"LLM Judge: {result['llm_judge']}")
    
    else:
        # 批量抽样，自动调用 generate.py 生成对比
        units = sample_units(args.sample, args.course)
        print(f"抽样 {len(units)} 个真实讲解单元，自动调用 generate 生成对比...\n")
        
        # 构造查询语句
        model_outputs = []
        for i, unit in enumerate(units):
            lesson = unit.get('lesson', '')
            lesson_slug = unit.get('lesson_slug', '')
            
            # 直接用课程名构造 query，确保 intent 能解析到 lesson
            query = f"讲{lesson}"
            
            print(f"[{i+1}/{len(units)}] {lesson} - {lesson_slug}")
            print(f"    Query: {query}")
            
            try:
                # 调用 generate 模块生成
                cfg = generate.load_config()
                
                # 手动指定 lesson，避免解析失败
                intent = generate.parse_intent(query, cfg)
                intent["lesson"] = lesson  # 强制设置
                
                # 获取课文事实
                facts = generate.fetch_lesson_facts(cfg, query, intent)
                
                if not facts or not facts.get("sentences"):
                    print(f"    ✗ 未获取到课文事实")
                    model_outputs.append("")
                    continue
                
                # 检索相关语料
                enc = generate.Encoder()
                segments = generate.retrieve(query, intent, enc, rerank=False)
                
                # 加载风格
                style = generate.load_style()
                
                # 构造 prompt 并调用 LLM
                system, user = generate.build_prompt(query, intent, facts, style, segments)
                result_text = generate.call_deepseek(cfg, system, user).strip()
                
                model_outputs.append(result_text)
                print(f"    ✓ 生成完成 ({len(result_text)} 字)")
                
            except Exception as e:
                print(f"    ✗ 生成失败: {e}")
                import traceback
                traceback.print_exc()
                model_outputs.append("")
                
            except Exception as e:
                print(f"    ✗ 生成失败: {e}")
                model_outputs.append("")
            
            print()
        
        # 评估对比
        agg = evaluator.evaluate_batch(units, model_outputs)
        
        if args.json:
            print(json.dumps(agg, ensure_ascii=False, indent=2))
        else:
            print("\n" + format_report(agg))


if __name__ == "__main__":
    main()
