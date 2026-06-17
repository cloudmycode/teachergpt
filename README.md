# 实现AI仿真老师 —— 技术方案与实现路径
https://www.jingjiangke.com   - 域名是“精讲课”的拼音
> 目标：用 ~1000 节课的音频课程，建立本地知识库，提取讲课风格微调大模型，做出一个尽量接近真实老师上课内容和风格的模型，并工程化推进产品，让它能讲任意指定的语文课。

### 读完这篇文档，你能学到什么

这篇文档不仅是一份技术方案，也是一份 AI 应用开发的实战笔记。熟悉后你将掌握：

**1. 大模型应用开发的核心模式**
- **RAG（检索增强生成）**：如何用向量数据库 + 语义检索，让模型基于真实材料生成内容
- **Prompt Engineering**：如何设计 System Prompt 让模型扮演特定角色（本项目是"仿真老师"）
- **LoRA 微调**：当 Prompt 不够时，如何用少量数据微调模型"焊入"风格

**2. 从数据到产品的完整链路**
- 音频转录 → 文本清洗 → 结构化切分 → 向量化索引 → 在线检索 → Prompt 组装 → LLM 生成
- 每个环节都有对应脚本，可直接运行

**3. NLP 实战技能**
- 语料清洗与结构化（正则、分句、实体提取）
- 向量数据库（ChromaDB）的使用
- Embedding 模型（BGE）与 Reranker 的配合
- 风格统计与量化评估

**4. 工程化思维**
- 分阶段交付：先跑通再优化，RAG 先行，LoRA 兜底
- 可验证的评估体系：口头禅命中率、句长分布、LLM-as-Judge
- 增量更新与冷启动策略

**5. 一个完整的 AI 项目**
- 从需求分析到上线部署的全貌
- 脚本调用关系与数据流转
- 可复用的架构模式

---

## 0. 拆清问题

"真实老师"包含三个维度的特点：

| 维度 | 含义 | 数据来源 | 难度 |
|------|------|---------|------|
| **风格 Style** | 口头禅、语气、节奏、怎么打比方、怎么和学生互动 | 转录文本里大量存在 | 中 |
| **教学法 Pedagogy** | 怎么导入、怎么拆段、提问设计、情感升华的套路 | 需从转录里结构化抽取 | 高 |
| **知识 Knowledge** | 具体课文的讲法、考点、背景 | 部分在转录里，部分要外部教材补 | 中 |

产品上真正打动人的是 **Style + Pedagogy**。知识可以靠 RAG 和底座模型补，本文只处理文本讲解风格。

---

## 1. 实现方案对

### RAG（真人语料检索）+ 风格 Prompt，跑通后再叠加 LoRA
**分阶段叠加**。核心思路：

> 让老师"讲过的真话"被检索出来当上下文，而不是让模型凭空模仿。

- RAG 检索的是**真实老师对课文/知识点的真实讲法片段**，模型基于真材料改写 → 风格和知识同时贴近，幻觉大幅下降。
- 风格 Prompt 负责兜底语气和结构。
- 当 RAG+Prompt 的风格保真度被验证为不够时，再用同一批语料做 LoRA，把风格"焊进"模型，RAG 继续负责知识。

---

## 2. 落地路线（按阶段，可随时止步）

```
阶段 A: 数据底座（必做，所有方案共用）
   └─ 这步做扎实，后面随便选方案

阶段 B: RAG + 风格 Prompt  → 先上线、拿基线、跑评估
   └─ 多数情况这一步就够产品用了

阶段 C: LoRA 风格微调      → 当 B 的风格保真度不达标再做
   └─ RAG 继续管知识，LoRA 管风格
```
---

## 3. 阶段 A：数据底座（最重要，决定上限）

这是整个项目真正的工作量所在。产出物是结构化、可检索、可训练的语料库。

### A.1 转录清洗

这一步的目标：把 Whisper 原始输出（时间戳 + 带错字的口语文本）变成**可直接用于 RAG 和训练的干净结构化片段**。Whisper 原始文本常见问题：同音错别字（如"蹒跚"→"盘山"）、口语冗余、没有段落、专有名词错。

#### 步骤 1：Whisper 转录
- 输入：每节课的 mp3 文件。
- 工具：`asr/transcribe.py`（faster-whisper medium）。
- 输出：每节课一个 txt，格式为 `start_time → end_time | 文本`，例如：
  ```
  312.40s → 315.20s | 同学们注意啊，这个盘山两个字
  315.20s → 320.10s | 你们体会一下这个父亲爬月台的样子
  ```
- 同时可选输出 JSON（`--json`），包含段级和词级时间戳，后续切分和定位用得上。

#### 步骤 2：大模型纠错
- 输入：Whisper 原始 txt + 对应的音频时间戳（用于必要时回听）。
- 工具：`script/clean_transcription.py`（依赖DeepSeek上下文模型）。
- 方法：把同一节课的所有段作为一个上下文窗口送进模型，要求它：
  1. **纠正错别字**：结合上下文判断同音字，优先参考语文教材常见词。例如"盘山"→"蹒跚"。
  2. **修正专有名词**：课文名、作者名、典故名。可额外提供一份《语文课本人名地名词表》作为 few-shot 提示，提升准确率。
  3. **保留口头禅和语气词**："同学们注意啊""是不是""你们体会一下"等是风格本身，**绝对不能删**。只删纯粹的口水词（"嗯""呃"单独出现且无意义时）。
  4. **不断句**：这一步只做字符级纠错，不重新组织句子。
- Prompt 模板示例：
  ```
  你是语文课转录纠错员。下面是一节语文课的语音识别结果，存在同音错别字。
  请逐行纠正，规则：
  1. 只纠正明显的同音错别字，尤其是人名、地名、课文名、四字词语。
  2. 保留所有口头禅、语气词、学生互动内容。
  3. 不要重新组织句子，只做字符替换。
  4. 输出格式与输入相同：start_time → end_time | 纠错后文本

  【常见词表参考】
  蹒跚 朱自清 颐和园 荷塘月色 背影 父亲 月台 橘子 紫色 深青

  【原始转录】
  312.40s → 315.20s | 同学们注意啊，这个盘山两个字
  315.20s → 320.10s | 你们体会一下这个父亲爬月台的样子
  ...
  ```
- 输出：纠错后的 txt，格式不变，时间戳和行号一一对应。
- 质量兜底：对高频纠错词（如"盘山→蹒跚"）随机抽 10% 回听原始音频验证；低置信的可人工复核。

#### 步骤 3：断句与段落合并
- 工具：`script/clean_transcription.py`（依赖DeepSeek上下文模型）。
- 输入：步骤2纠错后的 txt（多行为 Whisper 自动切分的短片段）。
- 目的：把碎片化的短句合并成**一个完整的讲解动作**（如"导入+提问""字词分析+追问"），便于后续切分检索单元。
- 方法：用大模型做"句子级合并"——把相邻几行合并为一句/一段自然的话，同时保留时间戳范围：
  ```
  输入（Whisper 自动切分）：
  312.40s → 315.20s | 同学们注意啊
  315.20s → 318.50s | 这个蹒跚两个字
  318.50s → 322.10s | 你们体会一下

  合并后：
  312.40s → 322.10s | 同学们注意啊，这个"蹒跚"两个字，你们体会一下。
  ```
- 合并粒度：让模型输出 200~500 字一段，大致对应一个"讲解动作"。太碎检索丢上下文，太粗检索不准。
- 注意：合并时**不要丢弃时间戳**，取该段首行的 start、末行的 end。

#### 步骤 4：结构化 & 标签
- 工具：`script/clean_transcription.py`（依赖DeepSeek上下文模型）。
- 目的：把每段话打上标签，便于后续 RAG 检索。
- 输入：步骤2合并后的段落 + 该课的元信息（课名、课本版本、年级）。
- 方法：用大模型对每段做轻量分类，打标签。Prompt 示例：
  ```
  对下面这段语文课讲解打标签，可选：导入/背景/字词/句析/提问/互动/情感/总结/其他。
  可多选，用逗号分隔。

  课文：《背影》
  片段：同学们注意啊，这个"蹒跚"两个字，你们体会一下父亲爬月台时的样子……
  → 字词,情感
  ```
- 输出：结构化 JSON，每段一个对象，包含 `lesson`、`segment_id`、`t_start`、`t_end`、`text`、`tags`。

#### 清洗后效果示例

清洗前（Whisper 原始）：
```
312.40s → 315.20s | 同学们注意啊，这个盘山两个字
315.20s → 318.50s | 你们体会一下
318.50s → 322.10s | 这个父亲爬月台的样子
322.10s → 326.40s | 是不是特别的
326.40s → 329.80s | 让人心酸
```

清洗后（纠错 + 合并 + 标签）：
```json
{
  "lesson": "背影",
  "segment_id": "beiying_0007",
  "t_start": 312.4,
  "t_end": 329.8,
  "text": "同学们注意啊，这个'蹒跚'两个字，你们体会一下——父亲爬月台买橘子那个样子，是不是特别让人心酸？",
  "tags": ["字词", "情感", "重点段"]
}
```

#### 工程化
- 步骤 2~4 可写成一个 pipeline 脚本 `clean_transcription.py` ，输入一节课的纠错后 txt，输出结构化 JSONL，放在chinese_clean目录下。
- 可选：纠错和合并用大模型的 Batch API 跑，比单条调用便宜 5~10 倍，1000 节课几块钱成本。

**依赖安装**：
```bash
# 无额外依赖，使用 Python 标准库 urllib 调用 API
```

### A.2 切分成检索单元

这一步的目标：把 A.1 产出的结构化段落，进一步切成**适合 RAG 检索的独立单元**。保证粒度不对太粗或太细。太碎→召回上下文不完整；太粗→不同话题混在一起、检索不精准。
- 工具：`script/split_units.py`（依赖DeepSeek上下文模型）。

#### 步骤 1：确定切分粒度
- 目标：每个单元 ≈ 一个"完整讲解动作"，例如"导入+抛问题""字词分析+追问""情感升华收尾"。
- 实操方法：看 A.1 标签里的 `tags` 字段，当标签发生**主题切换**时切一刀。例如：
  ```
  段落1: tags=["导入","提问"]     ← 保留，不切
  段落2: tags=["字词"]            ← 切一刀，新单元开始
  段落3: tags=["句析","情感"]     ← 同一单元，不切
  段落4: tags=["总结"]            ← 切一刀
  ```
- 量化标准：切完后每个单元 200~800 字，中位数 400 字左右。太短的单元可以和相邻同主题段合并；太长的（>1000字）按语义断点再切。

#### 步骤 2：生成 segment_id
- 格式建议：`{lesson_slug}_{序号}`，例如 `beiying_0007`。
- lesson_slug 用拼音或课名简写，保证全局唯一。
- 序号按该课的原始音频时间顺序编，便于后续定位回放。

#### 步骤 3：为每个单元补足元数据
- 在 A.1 的 `tags` 基础上，补充检索用的元数据：
  ```json
  {
    "segment_id": "beiying_0007",
    "lesson": "背影",
    "lesson_slug": "beiying",
    "grade": "八年级下",
    "textbook": "人教版",
    "t_start": 312.4,
    "t_end": 358.1,
    "text": "...",
    "tags": ["字词", "情感", "重点段"],
    "summary": "讲解'蹒跚'一词，结合父亲爬月台的动作细节，引导学生体会父爱。",
    "entities": ["朱自清", "背影", "蹒跚"],
    "source_refs": []
  }
  ```
- `summary`：用大模型对长文本做一句话摘要（50字内），用于快速浏览和粗筛。
- `entities`：从文本中抽实体（人名、地名、课文名、术语），便于按知识点过滤检索。
- `source_refs`：本单元讲到/引用的**课文原文原句**，对应到课文段落号。当前没有文章库时先留空；等文章库建好后，只从文章库或人工复核结果中写入，不让大模型凭记忆生成。作用有二：
  1. **检索锚点**——用户 query 常是"讲《背影》第二段""讲蹒跚那句"，挂上原句和段落号后能按原文精确召回/过滤（配合 B.2 步骤3 的 `where={"lesson":...,"para":...}`），比只靠讲解文本的语义匹配更准。
  2. **防幻觉素材**——生成时把权威原文一起喂给模型，避免它把课文背错、引错句子（呼应第 32 行的知识幻觉风险）。原文是"事实锚"，老师讲解是"风格锚"，两者分工。
  - 角色定位：`source_refs` 是**附加锚点**，不替代 `text`（讲解文本仍是主体）。
  - 抽取方式：先用 `entities`、老师讲解中的短引文、课文名和段落线索，在本地文章库里做精确/模糊匹配；匹配不到就留空。
  - **前提**：必须有可信文章库或人工复核结果。不要依赖 DeepSeek 等模型的训练记忆来补课文原句、段落号或出处。
  - 对齐粒度：老师是跳着讲、引片段，不逐句念，所以做到**段落级 + 引用句**即可，不追求逐句严格对齐；拿不准的留空，靠下方"人工复核"环节顺带补。

#### 步骤 4：把以上unit单元合并导出为 JSONL
- 由 `script/split_units.py` 实现
- 输入：`data/chinese_clean/<课程>/*.jsonl`（清洗后的段落）
- 输出：`data/chinese_units/<课程>/*.jsonl`（检索单元，每行一个 JSON 对象）
- 切分规则：tags 有交集则合并，无交集则切分；<200 字尝试合并，>1000 字在句号处再切

#### 切分效果示例

切分前（A.1 的原始段落，按时间顺序）：
```json
{"segment_id": "beiying_0005", "tags": ["导入","提问"], "text": "今天我们来学《背影》，同学们想想，你有没有一个印象特别深的、关于家人的背影？", ...}
{"segment_id": "beiying_0006", "tags": ["字词"], "text": "先看几个字词，'踌躇'——大家注意这个'躇'读 chú……", ...}
{"segment_id": "beiying_0007", "tags": ["字词","情感","重点段"], "text": "'蹒跚'两个字，你们体会一下，父亲爬月台那个样子……", ...}
{"segment_id": "beiying_0008", "tags": ["句析","情感"], "text": "'他用两手攀着上面，两脚再向上缩'——这一句，每一个动词都……", ...}
```

切分后（合并同主题段落，形成独立检索单元）：
```json
{
  "unit_id": "beiying_u001",
  "segment_ids": ["beiying_0005", "beiying_0006", "beiying_0007", "beiying_0008"],
  "lesson": "背影",
  "t_start": 120.0,
  "t_end": 358.1,
  "text": "今天我们来学《背影》，同学们想想，你有没有一个印象特别深的、关于家人的背影？先看几个字词，'踌躇'——大家注意这个'躇'读 chú……'蹒跚'两个字，你们体会一下，父亲爬月台那个样子……'他用两手攀着上面，两脚再向上缩'——这一句，每一个动词都……",
  "tags": ["导入", "字词", "情感", "重点段", "句析"],
  "summary": "导入课文，讲解'踌躇''蹒跚'等重点字词，结合父亲爬月台的动作细节分析父爱。",
  "entities": ["朱自清", "背影", "蹒跚", "踌躇", "月台"],
  "source_refs": []
}
```

#### 工程化
-   **输入**：`data/chinese_clean/<课程>/*.jsonl`（`clean_transcript.py` 的输出）
-   **输出**：`data/chinese_units/<课程>/*.jsonl`（检索单元）
-   脚本：`script/split_units.py`
-   `python3 script/split_units.py` — 处理全部；
-   `python3 script/split_units.py --file xxx.jsonl` — 单文件；
-   `python3 script/split_units.py --dry-run` — 仅切分不调模型；
-   `python3 script/split_units.py --sample-check 0.1` — 切分后抽 10% 打印摘要供人工复核。

**依赖安装**：
```bash
# 无额外依赖，使用 Python 标准库 urllib 调用 API
```

> `source_refs` 只能来自本地文章库匹配或人工复核。匹配置信度低、版本不确定、段落号不确定时一律留空。
> 后续可加一个"人工复核"环节：随机抽 10% 切分结果，检查是否有不当切点（如把一个完整提问从中间切断）；文章库上线后再顺带核对 `source_refs` 原句对应是否正确。

### A.3 风格画像（给 Prompt 和评估用）

#### 这一步在干什么
从全量语料里**离线**提炼出一份《老师风格档案》（一个结构化 JSON / Markdown）。它是对"真实老师怎么说话、怎么上课"的可复用描述，后面有三个用途：喂给阶段 B 的 System Prompt、当阶段 C 训练数据的指令模板、当评估时判断"像不像"的标尺。

关键原则：**能靠程序统计得到的就别让模型编，能靠真实语料归纳的就别凭感觉写。** 所以分两条腿：
- ①程序统计（客观、可量化）
- ②大模型归纳（主观、需人工校对）。

#### 第一条腿：程序统计（不调模型，纯代码跑）
这些维度直接从清洗后的语料用代码算，结果客观、还能复用为评估指标：

| 统计项 | 怎么算 | 产出 |
|--------|--------|------|
| 高频口头禅/语气词 | 分词后做 n-gram 词频，对比通用语料做差集（找出他**异常高频**的词） | "同学们注意啊""你们体会一下""是不是？" 及其频次 |
| 开场白/收尾句 | 取每节课/每段的首句、尾句聚类 | 典型开场、过渡、收尾句式列表 |
| 提问句式 | 正则/句法抽疑问句，统计句式模板 | "……为什么？""你觉得呢？"占比 |
| 句长/节奏 | 句子长度分布、停顿（用时间戳算语速） | 平均句长、长短句比例 |
| 互动密度 | 单位时长内提问/呼唤学生的次数 | 每分钟互动 N 次 |

这些数字本身就是阶段 6 评估里"风格保真度"的自动指标，一举两得。

> **已实现**：`script/style_stats.py`。
> - **输入**：`data/chinese_clean/<课程>/*.jsonl`（`clean_transcript.py` 的输出）
> - **输出**：`data/style/style_stats.json` 或 `--out stats.md`
> - **用法**：`python3 script/style_stats.py` | `--top-k 30` | `--out stats.md`。
> 
> **依赖安装**：
> ```bash
> pip3 install jieba
> ```

#### 第二条腿：大模型归纳（统计算不出的"套路"靠它）
讲解结构、举例偏好、情感表达这类"软"特征，统计抽不出来，用大模型读真实片段归纳。

**给模型什么（输入）：** 一批（如 20~50 段）真实讲解片段 + 第一条腿的统计结果 + 一个要求结构化输出的指令。

```
你是教学风格分析师。下面是某语文老师的 N 段真实课堂转录，
以及对他用词的统计结果。请归纳他的教学风格，严格按给定 JSON 结构输出，
每条结论必须能在片段里找到依据，不要编造。

【统计结果】
高频口头禅：同学们注意啊(312次)、你们体会一下(180次)...
平均句长：23字；每分钟互动：2.1次

【真实片段】
1. "同学们注意啊，这个'蹒跚'两个字……"
2. "我们来想一想，父亲为什么……"
... (20~50 段)

【输出 JSON 结构】
{ "口头禅": [...], "开场套路": "...", "讲解结构": [...],
  "提问方式": "...", "举例偏好": "...", "情感表达": "...", "禁忌(不会说的话)": [...] }
```

**得到什么（输出）：** 一份结构化风格档案，例如：

```json
{
  "persona": "亲切、重情感体验、爱追问、常拿生活小事打比方",
  "口头禅": ["同学们注意啊", "你们体会一下", "是不是这个道理"],
  "开场套路": "先抛一个和课文情感相关的生活化问题引入",
  "讲解结构": ["情感导入", "背景补充", "逐句精读", "连续追问", "情感升华收尾"],
  "提问方式": "高频反问+追问，少给标准答案，引导学生自己说",
  "举例偏好": "用学生日常生活、家庭场景类比",
  "情感表达": "讲到亲情段落语速放慢、反复强调关键词",
  "句式特征": "短句为主，平均23字，常用'啊/呢/吧'结尾",
  "禁忌": ["不堆术语", "不长篇背景灌输", "不直接报答案"]
}
```

#### 怎么保证质量
- 大模型归纳完，**人工校对一遍**（你或熟悉这位老师的人），删掉模型脑补的、补上漏掉的。
- 跑分批归纳再合并：50 段一批跑几批，结果取交集/高频项，比一次塞太多更稳。
- 这份档案是"活"的，阶段 B 上线后发现哪条风格没抓准，回来改档案即可，成本极低。

> **已实现**：`script/style_profile.py`（第二条腿）。
> - **输入**：`data/style/style_stats.json`（`style_stats.py` 的输出）+ `data/chinese_clean/<课程>/*.jsonl`
> - **输出**：`data/style/style_profile.json`（结构化风格档案）
> - **用法**：`python3 script/style_stats.py && python3 script/style_profile.py --batches 3 && python3 script/style_profile.py --merge` | `--samples 50` | `--dry-run`。
> 
> **依赖安装**：
> ```bash
> # 无额外依赖，使用 Python 标准库 urllib 调用 API
> ```

#### 产出物用途回顾
1. **给阶段 B Prompt**：整段塞进 System，告诉模型"你要这样说话"。
2. **给阶段 C 训练**：用"讲解结构""提问方式"等当 instruction 模板，构造训练对。
3. **用于评估**：第一条腿的统计项直接当自动指标（口头禅命中率、句长分布、互动密度对比真实留出集）。

---

## 4. 阶段 B：RAG + 风格 Prompt

### B.1 索引（离线，一次性跑完）

#### 这一步在干什么
当用户问"讲《背影》第二段"时，我们要从 A 里几万条片段中，快速找出**老师真正讲过的、和这个任务最相关的几段话**。问题是：用关键词匹配（如 MySQL `LIKE`、Elasticsearch 全文）只能匹配字面词，匹配不了语义——比如用户问"父爱的细节描写"，老师原话是"父亲爬月台买橘子那段"，字面不重合，关键词搜不到，但语义高度相关。

**Embedding（向量化）就是解决这个问题的：** 把每段文本用一个模型转成一串浮点数（比如 1024 维的向量），语义相近的文本，向量在空间里距离也近。检索时把用户的 query 也转成向量，去向量库里找**距离最近的 top-k 段**，就能召回语义相关的真实讲解。

"做索引"=**离线地**把 A 的所有检索单元算成向量、连同原文和元数据一起存进向量库，建立可快速最近邻搜索的结构。这是一次性（或增量）的预处理，不是每次请求都做。

#### 步骤 1：选择 Embedding 模型
- 选 `BAAI/bge-large-zh-v1.5`（中文语义检索常用，1024 维，纯中文场景效果好）。
- 备选 `bge-m3`（多语言、支持长文本、能同时出稠密/稀疏向量，片段较长或想做混合检索时更稳）。
- **注意：建库和查询必须用同一个 embedding 模型**，换模型要全量重建索引。

#### 步骤 2：选择向量库
- 数据量不大（几万~几十万段）：**Chroma**（本地零运维，最快上手）。

#### 步骤 3：批量向量化入库

**向量索引用 `text`（讲课全文）。**
精确维度（课文/段落）靠元数据过滤，语义维度（概念/风格/讲法）靠 `text` 向量——两者分工。

- 读 A.2 JSONL，编码 `text`。
- metadata 里保存 lesson + tags + summary；文章库建好并回填 `source_refs` 后，再额外保存 para（从 `source_refs` 提取的段落号），用于精确过滤。

**已实现**：`script/build_index.py`（调度 `bge/encode.py` 的 Encoder）。

**输入**：
- `data/chinese_units/<课程>/*.jsonl`（A.2 检索单元，每行一个 JSON 对象）
- 每个 JSON 包含：`unit_id, text, lesson, tags, summary, source_refs, entities`

**输出**：
- `data/vecdb/`（ChromaDB 持久化向量库）
- Collection 名称：`teacher_units`
- 字段：
  - `documents`：text（讲课原文，前面拼接了 entities + tags + summary 增强关键词密度）
  - `embeddings`：bge-large-zh-v1.5 1024 维向量
  - `metadatas`：lesson, t_start, tags, summary, paras（段落号，来自 source_refs）

**用法**：
```bash
python3 script/build_index.py              # 增量索引（跳过已入库 unit_id）
python3 script/build_index.py --rebuild     # 清空向量库后重建
python3 script/build_index.py --dry-run     # 仅检查，不实际入库
python3 script/build_index.py --model-dir ./bge/models  # 指定模型目录
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--src` | `data/chinese_units` | A.2 检索单元 JSONL 目录 |
| `--db` | `data/vecdb` | ChromaDB 持久化路径 |
| `--model-dir` | `bge/models` | BGE 模型缓存目录 |
| `--batch-size` | 64 | 编码 batch size |
| `--rebuild` | - | 清空向量库后重建 |
| `--dry-run` | - | 仅检查，不实际入库 |

**依赖安装**：
```bash
pip3 install chromadb sentence-transformers huggingface-hub
```

#### 步骤 4：索引测试
构建索引后，用 `script/build_index_test.py` 验证入库效果：

```bash
# 完整测试（连接、元数据、语义检索）
python3 script/build_index_test.py

# 指定查询语句
python3 script/build_index_test.py --query "朱自清的父爱"

# 返回更多结果
python3 script/build_index_test.py --top-k 10

# 按课文过滤
python3 script/build_index_test.py --lesson "背影"
```

**测试项**：
1. **连接测试**：集合状态、总单元数
2. **元数据完整性**：检查 lesson/tags/summary/paras 是否填充
3. **语义检索**：给定 query 返回 top-k 结果及相似度
4. **过滤检索**：按 lesson 元数据过滤

#### 步骤 5：加 reranker（精排模型）
向量召回（top-20）是"粗筛"，速度快但有时把不够相关的也召回了。再用 **`bge-reranker-v2-m3`** 这种交叉编码器，对 query 和每个候选段两两打分，重排后取真正最相关的 top-N（如 N=3~5）塞进 Prompt：

"粗召回(向量, top-20) → 精排(reranker, top-5)"是 RAG 的标准两段式，能明显提升喂给 LLM 的上下文质量。

**已实现**：`script/search.py` 支持 `--rerank` 启用两段式检索。首次运行会下载 `bge-reranker-v2-m3`（~2.3GB）。 
> search.py为测试效果使用，不参与工程化。

#### 步骤 6：增量更新
- 新转录的课随时 `col.add` 增量入库，不用全量重跑。
- 只有在**换 embedding 模型**或**改切分粒度**时才需要全量重建索引。

**已实现**：`script/build_index.py` 默认增量模式（跳过已入库 unit_id）。换模型用 `--rebuild`。

### B.2 在线流程（用户请求时跑）

用户发起一个请求（如"讲《背影》第二段"），系统按以下步骤实时响应：

#### 步骤 1：解析用户意图
- 从用户输入中提取：**课文名**（如"背影"）、**范围**（如"第二段"）、**意图**（如"精读讲解"/"概括大意"/"考试点拨"）。
- 方法：用大模型做一次轻量意图解析，或用规则匹配（如输入含"第X段""精读""概括"等关键词）。
- 输出结构：`{"lesson": "背影", "scope": "第二段", "intent": "精读讲解"}`

#### 步骤 2：构建检索 query
- 把解析后的意图转成一段自然语言 query，用于向量检索。
- bge 在 query 侧建议加指令前缀，召回更准：
  ```python
  q = "为这句话设计课堂讲解：" + user_query
  q_emb = model.encode([q], normalize_embeddings=True).tolist()
  ```

#### 步骤 3：向量检索 + rerank
- 用 query 向量去向量库检索 top-20 候选段（可加元数据过滤，如 `where={"lesson": "背影"}` 优先召回同一课文的片段）。
- reranker 精排后取 top-3~5 段。

#### 步骤 3.5：课文硬事实（过渡方案）
- 为了先把项目跑起来，文章库还没建好时，临时调一次大模型获取指定课文的**原文、作者、朝代/出处、段落范围、白话翻译和重点字词释义**，形成"逐句精讲骨架"。
- 输出结构：`{"source_type":"llm_temporary","author":"","dynasty":"","source":"","excerpt":"","sentences":[{"text":"","translation":"","keywords":[{"word":"","note":""}]}],"synopsis":"","keyPoints":[]}`
- 这批内容只作为**临时事实源**：可以用于生成讲解，但不能回写进 `source_refs`、向量库元数据或训练数据；也不能当作最终权威库。
- Prompt 中要明确：若大模型返回的原文/段落/出处不确定，必须标记为"待核验"，不要把不确定信息说成定论。
- 等本地文章库建好后，把这里替换成文章库查询；文章库返回的原文、段落、作者、出处、注释、译文才作为不可违背的权威事实上下文注入 Prompt。

#### 步骤 4：组装 Prompt
- 模板结构：
  ```
  [System] 你是XXX老师的课堂克隆。风格档案：{A.3 产出的风格档案 JSON}
  临时课文事实：{大模型返回的原文/作者/出处/逐句骨架；source_type=llm_temporary，待文章库上线后替换}

  [Few-shot] 以下是这位老师讲过的真实片段（仅供参考风格，不要照抄）：
  ---
  片段1: {检索到的段落1}
  片段2: {检索到的段落2}
  片段3: {检索到的段落3}
  ---

  [Task] 用户请求：{用户原始输入}
  课文：{lesson}，范围：{scope}，意图：{intent}

  [输出格式]
  【课文信息】可列出临时事实源提供的作者、出处、原文范围；不确定项标记"待核验"。
  【精讲正文】按临时逐句骨架讲解：引原文 → 解读重点词 → 串讲情感/写法。

  [要求]
  1. 用第一人称"我"，课堂口吻。
  2. 体现风格档案中的口头禅、句式、提问方式。
  3. 可参考 few-shot 片段的讲解结构，但不要照搬。
  4. **不限字数**；篇幅短则逐句讲，篇幅长则把连贯的几句合成一段逐段讲。
  5. 临时事实源里没有的字词读音/释义/出处，宁可不展开也不编。
  6. 不要把临时事实源写入长期数据；文章库上线后必须用文章库结果替换。
  7. 结尾可用提问或互动引导。
  ```

#### 步骤 5：调用 LLM 生成
- 把组装好的 Prompt 发给底座模型（Qwen / DeepSeek 等国产长上下文模型）。
- 流式返回生成文本，提升课堂体验。

#### 在线流程效果示例

输入：`讲《背影》第二段`

解析结果：`{"lesson": "背影", "scope": "第二段", "intent": "精读讲解"}`

检索到的 few-shot 片段（示例）：
```
片段1: 同学们注意啊，这个"蹒跚"两个字，你们体会一下——父亲爬月台买橘子那个样子，是不是特别让人心酸？
片段2: 我们来看"他用两手攀着上面，两脚再向上缩"——这一句，每一个动词都不要放过，作者为什么要写得这么细？
片段3: 同学们，你们有没有过类似的经历？就是明明知道家人在身后，但你没有回头……
```

输出（模型生成，风格贴近）：
```
好，我们来看《背影》第二段。同学们注意啊，这一段写的是父亲送"我"到浦口车站，买橘子的那个场景——你们体会一下，作者为什么要把父亲的背影写得这么细？"他用两手攀着上面，两脚再向上缩"，每一个动词都不要放过。你们想想，父亲那时候多大年纪了？身体又不好，还坚持去买橘子——是不是特别让人心酸？
```

**已实现**：`script/generate.py`。端到端流程：意图解析（规则+模型）→ 向量检索（+可选reranker）→ 风格Prompt组装 → LLM生成。

**用法**：
```bash
# 基本用法
python3 script/generate.py "讲德行篇第25则"

# 详细模式（输出检索结果、完整 Prompt）
python3 script/generate.py "讲德行篇第25则" -v

# 指定参数
python3 script/generate.py "讲德行篇第25则" --rerank --lesson "世说新语精读" --top 10
```

**详细模式（`-v`）输出内容**：
1. **意图解析结果**：lesson、scope、intent
2. **课文事实**（如有）：author、dynasty、source、excerpt、sentences 等
3. **ChromaDB 查询详情**：
   - 集合名、总单元数
   - 查询文本、过滤条件
   - 召回结果列表（unit_id、lesson、tags、summary、paras、相似度）
4. **Rerank 精排结果**（如启用）：rerank 分数
5. **最终检索结果**：top-N 段的完整信息
6. **SYSTEM PROMPT（完整）**：风格设定 + 课文硬事实
7. **USER PROMPT（完整）**：few-shot 片段 + 任务指令
8. **生成结果**：模型输出

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `query` | 必填 | 用户请求，如"讲《背影》第二段" |
| `--lesson` | 自动解析 | 手动指定课文名 |
| `--top` | 5 | 检索段数 |
| `--rerank` | - | 启用 bge-reranker 精排 |
| `--model-dir` | bge/models | BGE 模型目录 |
| `-v` / `--verbose` | - | 详细输出模式 |

> generate.py为效果测试代码，不参与工程化。

**依赖安装**：
```bash
# 无额外依赖，复用 B.1 的 chromadb + sentence-transformers
# DeepSeek API 调用使用 Python 标准库 urllib
```

### B.3 Prompt 结构（要点）
- System：角色设定 + 风格档案（口头禅/句式/讲解套路）。
- Few-shot：**用检索到的真实片段**当示例，而不是手写假例子——这是和方案 2 的本质区别。
- 约束：长度、结构、必须用第一人称课堂口吻、可引用但不照抄。

底座模型选国产长上下文的（Qwen / DeepSeek 系），中文课堂语感更好。

---

## 5. 阶段 C：LoRA 风格微调

### C.1 训练数据构造（核心难点）

这一步的目标：把 A 产出的口语独白，转成大模型能训练的 `instruction → output` 对。**只用真实 output 当监督信号**，instruction 可以合成。切忌用别的模型生成的"假讲解"当 output，会污染风格。

#### 方式 1：反向生成指令（推荐，最常用）
- 输入：A.2 产出的一段真实讲解文本（已清洗、已合并）。
- 方法：用大模型反推"老师当时在讲什么任务"，生成 instruction。
- Prompt 示例：
  ```
  你是一个教学数据标注员。下面是一段语文老师的课堂讲解实录。
  请推断：这位老师当时在做什么教学任务？用一句话描述。
  输出格式：instruction: "用课堂讲解的方式XXX"

  【讲解实录】
  同学们注意啊，这个"蹒跚"两个字，你们体会一下——父亲爬月台买橘子那个样子，是不是特别让人心酸？我们来看"他用两手攀着上面，两脚再向上缩"——这一句，每一个动词都不要放过。

  → instruction: "用课堂讲解的方式分析《背影》中父亲买橘子一段的字词和情感"
  ```
- 输出：
  ```json
  {
    "instruction": "用课堂讲解的方式分析《背影》中父亲买橘子一段的字词和情感",
    "output": "同学们注意啊，这个"蹒跚"两个字，你们体会一下——父亲爬月台买橘子那个样子，是不是特别让人心酸？我们来看"他用两手攀着上面，两脚再向上缩"——这一句，每一个动词都不要放过。"
  }
  ```
- 数据量：1000 节课 → 约 2~5 万条检索单元 → 构造 2~5 万条训练对。对 LoRA 来说绰绰有余（通常几千到几万条就够）。

#### 方式 2：续写式
- 输入：一段真实讲解的前半部分。
- 方法：让模型学着按老师的风格，接下半段。
- 构造方式：把每段讲解从中间切一刀，前半段当 instruction（加前缀"请接着讲："），后半段当 output。
- 适用场景：训练模型的"连贯性"和"节奏感"。

#### 方式 3：风格改写式
- 输入：一段**中性化**的课堂讲解（由大模型生成的"标准教师腔"）+ 风格指令。
- instruction："用这位老师的风格重讲下面这段内容"。
- output：**必须用 A.2 的真实片段**，不是模型生成的。
- 适用场景：显式训练"风格迁移"能力。

#### 数据质量兜底
- 随机抽 10% 训练对人工检查：instruction 是否和 output 语义匹配？output 是否来自真实语料？
- 去重：同一段讲解不要出现多次（instruction 稍微改了也算重复）。
- 长度分布：确保训练数据覆盖短段（100字）到长段（800字），不要全是最长的。

### C.2 训练

#### 步骤 1：环境准备
- 工具：**LLaMA-Factory**（社区最成熟、配置最简单的 LoRA 训练框架）。
- 硬件：单卡 4090（24GB）或 A100（40GB）即可。1000 节课的数据量训练成本很低。

#### 步骤 2：选底座模型
- 推荐：`Qwen2.5-7B-Instruct`（起步）或 `Qwen2.5-14B-Instruct`（效果更好，看显存）。
- 为什么选 Qwen：中文课堂语感好、社区活跃、LLaMA-Factory 原生支持。

#### 步骤 3：配置 LoRA 训练
- 用 LLaMA-Factory 的 `ds_config.json` 配置 QLoRA（4-bit 量化），省显存。
- 关键参数：
  - `lora_rank`：16~32（起步用 16，够了）
  - `lora_alpha`：32（alpha = 2 × rank 是常用比例）
  - `target_modules`：`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`（全注意力层 + FFN）
  - `epochs`：3~5
  - `batch_size`：2（QLoRA 显存有限）
  - `learning_rate`：2e-5

#### 步骤 4：跑训练
```bash
llamafactory-cli train qwen2.5-7b-instruct.yaml
```
- 训练数据格式：JSON，每条 `{"instruction": "...", "output": "..."}`。
- 训练时长：7B 模型 + 3 万条数据，约 2~4 小时（4090）。

### C.3 上线形态

#### 步骤 1：合并权重
- LLaMA-Factory 训练完会产出 LoRA adapter 权重文件。
- 用 `llamafactory-cli export` 把 adapter 合并回底座模型，得到一个完整的 GGUF 或 HuggingFace 格式模型。

#### 步骤 2：部署推理
- 用 **vLLM** 部署合并后的模型。
- LoRA 权重热加载：如果不想合并，vLLM 也支持运行时加载 LoRA adapter，更灵活。

#### 步骤 3：RAG 不变
- LoRA 只负责"**怎么说**"（风格），RAG 继续负责"**说什么**"（知识）。
- 在线流程和阶段 B 完全一样：检索 → 组装 Prompt → LLM 生成。只是 LLM 从底座模型换成了微调后的模型。

#### 上线效果对比

| 维度 | 阶段 B（纯 Prompt） | 阶段 C（LoRA + RAG） |
|------|-------------------|---------------------|
| 风格保真度 | 依赖 Prompt 质量，长输出容易漂 | 风格"焊进"模型，稳定 |
| 内容准确性 | RAG 提供，一样 | RAG 提供，一样 |
| 迭代成本 | 改 Prompt 即可 | 需重新训练 |
| 推理成本 | 低（用 API） | 中（需本地 GPU 推理） |

---

## 6. 评估（贯穿 B/C，没有评估就是瞎调）

不能只靠"感觉像"。建一个评估集，用数据说话。

### 6.1 构建评估集

#### 步骤 1：留出真实片段
- 从 A.2 的检索单元中，随机抽 30~50 段老师真实讲解。
- **重要：这些片段不能用于训练（阶段 C）和检索（阶段 B 的 RAG），只能用于评估。**
- 覆盖不同课文、不同讲解类型（导入/字词/句析/情感/总结）。

#### 步骤 2：构造评估任务
- 把每段真实片段"改写"成用户请求：提取课文名、范围、意图，组成 query。
- 例如：真实片段是"讲解《背影》第二段的父亲动作细节" → 评估 query 是"讲《背影》第二段"。

#### 步骤 3：让模型生成
- 用评估 query 分别跑阶段 B 和阶段 C 的系统，得到模型输出。

### 6.2 自动指标

| 指标 | 怎么算 | 目标 |
|------|--------|------|
| 口头禅命中率 | 模型输出中包含风格档案里高频口头禅的比例 | 接近真实语料均值，不低于真实均值的 70% |
| 口头禅过度使用率 | 每 100 字口头禅次数 vs 真实语料均值 | 不超过真实均值的 1.5 倍 |
| 句长分布相似度 | 模型输出的句长分布 vs 真实语料的 KL 散度 | 越小越好 |
| 互动密度 | 模型输出中每 100 字的提问/反问句数量 | 接近真实均值 |
| 检索命中率@k | 评估 query 对应的真实留出片段/同课同主题片段是否出现在 top-k 候选中 | top-5 ≥70%，top-10 ≥85% |
| 范围准确率 | 用户指定课文/段落/篇章范围是否被正确解析并用于过滤 | ≥90% |
| 临时事实可用率 | 文章库上线前，大模型返回的原文/段落/释义经人工抽检后可直接用于讲解的比例 | ≥80%，低于该值优先建文章库 |
| 事实引用正确率 | 输出中出现的作者、出处、原文、段落号、字词释义是否能被临时事实源、文章库或人工标注验证 | 文章库上线前人工抽检；上线后 ≥95% |
| 事实幻觉率 | 输出中无法被检索片段、用户输入、临时事实源或文章库支撑的硬事实占比 | ≤5% |
| 语义覆盖度 | 模型输出是否覆盖真实片段里的核心讲解点，而不是只在 embedding 上相似 | 人工/LLM-as-judge 打分 ≥4/5 |
| 复述/照抄率 | 输出与检索片段的长 n-gram 重合比例 | 不长段照抄，连续重合超过 50 字需人工检查 |

**已实现**：`script/evaluate.py`。用 `data/chinese_units` 真实讲解数据作为 ground truth 自动评估。

**用法**：
```bash
# 抽样评估（从真实讲解随机抽 N 个，自评作为 baseline）
python3 script/evaluate.py --sample 10

# 指定课程评估
python3 script/evaluate.py --sample 20 --course "世说新语精读"

# 对比评估（传入模型输出，与随机真实讲解对比）
python3 script/evaluate.py --compare "模型生成的讲解文本..." --course "世说新语精读"

# 启用 LLM-as-Judge（需要 API key）
python3 script/evaluate.py --sample 10 --llm-judge

# 输出 JSON 格式
python3 script/evaluate.py --sample 10 --json
```

**自动指标**：

| 指标 | 说明 | 目标 |
|------|------|------|
| 长度比 | 模型输出长度 / 真实讲解长度 | 接近 1.0 |
| 字符级 Jaccard | 两段文本的字符集合相似度 | 越高越好 |
| 词级 Jaccard | 两段文本的词集合相似度 | 越高越好 |
| 2-gram 重叠 | 2 字符 n-gram 重叠率 | 越高越好 |
| 口头禅命中率 | 模型输出中包含风格档案里高频口头禅的比例 | ≥70% |
| 口头禅过度使用率 | 每 100 字口头禅次数 vs 真实语料均值 | ≤1.5x |
| 问句密度 | 模型输出 vs 真实讲解的问句数量对比 | 接近真实均值 |
| LLM-as-Judge | 大模型评判综合评分（需 API key） | ≥4.0/5 |

### 6.3 人工盲评（最关键）

#### 步骤 1：准备盲测评分表
- 把模型输出和真实片段混在一起，不标注来源。
- 每个样本请 2~3 位评判者（你/熟悉这位老师的人）评分。

#### 步骤 2：评分维度
- **风格相似度**（1~5 分）：语气、口头禅、提问方式是否像这位老师。
- **内容质量**（1~5 分）：讲解是否有价值、是否准确。
- **整体印象**（1~5 分）：综合打分。
- **猜测来源**：判断这段是"真人讲的"还是"模型生成的"。

#### 步骤 3：达标标准
- **门槛**："像他"的猜测准确率 ≥70%（即评判者猜对模型输出来源的比例 ≤30%），认为风格保真度达标。
- 不达标 → 进入阶段 C（LoRA 微调）。

### 6.4 评估节奏
- 阶段 B 上线前：跑一次完整评估，决定是否需要进入阶段 C。
- 阶段 C 训练后：再跑一次评估，对比 B 和 C 的效果差异。
- 上线后：每月抽样跑一次，监控风格漂移。

---

## 7. 端到端生成流程

从用户输入到最终产出（PPT/HTML/视频），按执行顺序组织。

### 7.1 整体流程

```
用户输入"木兰词 第一段"
      ↓
[server/app.py] POST /api/generate
      ├── 输入过长且没有范围 → 返回 need_scope
      ├── 创建匿名任务 task_id + token
      └── 写入 data/output/tasks.sqlite3
      ↓
[script/generate.py] 意图解析 + 语料检索 + 生成讲解词
      ├── parse_intent() - 提取课文名、范围、意图
      ├── fetch_lesson_facts() - 获取作者/朝代/句子
      └── 输出 script.json（含 sentences[].text/keywords/narration）
      ↓
[script/build_timeline.py] 按句生成时间轴 + TTS
      ├── 逐句调 DeepSeek 生成 narration（带上下文过渡）
      ├── edge-tts 合成音频 → audio/01.mp3 ...
      └── 输出 timeline.json（含 audio/duration/keyword_timings）
      ↓
[script/build_pptx.py] 生成 PPT
      ├── 读取 timeline.json + 音频
      ├── 每页嵌入旁白 + 自动播放 timing
      └── 输出 {lesson}.pptx
      ↓
[script/build_player.py] 生成 HTML 预览
      └── 输出 player.html
      ↓
前端轮询 /api/task/{task_id}?token=...
      ├── 完成后 iframe 展示 /api/preview/{task_id}?token=...
      └── 下载 /api/download/{task_id}?token=...
```

### 7.2 generate.py — 讲解词生成

复用 B.2 的完整流程：意图解析 → 课文事实获取 → 语料检索 → Prompt 组装 → LLM 生成。

**用法**：
```bash
python3 script/generate.py "讲德行篇第25则"
python3 script/generate.py "讲德行篇第25则" -v  # 详细输出
```

**输出**：`data/timelines/{课程拼音}/{章节拼音}/YYYYMMDD_HHMMSS.json`

```json
{
  "title": "《世说新语·德行篇》第25则",
  "sentences": [
    {
      "text": "华歆、王朗俱乘船避难。",
      "keywords": [{"word": "避难", "note": "躲避灾祸"}],
      "narration": "好，我们来看第一句..."
    }
  ],
  "intro": "同学们，今天我们要学的是《世说新语·德行篇》第25则..."
}
```

### 7.3 build_timeline.py — 时间轴生成

所有渲染输出的核心入口。输入查询，输出 `timeline.json` + 音频文件。

**用法**：
```bash
python3 script/build_timeline.py "诫子书 全文"
python3 script/build_timeline.py "诫子书" --max-sentences 3 --verbose
python3 script/build_timeline.py "诫子书" --skip-tts  # 只生成讲稿
```

**执行流程**：
```
调用 generate.py 获取讲解词
  → 逐句生成/补充 narration（带全文上下文 + 上一句结尾过渡）
  → edge-tts 合成 mp3（audio/01.mp3 …）
  → mutagen 读时长 → 写入 duration
  → 线性估时对齐 → 写入 keyword_timings
  → 输出 data/timelines/{课程拼音}/{章节拼音}/YYYYMMDD_HHMMSS/
```

**输出目录结构**：
```
data/timelines/{课程拼音}/{章节拼音}/YYYYMMDD_HHMMSS/
├── script.json       # 纯文本讲稿（含 intro）
├── timeline.json     # 完整时间轴（含音频路径、时长）
├── audio/
│   ├── 00_intro.mp3  # 导入语音频
│   ├── 01.mp3        # 第1句音频
│   └── ...
└── slides.pptx       # 由 build_pptx.py 生成
```

**timeline.json 结构**：
```json
{
  "title": "《诫子书》精讲",
  "intro": "同学们...",
  "sentences": [
    {
      "id": 1,
      "text": "非淡泊无以明志，非宁静无以致远。",
      "keywords": [{"word": "淡泊", "note": "恬淡寡欲"}],
      "narration": "好，我们来看第一句...",
      "audio": "audio/01.mp3",
      "duration": 15.5,
      "keyword_timings": [{"word": "淡泊", "start": 3.4}]
    }
  ]
}
```

**依赖**：
```bash
pip3 install edge-tts mutagen pypinyin
```

### 7.4 build_pptx.py — PPT 生成

`python-pptx` 逐句生成幻灯片，**已实现嵌入旁白 + 自动播放**。

**用法**：
```bash
python3 script/build_pptx.py "诫子书 全文"
python3 script/build_pptx.py "诫子书" --out 诫子书.pptx
```

**特性**：
- 封面页 + 逐句内容页（原文大字 + 关键词卡片 + 译文）
- 封面页备注放入导入语（intro），音频时长动态匹配
- 每页嵌入对应句的 TTS 音频
- 自动播放 timing：切到该页即播放，时长 = 旁白时长
- 用 PowerPoint 打开后「文件→导出→创建视频」即可得到带声 MP4

**依赖**：
```bash
pip3 install python-pptx
```

### 7.5 build_player.py — HTML 播放器

一个 HTML 文件，加载 `timeline.json`，CSS + JS 驱动高亮。

**用法**：
```bash
python3 script/build_player.py "诫子书 全文"
```

**特性**：
- 逐页卡片式：每页一句（原文 + 关键词卡片 + 译文）
- 音频播放器：加载当前句 TTS 音频，播完自动跳下一句
- 键盘控制：← → 切句，空格播放/暂停
- 进度条：显示当前句播放进度

**依赖**：无（纯 HTML + JS）

---

## 8. Web 服务架构

### 8.1 最小架构

```
Client（一个输入框）
  ↓
API (FastAPI: server/app.py)
  ├─ 输入长度/范围校验：太长则返回 need_scope
  ├─ 匿名任务创建：task_id + token + client_id
  └─ 后台生成任务（SQLite 持久化状态）
        ↓
      build_timeline.py 生成 timeline + 音频
        ↓
      build_pptx.py 生成 PPTX
        ↓
      build_player.py 生成 HTML 预览
        ↓
      页面轮询任务状态 → 展示 HTML 预览 + 下载 PPT
```

### 8.2 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| API 框架 | FastAPI | Python 生态、自动文档、异步支持 |
| 匿名任务 | SQLite | MVP 阶段持久化任务状态，页面关闭后可恢复 |
| 缓存 | Redis（可选） | 相同 query 直接返回缓存结果 |

### 8.3 API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/generate` | POST | 创建生成任务，返回 task_id + token |
| `/api/task/{task_id}?token=...` | GET | 查询任务状态 |
| `/api/preview/{task_id}?token=...` | GET | 返回 HTML 播放器 |
| `/api/download/{task_id}?token=...` | GET | 下载 PPTX |
| `/task/{task_id}?token=...` | GET | 恢复链接入口 |

### 8.4 匿名用户与任务恢复

用户不登录时，用两层机制恢复任务：

1. 前端第一次打开页面时生成 `client_id`，保存到 `localStorage`。
2. 每次生成任务时，后端生成 `task_id + token`，写入 `data/output/tasks.sqlite3`。
3. 页面关闭后再次打开，前端从 `localStorage` 读取最近任务并轮询状态。
4. 换浏览器或清缓存时，用户可通过 `/task/{task_id}?token=...` 恢复。

`task_id` 只用于定位任务，`token` 才是访问凭证；下载和预览都必须带 token。

### 8.5 项目结构

```
teacher-clone/
├── server/
│   ├── app.py              # FastAPI 入口
│   ├── requirements.txt    # Web 服务依赖
│   └── static/
│       └── index.html      # 单输入框页面
├── script/                 # 所有脚本
├── data/
│   ├── output/
│   │   └── tasks.sqlite3  # 匿名任务状态库
│   ├── timelines/         # timeline/audio/pptx 输出
│   ├── style/             # 风格档案
│   └── vecdb/             # Chroma 向量库
└── config/
    └── config.toml        # API Key、模型配置
```

### 8.6 扩展路径

- SQLite 适合单机 MVP；多实例部署时改为 Redis/Postgres，并把生成文件放对象存储。
- 后续要多老师/多租户、计费、Marketplace 再演进。

---

## 9. 部署环境

### 9.1 系统要求

| 项目 | 最低 | 建议 |
|------|------|------|
| **Python** | 3.11+ | 3.11 |
| **系统包** | `ffmpeg` | |
| **内存** | 8GB | 16GB+ |
| **磁盘** | 20GB | 50GB+（含模型 ~1.3GB） |
| **GPU** | 不需要 | |

### 9.2 依赖安装

```bash
# 系统包
apt install ffmpeg

# Python 环境
python3.11 -m venv venv && source venv/bin/activate

# 项目依赖
pip3 install -r requirements.txt
pip3 install -r server/requirements.txt

# 可选：TTS 和 PPT
pip3 install edge-tts mutagen python-pptx pypinyin
```

### 9.3 模型文件

| 模型 | 位置 | 大小 | 下载方式 |
|---|---|---|---|
| BGE Embedding | `bge/models/` | ~1.3GB | `python3 bge/download_model.py --download` |

### 9.4 外部服务

- **DeepSeek API**：需要 `api_key`，配置文件 `script/config.toml`
- **edge-tts**：微软云端 TTS，免费，需联网
- BGE 向量检索本地运行

### 9.5 部署清单

```bash
# 1. 系统包
apt install ffmpeg

# 2. Python 环境
python3.11 -m venv venv && source venv/bin/activate

# 3. 项目依赖
pip3 install -r requirements.txt
pip3 install -r server/requirements.txt

# 4. 下载 BGE 模型
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-zh-v1.5')"

# 5. 配置 DeepSeek API Key
cp script/config.example.toml script/config.toml
# 编辑 config.toml 填入 api_key

# 6. 验证
python3 script/build_timeline.py "诫子书 全文" --skip-tts
python3 script/build_timeline.py "诫子书 全文"
python3 script/build_pptx.py "诫子书 全文"

# 7. 启动 Web 服务
python3 -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 9.6 服务器硬件要求

在线服务只涉及 BGE 向量编码 + API 调用 + 文件生成，不需要 GPU：

| 配置 | 最低要求 | 说明 |
|---|---|---|
| CPU | 2 核 | BGE 编码 + API 调用 |
| 内存 | 4GB | BGE 模型加载约 1.5GB，留余量 |
| 硬盘 | 20GB | vecdb + 依赖 + 生成文件 |
| 网络 | 能出网 | 调 DeepSeek API、下载模型、Edge-TTS |
| 系统 | Linux | 不建议 Windows |

### 9.7 本地 vs 服务器文件

| 文件 | 需要上传到服务器 | 说明 |
|---|---|---|
| `script/` | ✅ | 脚本代码 |
| `server/` | ✅ | Web 服务和静态页面 |
| `data/vecdb/` | ✅ | 本地跑出来的向量库 |
| `data/style_profile.json` | ✅ | 风格画像（如有） |
| `config.toml` | ✅ | API Key 配置 |
| `data/output/tasks.sqlite3` | ❌ | 运行时任务库，服务器自动创建 |
| `data/timelines/` | 视情况 | 生成结果目录 |
| `asr/models/` | ❌ | ASR 离线用，线上不需要 |
| `bge/models/` | ❌ | 服务器上直接下载即可 |

---

## 10. 脚本清单

| 脚本 | 用途 | 阶段 |
|---|---|---|
| `script/generate.py` | 讲解词生成：意图解析→检索→Prompt→LLM | B.2 |
| `script/build_timeline.py` | 时间轴生成：按句生成+TTS+对齐 | 7.3 |
| `script/build_pptx.py` | PPT 生成（含嵌入旁白+自动播放） | 7.4 |
| `script/build_player.py` | HTML 播放器生成 | 7.5 |
| `script/search.py` | 语义搜索测试 | B.1 |
| `script/build_index.py` | 向量索引构建 | B.1 |
| `script/build_index_test.py` | 索引测试 | B.1 |
| `script/split_units.py` | 检索单元切分 | A.2 |
| `script/style_stats.py` | 风格统计 | A.3 |
| `script/style_profile.py` | 风格画像 | A.3 |
| `script/evaluate.py` | 评估：模型 vs 真实讲解 | 6 |
| `asr/transcribe.py` | 批量转录 | A.1 |

---

## 11. 离线训练流程

从原始音频构建语料库：

```
原始音频文件 (.mp3)
      ↓
[asr/transcribe.py] faster-whisper 转录 → 文本+时间戳
      ↓
[script/split_units.py] 拆分讲解单元（句子级）
      ↓
[script/build_index.py] 构建向量索引（chromadb）
      ↓
[script/style_stats.py] + [script/style_profile.py] → 风格档案
```
