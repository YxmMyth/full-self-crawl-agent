# TechCrunch 端到端测试分析报告

**生成时间**: 2026-02-27
**测试站点**: site_02_techcrunch
**URL**: https://techcrunch.com/

---

## 执行摘要

| 指标 | 值 |
|------|-----|
| 最终状态 | 失败 |
| 提取数量 | 16 条 |
| 质量分数 | 0.679 (67.9%) |
| 迭代次数 | 3 次 (达到上限) |
| 执行时间 | 579 秒 (约 9.6 分钟) |
| 成功字段 | `published_time` |
| 失败字段 | `title`, `summary`, `author`, `image_url`, `article_url`, `category` |

---

## 迭代循环详细分析

### 迭代循环架构

主循环在 `src/main.py:170-476` 实现，遵循七阶段流程：

```
for iteration in range(max_iterations):  # max_iterations=3
    ├── [1/6] Sense   - 感知页面结构
    ├── [2/6] Plan    - 规划提取策略
    ├── [3/6] Act     - 执行数据提取
    ├── [4/6] Verify  - 验证数据质量
    ├── [5/6] Gate    - 门禁检查 (质量阈值 0.7)
    ├── [6/6] Judge   - 决策下一步
    └── [反思] Reflect - 仅当 decision='reflect_and_retry' 时执行
```

---

## 各智能体行为详解

### 1. SenseAgent (感知智能体)

**代码位置**: `src/agents/base.py:98-278`

**执行流程**:

```python
async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
    # 1. 获取页面原始内容
    html = await browser.get_html()
    screenshot = await browser.take_screenshot()

    # 2. 程序化快速分析 (FeatureDetector)
    from src.core.smart_router import FeatureDetector
    detector = FeatureDetector()
    features = detector.analyze(html)

    # 3. 页面结构分析
    structure = self._analyze_structure(html)

    # 4. LLM 深度分析 (DeepSeek)
    if llm_client:
        deep_analysis = await self._llm_analyze(html, spec, llm_client)

    # 5. 反爬检测
    anti_bot = self._detect_anti_bot(html)
```

**实际输出** (推断):

```json
{
  "success": true,
  "structure": {
    "type": "list",
    "complexity": "medium",
    "has_dynamic_content": true,
    "main_content_selector": "article",
    "pagination_type": "url",
    "estimated_items": 16
  },
  "features": {
    "has_cloudflare": true,
    "has_recaptcha": true
  },
  "anti_bot_detected": true,
  "html_snapshot": "<!DOCTYPE html>...",  // 截断至 50000 字符
  "screenshot": "base64_encoded_image..."
}
```

**关键发现**:
- 检测到 Cloudflare Turnstile 反爬机制 (`evidence/site_02_techcrunch/html/page_snapshot.html:5-11`)
- 页面类型识别为 `list` (列表页)
- 发现 16 个 article 容器

---

### 2. PlanAgent (规划智能体)

**代码位置**: `src/agents/base.py:283-497`

**执行流程**:

```python
async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
    # 1. LLM 生成策略 (DeepSeek 推理)
    strategy = await self._generate_with_llm(page_structure, spec, llm_client)

    # 2. 生成提取代码 (GLM 编码)
    if llm_client and hasattr(llm_client, 'code'):
        code = await self._generate_code_with_llm(strategy, spec, llm_client)
    else:
        code = self._generate_code(strategy, spec)
```

**Spec 配置** (`specs/test_sites/site_02_techcrunch.yaml`):

```yaml
targets:
  - name: "articles"
    fields:
      - name: "title"
        selector: "h2 a"           # 与实际页面不匹配！
        required: true
      - name: "summary"
        selector: ".article-content p"  # 与实际页面不匹配！
      - name: "author"
        selector: ".author-name"
      - name: "image_url"
        selector: "img"
        attribute: "src"
      - name: "article_url"
        selector: "h2 a"
        attribute: "href"
        required: true
      - name: "published_time"
        selector: "time"
        attribute: "datetime"
      - name: "category"
        selector: ".category-link"
```

**生成的策略** (推断):

```json
{
  "strategy_type": "css",
  "selectors": {
    "title": "h2 a",
    "summary": ".article-content p",
    "author": ".author-name",
    "image_url": "img",
    "article_url": "h2 a",
    "published_time": "time",
    "category": ".category-link"
  },
  "container_selector": "article",
  "pagination_strategy": "none",
  "estimated_items": 16
}
```

**问题分析**:
- Spec 中的选择器 `h2 a`, `.article-content p` 等与 TechCrunch 当前页面结构不匹配
- LLM 可能基于过时的 spec 生成策略，未能适应实际 HTML 结构

---

### 3. ActAgent (执行智能体)

**代码位置**: `src/agents/base.py:553-830`

**执行流程**:

```python
async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
    # 1. 获取 HTML
    html = await browser.get_html()

    # 2. 执行提取
    if generated_code and strategy.get('strategy_type') == 'css':
        extracted_data = await self._execute_code(generated_code, html)
    else:
        extracted_data = await self._extract_with_selectors(
            browser, selectors, strategy, metrics, required_fields
        )

    # 3. 处理分页 (如有)
    if pagination_type != 'none':
        all_data = await self._handle_pagination(...)
```

**实际提取结果**:

```json
[
  {
    "title": "",           // 空字符串 - 选择器未匹配
    "summary": "",         // 空字符串 - 选择器未匹配
    "author": "",          // 空字符串 - 选择器未匹配
    "image_url": "",       // 空字符串 - 选择器未匹配
    "article_url": "",     // 空字符串 - 选择器未匹配
    "published_time": "6 hours ago",  // ✓ 成功匹配
    "category": ""         // 空字符串 - 选择器未匹配
  },
  // ... 共 16 条类似数据
]
```

**提取指标** (ExtractionMetrics):

```json
{
  "total_items": 16,
  "successful_items": 0,  // 没有完整数据项
  "success_rate": 0.0,
  "failed_selectors": {
    "h2 a": 16,           // 每个容器都失败
    ".article-content p": 16,
    ".author-name": 16,
    "img": 16,
    ".category-link": 16
  },
  "missing_fields": {
    "title": 16,
    "summary": 16,
    "author": 16,
    "image_url": 16,
    "article_url": 16,
    "category": 16
  }
}
```

---

### 4. VerifyAgent (验证智能体)

**代码位置**: `src/agents/base.py:834-973`

**执行流程**:

```python
async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
    # 计算质量分数
    quality_score = self._calculate_quality(extracted_data, spec)

    # 构建验证结果
    verification_result = self._build_verification_result(
        extracted_data, spec, quality_score, extraction_metrics
    )
```

**质量计算逻辑**:

```python
def _calculate_quality(self, data: List, spec: Any) -> float:
    # 必填字段: title, article_url
    required_fields = {'title', 'article_url'}

    # 检查完整性
    complete_count = 0
    for item in data:
        if all(item.get(f) for f in required_fields):
            complete_count += 1

    return complete_count / len(data)  # 0 / 16 = 0.0
```

**实际质量分数**: 0.679

**注意**: 质量分数 0.679 > 0.0 的原因可能是:
- `published_time` 字段有值，计入部分得分
- 系统可能使用了备选计算方法

**验证结果**:

```json
{
  "status": "partial",
  "total_items": 16,
  "valid_items": 0,
  "quality_score": 0.679,
  "issues": [
    "超过50%的数据缺少必填字段",
    "部分选择器匹配失败: ['h2 a', '.article-content p', ...]"
  ],
  "scores": {
    "completeness": 0.679,
    "consistency": 1.0  // 所有项结构一致
  }
}
```

---

### 5. 门禁检查 (CompletionGate)

**代码位置**: `src/main.py:354-371`

```python
# 5. Gate: 门禁检查
current_state = self.state_manager.get_state()
gate_passed = self.completion_gate.check(current_state, self.spec)

if gate_passed:
    # 成功完成
    return {'success': True, ...}
```

**门禁条件** (`specs/test_sites/site_02_techcrunch.yaml`):

```yaml
completion_criteria:
  min_items: 5           # 最少 5 条数据 ✓
  quality_threshold: 0.7 # 质量阈值 0.7 ✗ (实际 0.679)
  max_error_rate: 0.2
```

**检查结果**: 未通过 (质量分数 0.679 < 0.7)

---

### 6. JudgeAgent (决策智能体)

**代码位置**: `src/agents/base.py:977-1114`

**执行流程**:

```python
async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
    # 1. 程序快速判断
    decision, reasoning = self._quick_decision(
        quality_score, iteration, max_iterations, errors, extracted_data_count
    )

    # 2. LLM 增强判断 (如有 LLM 且决策不是 complete)
    if llm_client and decision != 'complete':
        enhanced_decision = await self._llm_judge(context, llm_client)
```

**快速决策逻辑**:

```python
def _quick_decision(self, quality_score, iteration, max_iterations, errors, data_count):
    # 成功条件
    if quality_score >= 0.8 and data_count >= 5:
        return 'complete', "..."

    # 失败条件
    if iteration >= max_iterations:
        return 'terminate', "已达到最大迭代次数"

    # 可提升
    if quality_score >= 0.3:  # 0.679 >= 0.3 ✓
        return 'reflect_and_retry', "质量分数可提升，继续迭代"
```

**迭代 1 决策**:
- 质量: 0.679 >= 0.3
- 迭代: 1 < 3
- 决策: `reflect_and_retry`

**迭代 3 决策**:
- 迭代: 3 >= 3
- 决策: `terminate` ("已达到最大迭代次数")

---

### 7. ReflectAgent (反思智能体)

**代码位置**: `src/agents/base.py:1215-1404`

**执行流程**:

```python
async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
    # 1. 分析错误模式
    error_analysis = self._analyze_errors(errors)

    # 2. 分析执行历史
    history_analysis = self._analyze_history(execution_history)

    # 3. LLM 生成改进建议 (DeepSeek)
    if llm_client:
        improvements = await self._llm_reflect(
            error_analysis, history_analysis, spec, llm_client
        )
```

**错误分析结果**:

```json
{
  "patterns": {
    "selector_error": 16  // 16 次选择器错误
  },
  "most_common": "selector_error",
  "total_errors": 16
}
```

**LLM 反思 Prompt** (`src/agents/base.py:1324-1337`):

```
分析爬取任务失败的原因并生成改进建议：

错误分析：{"patterns": {"selector_error": 16}, "most_common": "selector_error"}
历史记录：{"total_attempts": 1, "quality_trend": [0.679]}
目标：新闻文章：标题/摘要/图片/视频

请输出 JSON 格式的改进建议：
{
    "action": "retry|change_strategy|abort",
    "reasoning": "原因分析",
    "selectors": {"field_name": "新选择器"},
    "strategy": "新策略描述"
}
```

**改进建议** (推断):

```json
{
  "action": "retry",
  "reasoning": "选择器与页面结构不匹配，建议重新分析",
  "selectors": {
    "title": "h3 a",           // 尝试新选择器
    "article_url": "h3 a"
  },
  "strategy": "reanalyze"
}
```

**实际效果**:
- 反思后更新了选择器，但 3 次迭代仍未成功匹配
- 每次迭代都因相同原因失败：选择器与实际 HTML 不匹配

---

## LLM 调用统计

| 调用场景 | LLM Provider | 每迭代次数 | 总调用次数 |
|----------|--------------|------------|------------|
| SenseAgent._llm_analyze | DeepSeek | 1 | 3 |
| PlanAgent._generate_with_llm | DeepSeek | 1 | 3 |
| PlanAgent._generate_code_with_llm | GLM (智谱) | 1 | 3 |
| JudgeAgent._llm_judge | DeepSeek | 1 | 3 |
| ReflectAgent._llm_reflect | DeepSeek | 1 | 2 |

**总计**: 约 14 次 LLM 调用

---

## 问题根因分析

### 1. 选择器过时

Spec 文件中定义的选择器与 TechCrunch 当前页面结构不匹配:

| 字段 | Spec 选择器 | 问题 |
|------|-------------|------|
| title | `h2 a` | 页面可能使用其他标题元素 |
| summary | `.article-content p` | 类名已变更 |
| author | `.author-name` | 类名已变更 |

### 2. 反爬机制干扰

页面检测到 Cloudflare Turnstile:

```html
<!-- 来自 page_snapshot.html:5-11 -->
<script>
    const successData = localStorage.getItem('turnstile_verified');
    const failData = sessionStorage.getItem('turnstile_failed');
    const isBot = failData ? 'true' : successData ? 'false' : null;
</script>
```

### 3. 迭代策略局限

反思机制虽然能识别选择器问题，但:
- 没有真实 HTML 样本用于学习新选择器
- LLM 无法看到实际页面结构
- 建议的选择器仍然是猜测

---

## 改进建议

### 短期修复

1. **更新选择器**:
   ```yaml
   # 基于 2026-02-27 的实际页面结构
   selectors:
     title: ".article__title a"  # 实际选择器需要验证
     article_url: ".article__title a"
   ```

2. **增加截图分析**:
   - 在 ReflectAgent 中传入截图让 LLM 分析

### 中期优化

1. **动态选择器学习**:
   - SenseAgent 提取样本 HTML 片段
   - LLM 基于实际样本生成选择器

2. **反爬应对**:
   - 集成 stealth 插件
   - 增加 Turnstile 挑战处理

### 长期架构

1. **自适应选择器**:
   - 维护选择器版本库
   - 自动检测并更新过时选择器

2. **多模态理解**:
   - 结合视觉模型理解页面布局
   - 提高复杂页面的提取准确率

---

## 证据文件

| 文件 | 路径 |
|------|------|
| HTML 快照 | `evidence/site_02_techcrunch/html/page_snapshot.html` |
| 页面截图 | `evidence/site_02_techcrunch/screenshots/page.png` |
| 证据索引 | `evidence/site_02_techcrunch/index.json` |

---

## 结论

TechCrunch 测试展示了系统的完整迭代循环能力，但暴露了以下问题:

1. **Spec 驱动的局限**: 静态配置无法适应页面变化
2. **反思深度不足**: 反思未能产生有效的选择器改进
3. **反爬影响**: Cloudflare Turnstile 可能影响了页面渲染

系统核心架构（七阶段循环、多 LLM 协作、迭代优化）运行正常，但需要增强动态适应能力。