"""
智能路由 - 管理层
三层决策模式：程序→规则→LLM

根据 IMPLEMENTATION.md 第3.1节的完整设计，包括：
1. 特征检测器 (FeatureDetector) - 程序快速分析
2. 智能路由 (SmartRouter) - 三层决策模式
3. 策略库和备选方案
"""

from typing import Dict, Any, List, Optional, Literal, Tuple
from datetime import datetime
import re
import json

try:
    from bs4 import BeautifulSoup, Tag
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

# -- FeatureDetector 调优常量 --
_SPA_BODY_TEXT_THRESHOLD = 200          # body 文本少于此值时认为是薄内容（SPA 信号）
_DETAIL_PARAGRAPH_THRESHOLD = 200       # 单个 <p> 超过此字数认为是长文本详情页
_CONTAINER_SIMILARITY_THRESHOLD = 0.6  # 重复容器子结构相似度阈值


# ==================== 特征检测器 ====================

class FeatureDetector:
    """特征检测器 - 程序快速分析（<50ms）"""

    def analyze(self, html: str, url: Optional[str] = None) -> Dict[str, Any]:
        """
        快速分析页面特征

        返回包含以下字段的字典：
        - has_login: 是否有登录表单
        - has_pagination: 是否有分页
        - is_spa: 是否是SPA应用
        - anti_bot_level: 反爬等级
        - page_type: 页面类型
        - complexity: 复杂度评级
        - container_info: 重复容器检测结果（新增）
        """
        soup = BeautifulSoup(html, 'html.parser') if _BS4_AVAILABLE else None

        features = {
            'has_login': self._detect_login_form(html),
            'has_pagination': self._detect_pagination(soup, html),
            'is_spa': self._detect_spa(soup, html),
            'anti_bot_level': self._detect_anti_bot(html),
        }

        # 新增：重复容器检测
        features['container_info'] = self._detect_repeating_containers(soup, html)

        # 统一页面类型分类（新增），同时保留 page_type 字段
        features['page_type'] = self._classify_page_type_unified(features, soup, html)
        features['complexity'] = self._assess_complexity(features)

        return features

    def _detect_login_form(self, html: str) -> bool:
        """检测是否有登录表单"""
        login_patterns = [
            r'<input[^>]+type=["\']?password["\']?[^>]*>',
            r'<form[^>]+action=["\']?login["\']?[^>]*>',
        ]
        for pattern in login_patterns:
            if re.search(pattern, html, re.IGNORECASE | re.MULTILINE):
                return True
        return False

    def _detect_pagination(self, soup: Any, html: str) -> bool:
        """检测是否有分页（重写版：rel="next"优先 + 容器 + 链接模式 + 精确文本）"""
        # 信号1：rel="next" 链接（最可靠）
        if soup is not None:
            if soup.find('a', rel=lambda r: r and 'next' in r):
                return True
            if soup.find('link', rel=lambda r: r and 'next' in r):
                return True

        # 信号2：分页容器（class/id 含 pagination/pager/pages）
        if re.search(
            r'class=["\'][^"\']*\b(pagination|pager|pages)\b[^"\']*["\']|'
            r'id=["\'][^"\']*\b(pagination|pager|pages)\b[^"\']*["\']',
            html, re.IGNORECASE
        ):
            return True

        # 信号3：URL 分页模式
        if re.search(r'(?:href|action)[^>]*[?&/]page[=/]\d+', html, re.IGNORECASE):
            return True

        # 信号4：精确文字（避免误判 "next generation" 等短语）
        if re.search(r'\b(下一页|上一页|next\s+page|prev(?:ious)?\s+page)\b', html, re.IGNORECASE):
            return True

        return False

    def _detect_spa(self, soup: Any, html: str) -> bool:
        """检测是否是SPA（重写版：body内容薄 + 框架标记 + JS bundle数量三信号）"""
        score = 0

        # 信号1：body 内容薄（body 文本极少，说明依赖 JS 渲染）
        if soup is not None:
            body = soup.find('body')
            if body:
                body_text = body.get_text(separator=' ', strip=True)
                if len(body_text) < _SPA_BODY_TEXT_THRESHOLD:
                    score += 1

        # 信号2：框架挂载点标记
        framework_patterns = [
            r'<div[^>]+id=["\']app["\']',
            r'<div[^>]+id=["\']root["\']',
            r'<div[^>]+id=["\']__next["\']',
            r'data-reactroot',
            r'ng-version=',
            r'data-v-\w+',           # Vue scoped attribute
            r'__vue_app__',
            r'__NEXT_DATA__',
            r'window\.__NUXT__',
        ]
        if any(re.search(p, html, re.IGNORECASE) for p in framework_patterns):
            score += 1

        # 信号3：多个 JS bundle（chunk/bundle 文件 ≥2）
        js_bundles = re.findall(
            r'<script[^>]+src=["\'][^"\']*(?:chunk|bundle|vendor|main|app)[^"\']*\.js[^"\']*["\']',
            html, re.IGNORECASE
        )
        if len(js_bundles) >= 2:
            score += 1

        return score >= 2

    def _detect_anti_bot(self, html: str) -> str:
        """检测反爬等级"""
        lower_html = html.lower()
        if any(keyword in lower_html for keyword in ['cloudflare', 'recaptcha', 'captcha', 'turnstile']):
            return 'high'
        elif any(keyword in lower_html for keyword in ['user-agent', 'referer', 'csrf']):
            return 'medium'
        return 'none'

    # ------------------------------------------------------------------
    # 新增方法
    # ------------------------------------------------------------------

    def _clean_class_name(self, class_name: str) -> str:
        """去掉 CSS Modules / Tailwind hash 后缀（如 _abc123、--xYzQ）。

        仅去掉包含数字的 hash 后缀（≥4 位），避免误删 _main 之类的有意义后缀。
        """
        # 匹配形如 _<hash> 或 --<hash> 的后缀，要求包含至少一个数字
        cleaned = re.sub(r'[_-]{1,2}(?=[a-zA-Z0-9]*\d)[a-zA-Z0-9]{4,}$', '', class_name)
        return cleaned.strip()

    def _compute_structure_similarity(self, elem_a: Any, elem_b: Any) -> float:
        """
        计算两个元素的子标签序列相似度。

        算法：位置敏感的标签序列匹配，以两序列最大长度归一化。
        纯文本叶节点（无子标签）返回 0.0，不视为结构相似。
        """
        if not _BS4_AVAILABLE or elem_a is None or elem_b is None:
            return 0.0

        def child_tags(elem: Any) -> List[str]:
            return [c.name for c in elem.children if hasattr(c, 'name') and c.name]

        tags_a = child_tags(elem_a)
        tags_b = child_tags(elem_b)

        # 纯文本叶节点（无子标签）不构成结构相似的证据
        if not tags_a and not tags_b:
            return 0.0
        if not tags_a or not tags_b:
            return 0.0

        # 使用长度归一化的匹配度（顺序敏感）
        matches = sum(1 for a, b in zip(tags_a, tags_b) if a == b)
        similarity = matches / max(len(tags_a), len(tags_b))
        return similarity

    def _is_detail_page(self, soup: Any, html: str) -> bool:
        """判断是否是详情页（article 标签 / 长文本段落 / h1+p 结构）"""
        if soup is None:
            return False

        # 信号1：存在 <article> 标签
        if soup.find('article'):
            return True

        # 信号2：单个 <h1> 且后面有较多 <p>
        h1_tags = soup.find_all('h1')
        if len(h1_tags) == 1:
            p_tags = soup.find_all('p')
            if len(p_tags) >= 3:
                return True

        # 信号3：body 中有长文本段落（单个 p 超过 200 字）
        for p in soup.find_all('p'):
            if len(p.get_text(strip=True)) > _DETAIL_PARAGRAPH_THRESHOLD:
                return True

        return False

    def _detect_repeating_containers(self, soup: Any, html: str) -> Dict[str, Any]:
        """
        检测重复结构容器（列表页识别），不依赖 class 名。

        返回 container_info 字典：
        - found: bool
        - tag: str（容器标签）
        - count: int（重复数量）
        - similarity: float（结构相似度）
        """
        empty = {'found': False, 'tag': None, 'count': 0, 'similarity': 0.0}
        if not _BS4_AVAILABLE or soup is None:
            return empty

        # 候选容器标签（语义化优先）
        candidate_tags = ['ul', 'ol', 'table', 'div', 'section', 'article']

        for tag_name in candidate_tags:
            containers = soup.find_all(tag_name)

            # 找到子元素数量足够的容器
            for container in containers:
                direct_children = [c for c in container.children
                                   if hasattr(c, 'name') and c.name]
                if len(direct_children) < 3:
                    continue

                # 取前几个子元素，判断彼此结构相似度
                samples = direct_children[:min(5, len(direct_children))]
                if len(samples) < 2:
                    continue

                similarities = []
                for i in range(len(samples) - 1):
                    sim = self._compute_structure_similarity(samples[i], samples[i + 1])
                    similarities.append(sim)

                avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
                if avg_sim >= _CONTAINER_SIMILARITY_THRESHOLD:
                    return {
                        'found': True,
                        'tag': tag_name,
                        'count': len(direct_children),
                        'similarity': round(avg_sim, 3),
                    }

        return empty

    def _classify_page_type_unified(
        self,
        features: Dict[str, Any],
        soup: Any,
        html: str,
    ) -> str:
        """
        统一页面类型分类，输出 list|detail|form|spa|interactive|other
        （取代旧的 _classify_page_type）
        """
        if features.get('is_spa'):
            return 'spa'

        if features.get('has_login'):
            return 'form'

        container_info = features.get('container_info', {})
        if container_info.get('found'):
            return 'list'

        if self._is_detail_page(soup, html):
            return 'detail'

        # 检测表单页（有 <form> 且无登录特征）
        if soup is not None and soup.find('form'):
            return 'form'
        elif re.search(r'<form\b', html, re.IGNORECASE):
            return 'form'

        return 'other'

    # ------------------------------------------------------------------
    # 保留旧方法（兼容调用方）
    # ------------------------------------------------------------------

    def _classify_page_type(self, features: Dict[str, Any]) -> str:
        """分类页面类型（旧版，保留兼容）"""
        if features.get('is_spa'):
            return 'spa'
        elif features.get('has_login'):
            return 'interactive'
        else:
            return 'static'

    def _assess_complexity(self, features: Dict[str, Any]) -> str:
        """评估复杂度"""
        score = 0
        if features.get('is_spa'):
            score += 2
        if features.get('has_login'):
            score += 2
        if features.get('anti_bot_level') == 'high':
            score += 2
        elif features.get('anti_bot_level') == 'medium':
            score += 1

        if score >= 4:
            return 'complex'
        elif score >= 2:
            return 'medium'
        else:
            return 'simple'


# ==================== 智能路由 ====================

class SmartRouter:
    """
    智能路由 - 混合判断核心

    三层决策模式：
    1. 程序快速判断（<50ms）- 规则明确、效率优先
    2. 规则引擎判断（<500ms）- 多条件组合、模式匹配
    3. LLM深度分析（2-3秒）- 语义理解、策略生成

    根据 IMPLEMENTATION.md 第3.1.2节实现
    """

    # 策略库
    STRATEGY_LIBRARY = {
        'direct_crawl': {
            'name': 'direct_crawl',
            'capabilities': ['sense', 'plan', 'act', 'verify'],
            'expected_success_rate': 0.95,
            'complexity': 'simple',
        },
        'pagination_crawl': {
            'name': 'pagination_crawl',
            'capabilities': ['sense', 'plan', 'act', 'verify', 'handle_pagination'],
            'expected_success_rate': 0.85,
            'complexity': 'medium',
        },
        'spa_crawl': {
            'name': 'spa_crawl',
            'capabilities': ['sense', 'spa_handle', 'verify'],
            'expected_success_rate': 0.70,
            'complexity': 'complex',
        },
        'login_required': {
            'name': 'login_required',
            'capabilities': ['detect_login', 'handle_login', 'sense', 'plan', 'act', 'verify'],
            'expected_success_rate': 0.60,
            'complexity': 'complex',
        },
        'strong_anti_bot': {
            'name': 'strong_anti_bot',
            'capabilities': ['sense', 'handle_anti_bot', 'slow_plan', 'act', 'verify'],
            'expected_success_rate': 0.50,
            'complexity': 'extremely_complex',
        },
    }

    def __init__(self, llm_client: Optional[Any] = None):
        """
        初始化智能路由

        Args:
            llm_client: LLM客户端，用于LLM深度分析
        """
        self.llm_client = llm_client
        self.feature_detector = FeatureDetector()
        self.program_decisions = 0
        self.rule_decisions = 0
        self.llm_decisions = 0

    async def route(
        self,
        url: str,
        goal: str,
        html: Optional[str] = None,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        路由决策 - 三层决策模式

        Args:
            url: 目标URL
            goal: 用户目标
            html: 页面HTML内容（可选）
            use_llm: 是否使用LLM深度分析

        Returns:
            路由决策字典，包含：
            - strategy: 策略名称
            - capabilities: 需要的能力列表
            - expected_success_rate: 预期成功率
            - complexity: 复杂度
            - page_type: 页面类型
            - special_requirements: 特殊需求
            - decided_at: 决策时间
            - decision_duration: 决策耗时（秒）
        """
        from datetime import datetime
        start_time = datetime.now()

        # ========== 第1级：程序快速判断（<50ms）==========
        if html:
            features = self.feature_detector.analyze(html, url)
        else:
            features = {}

        # 简单场景直接返回
        if features.get('complexity') == 'simple' and not use_llm:
            strategy = self.STRATEGY_LIBRARY['direct_crawl'].copy()
            self.program_decisions += 1
        else:
            # ========== 第2级：规则引擎 ==========
            strategy = self._match_from_library(features, goal)

            if strategy.get('complexity') in ['complex', 'extremely_complex'] and use_llm:
                # ========== 第3级：LLM深度分析（2-3秒）==========
                strategy = await self._generate_with_llm(features, goal, html)
                self.llm_decisions += 1
            else:
                self.rule_decisions += 1

        # ========== 程序验证 ==========
        if not self._validate_strategy(strategy):
            strategy = self._fallback_strategy(features)

        # 记录决策
        decision_duration = (datetime.now() - start_time).total_seconds()

        decision = {
            'strategy': strategy['name'],
            'capabilities': strategy['capabilities'],
            'expected_success_rate': strategy['expected_success_rate'],
            'complexity': features.get('complexity', 'simple'),
            'page_type': features.get('page_type', 'unknown'),
            'special_requirements': self._extract_requirements(features),
            'execution_params': strategy.get('params', {}),
            'fallback_strategies': strategy.get('fallback_strategies', []),
            'decided_at': datetime.now().isoformat(),
            'decision_duration': decision_duration,
        }

        return decision

    async def _generate_with_llm(
        self,
        features: Dict[str, Any],
        goal: str,
        html: Optional[str]
    ) -> Dict[str, Any]:
        """使用LLM生成策略 - 推理任务，使用 DeepSeek"""
        if self.llm_client is None:
            return self._match_from_library(features, goal)

        html_sample = (html or '')[:5000] if html else ''

        prompt = f"""
# 任务分析
{json.dumps(features, indent=2, ensure_ascii=False)}

# 用户目标
{goal}

# 页面片段（前5000字符）
```html
{html_sample}
```

# 请生成最适合的爬取策略

考虑以下方面：
1. 推荐使用哪些能力（从7种中选择：sense, plan, act, verify, judge, explore, reflect）
2. 具体的执行步骤
3. 可能遇到的挑战
4. 应对策略
5. 预期成功率（0.0-1.0）

# 输出格式（JSON）
```json
{{
    "strategy": "策略名称",
    "capabilities": ["能力1", "能力2"],
    "steps": ["步骤1", "步骤2"],
    "considerations": ["注意事项"],
    "expected_success_rate": 0.8
}}
```
"""

        try:
            # 推理任务 - 使用 reason() 方法或 task_type='reasoning'
            if hasattr(self.llm_client, 'reason'):
                response = await self.llm_client.chat(
                    prompt,
                    task_type='reasoning'
                )
            else:
                response = await self.llm_client.chat(prompt)

            # 解析响应
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                json_str = response.split('```')[1].split('```')[0]
            else:
                json_str = response

            strategy_dict = json.loads(json_str)

            return {
                'name': strategy_dict.get('strategy', 'custom'),
                'capabilities': strategy_dict.get('capabilities', ['sense', 'plan', 'act', 'verify']),
                'expected_success_rate': strategy_dict.get('expected_success_rate', 0.7),
                'params': {
                    'steps': strategy_dict.get('steps', []),
                    'considerations': strategy_dict.get('considerations', []),
                }
            }

        except Exception as e:
            print(f"LLM生成策略失败: {e}")
            return self._match_from_library(features, goal)

    def _match_from_library(self, features: Dict[str, Any], goal: str) -> Dict[str, Any]:
        """从策略库中匹配策略"""
        # 根据特征匹配策略
        if features.get('has_login'):
            return self.STRATEGY_LIBRARY['login_required'].copy()

        if features.get('is_spa'):
            return self.STRATEGY_LIBRARY['spa_crawl'].copy()

        if features.get('has_pagination'):
            return self.STRATEGY_LIBRARY['pagination_crawl'].copy()

        if features.get('anti_bot_level') == 'high':
            return self.STRATEGY_LIBRARY['strong_anti_bot'].copy()

        return self.STRATEGY_LIBRARY['direct_crawl'].copy()

    def _validate_strategy(self, strategy: Dict[str, Any]) -> bool:
        """验证策略的有效性"""
        if not strategy.get('name'):
            return False

        if not strategy.get('capabilities'):
            return False

        expected_rate = strategy.get('expected_success_rate', 0)
        if not (0 <= expected_rate <= 1):
            return False

        # 验证能力合理性
        valid_capabilities = {'sense', 'plan', 'act', 'verify', 'judge', 'explore', 'reflect',
                              'handle_login', 'handle_spa', 'handle_anti_bot', 'handle_pagination',
                              'detect_login', 'api_extract', 'slow_plan', 'spa_handle'}

        capabilities = strategy.get('capabilities', [])
        return all(cap in valid_capabilities for cap in capabilities)

    def _fallback_strategy(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """备选策略"""
        return {
            'name': 'fallback',
            'capabilities': ['sense', 'plan', 'act', 'verify'],
            'expected_success_rate': 0.5,
            'params': {'retry_count': 3, 'timeout': 60},
            'fallback_strategies': ['direct_crawl', 'pagination_crawl']
        }

    def _extract_requirements(self, features: Dict[str, Any]) -> List[str]:
        """提取特殊需求"""
        requirements = []
        if features.get('has_login'):
            requirements.append('login')
        if features.get('is_spa'):
            requirements.append('javascript')
        if features.get('has_pagination'):
            requirements.append('pagination')
        if features.get('anti_bot_level') == 'high':
            requirements.append('anti-bot-high')
        elif features.get('anti_bot_level') == 'medium':
            requirements.append('anti-bot-medium')
        return requirements

    def get_routing_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        total = self.program_decisions + self.rule_decisions + self.llm_decisions
        return {
            'program_decisions': self.program_decisions,
            'rule_decisions': self.rule_decisions,
            'llm_decisions': self.llm_decisions,
            'total_decisions': total,
            'program_ratio': self.program_decisions / total if total > 0 else 0,
            'rule_ratio': self.rule_decisions / total if total > 0 else 0,
            'llm_ratio': self.llm_decisions / total if total > 0 else 0,
        }


# ==================== 能力动态组合 ====================

def compose_capabilities(task_analysis: Dict[str, Any]) -> List[str]:
    """
    动态组合能力

    根据 IMPLEMENTATION.md 第3.2.2节实现
    """
    capabilities = ['sense', 'plan', 'act', 'verify']

    requirements = task_analysis.get('special_requirements', [])

    if 'login' in requirements:
        capabilities.insert(0, 'handle_login')

    if 'pagination' in requirements:
        capabilities.append('handle_pagination')

    if 'javascript' in requirements:
        capabilities.insert(1, 'handle_spa')

    if 'anti-bot' in requirements:
        capabilities.insert(1, 'handle_anti_bot')

    return capabilities


# ==================== 渐进式探索 ====================

class ProgressiveExplorer:
    """渐进式探索（从简单到复杂）"""

    STRATEGY_ORDER = [
        ('direct_crawl', 0.95),        # 简单页面
        ('pagination_crawl', 0.85),    # 列表页
        ('api_reverse', 0.70),         # SPA
        ('login_required', 0.40),      # 需登录
        ('headless_browser', 0.50),    # 复杂JS
    ]

    def __init__(self, router: Optional['SmartRouter'] = None):
        self.router = router or SmartRouter()

    async def explore(self, url: str, goal: str, html: Optional[str] = None,
                      strategies=None):
        """
        渐进式探索

        从简单到复杂依次通过 SmartRouter 评估各策略可行性，
        返回第一个预期成功率满足要求的路由决策。

        Args:
            url: 目标 URL
            goal: 爬取目标描述
            html: 可选的页面 HTML（已获取时传入以节省一次请求）
            strategies: 自定义策略顺序列表，格式 [(名称, 最低成功率), ...]

        Returns:
            满足条件的路由决策字典，或 {'success': False, 'error': ...}
        """
        if strategies is None:
            strategies = self.STRATEGY_ORDER

        last_decision: Optional[Dict[str, Any]] = None

        for strategy_name, min_success_rate in strategies:
            # 跳过与当前页面特征不匹配的策略
            if not self._is_applicable(strategy_name, url, html):
                continue

            # 通过 SmartRouter 路由（对复杂策略允许使用 LLM）
            use_llm = strategy_name in ('api_reverse', 'login_required', 'headless_browser')
            decision = await self.router.route(url, goal, html=html, use_llm=use_llm)

            last_decision = decision

            # 如果路由的预期成功率满足该策略的最低要求，则采用此决策
            if decision.get('expected_success_rate', 0) >= min_success_rate:
                return {
                    'success': True,
                    'strategy': strategy_name,
                    'decision': decision,
                }

        # 所有策略都不满足要求，返回最后一次路由决策供上层参考
        return {
            'success': False,
            'error': '所有策略的预期成功率均低于阈值',
            'last_decision': last_decision,
        }

    def _is_applicable(self, strategy_name: str, url: str,
                       html: Optional[str]) -> bool:
        """
        快速判断策略是否适用于当前页面

        基于 URL 模式和 HTML 特征进行轻量级过滤，
        避免对明显不适用的策略发起完整的路由分析。
        """
        if html is None:
            # 没有 HTML 时，只过滤掉需要已登录状态的策略
            return strategy_name != 'login_required'

        html_lower = html.lower()

        if strategy_name == 'direct_crawl':
            # 直接爬取适用于所有页面
            return True

        if strategy_name == 'pagination_crawl':
            # 只有存在分页特征时才适用
            return bool(
                re.search(r'page=\d+|/page/\d+|下一页|next\s*page|pagination', html_lower)
            )

        if strategy_name == 'api_reverse':
            # 存在 SPA/AJAX 特征时适用
            return bool(
                re.search(r'fetch\(|xmlhttprequest|div\s+id=["\']app["\']|div\s+id=["\']root["\']',
                          html_lower)
            )

        if strategy_name == 'login_required':
            # 检测到登录表单时适用
            return bool(
                re.search(r'type=["\']?password["\']?|action=["\']?login["\']?', html_lower)
            )

        if strategy_name == 'headless_browser':
            # 检测到 Cloudflare 或复杂 JS 时适用
            return bool(
                re.search(r'cloudflare|recaptcha|turnstile|challenge', html_lower)
            )

        return True
