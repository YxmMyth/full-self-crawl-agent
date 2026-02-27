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
        """
        features = {
            'has_login': self._detect_login_form(html),
            'has_pagination': self._detect_pagination(html),
            'is_spa': self._detect_spa(html),
            'anti_bot_level': self._detect_anti_bot(html),
        }

        # 分析页面类型和复杂度
        features['page_type'] = self._classify_page_type(features)
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

    def _detect_pagination(self, html: str) -> bool:
        """检测是否有分页"""
        pagination_patterns = [
            r'page=\d+',
            r'下一页|next\s*page',
            r'pagination|分页',
        ]
        for pattern in pagination_patterns:
            if re.search(pattern, html, re.IGNORECASE):
                return True
        return False

    def _detect_spa(self, html: str) -> bool:
        """检测是否是SPA"""
        spa_indicators = ['fetch(', 'XMLHttpRequest', 'div id="app"', 'div id="root"']
        lower_html = html.lower()
        return any(indicator.lower() in lower_html for indicator in spa_indicators)

    def _detect_anti_bot(self, html: str) -> str:
        """检测反爬等级"""
        lower_html = html.lower()
        if any(keyword in lower_html for keyword in ['cloudflare', 'recaptcha', 'captcha', 'turnstile']):
            return 'high'
        elif any(keyword in lower_html for keyword in ['user-agent', 'referer', 'csrf']):
            return 'medium'
        return 'none'

    def _classify_page_type(self, features: Dict[str, Any]) -> str:
        """分类页面类型"""
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
            'capabilities': ['sense', 'handle_spa', 'api_extract', 'verify'],
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
                              'detect_login', 'api_extract', 'slow_plan'}

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

    async def explore(self, url: str, goal: str, strategies=None):
        """
        渐进式探索

        从简单到复杂尝试策略
        """
        if strategies is None:
            strategies = self.STRATEGY_ORDER

        for strategy_name, expected_rate in strategies:
            # 这里应该调用实际的策略执行器
            # 暂时返回模拟结果
            result = {
                'strategy': strategy_name,
                'success': False,
                'quality': 0.0,
                'data': []
            }

            # 如果成功且质量达标，返回结果
            if result.get('success') and result.get('quality', 0) >= 0.6:
                return result

        return {'success': False, 'error': '所有策略失败'}
