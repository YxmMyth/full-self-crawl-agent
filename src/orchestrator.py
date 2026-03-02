"""
Orchestrator - SelfCrawlingAgent核心编排逻辑
负责整个爬虫代理的协调和流程控制
"""

import asyncio
import inspect
import json
import sys
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import time

from .agents.base import AgentPool, AgentCapability
from .core.crawl_frontier import CrawlFrontier, canonicalize_url
from .core.context_compressor import ContextCompressor
from .core.state_manager import StateManager
from .core.risk_monitor import RiskMonitor
from .core.meta_controller import MetaController
from .config.loader import load_config
from .core.spec_inferrer import SpecInferrer

logger = logging.getLogger(__name__)


class SelfCrawlingAgent:
    """
    自爬虫智能体 - 核心编排器
    完整的迭代循环实现：Sense → Plan → Act → Verify → Judge → Reflect
    """

    def __init__(self, config_path: Optional[str] = None):
        # 加载配置
        self.config = load_config(config_path)
        self.state_manager = StateManager()

        # 初始化组件
        from .tools.llm_client import LLMClient
        from .executors.executor import Sandbox
        from .tools.browser import Browser

        llm_cfg = self.config.get('llm', {})
        api_key = llm_cfg.get('api_key', '')
        # 从环境变量读取 API key（优先级：config > API_GATEWAY_KEY > DEEPSEEK_API_KEY）
        if not api_key or not api_key.strip():
            for env_var in ['API_GATEWAY_KEY', 'DEEPSEEK_API_KEY']:
                api_key = os.environ.get(env_var, '')
                if api_key and api_key.strip():
                    logger.info(f"从环境变量 {env_var} 读取 API key")
                    break
        if not api_key or not api_key.strip():
            logger.warning(
                "LLM API key 未设置或为空。Agent 将以纯确定性模式运行（无 LLM 规划/分析），"
                "提取质量会降低。请设置环境变量或 config 中的 llm.api_key。"
            )
        self.llm_client = LLMClient(
            api_key=api_key,
            model=llm_cfg.get('model', 'claude-opus-4-5-20251101'),
            api_base=llm_cfg.get('api_base') or llm_cfg.get('base_url')
        )
        self._llm_available = bool(api_key and api_key.strip())

        # Vision-LLM 客户端（用于多模态页面分析，默认 Gemini）
        vision_cfg = self.config.get('vision_llm', {})
        vision_key = vision_cfg.get('api_key', '') or api_key  # 回退使用主 LLM key
        vision_model = vision_cfg.get('model', 'gemini-2.5-flash')
        vision_base = vision_cfg.get('api_base') or llm_cfg.get('api_base')
        self.vision_llm_client = None
        if vision_key and vision_key.strip():
            self.vision_llm_client = LLMClient(
                api_key=vision_key,
                model=vision_model,
                api_base=vision_base
            )

        sandbox_cfg = self.config.get('sandbox', {})
        from .core.policy_manager import PolicyManager
        self.policy_manager = PolicyManager()
        self.sandbox = Sandbox(
            strict_mode=sandbox_cfg.get('strict_mode', True),
            default_timeout=sandbox_cfg.get('timeout', 60),
            policy_manager=self.policy_manager
        )
        browser_cfg = self.config.get('browser', {})
        self.browser = Browser(headless=browser_cfg.get('headless', True))

        # 智能体池
        self.agent_pool = AgentPool(
            llm_client=self.llm_client,
            sandbox=self.sandbox
        )

        # 核心组件
        self.frontier = CrawlFrontier()
        self.context_compressor = ContextCompressor(max_tokens=8000)
        self.risk_monitor = RiskMonitor()
        self.meta_controller = MetaController()

        # 用于跟踪性能指标
        self.metrics = {
            'total_requests': 0,
            'total_errors': 0,
            'start_time': None,
            'end_time': None
        }

    async def run(self, start_url: Optional[str] = None, spec: Optional[Dict[str, Any]] = None):
        """运行爬取任务"""
        if not hasattr(self, 'metrics'):
            self.metrics = {'total_requests': 0, 'total_errors': 0, 'start_time': None, 'end_time': None}
        self.metrics['start_time'] = datetime.now()

        try:
            spec = spec or getattr(self, 'spec', {}) or {}
            start_url = start_url or spec.get('start_url') or spec.get('url') or spec.get('target_url')
            if not start_url:
                raise ValueError("缺少 start_url/url/target_url")

            # 自动推断 Spec 缺失字段
            inferred_spec = await self._infer_spec(start_url, spec)
            self.spec = inferred_spec
            crawl_mode = inferred_spec.get('crawl_mode', 'full_site')

            print(f"开始爬取模式: {crawl_mode}")
            print(f"目标 URL: {start_url}")

            # 根据模式选择执行路径
            if crawl_mode == 'single_page':
                result = await self._run_single_or_multi(start_url, crawl_mode)
            elif crawl_mode == 'multi_page':
                result = await self._run_single_or_multi(start_url, crawl_mode)
            elif crawl_mode == 'full_site':
                full_site_fn = self._run_full_site
                if len(inspect.signature(full_site_fn).parameters) == 1:
                    result = await full_site_fn(start_url)
                else:
                    result = await full_site_fn(start_url, inferred_spec)
            else:
                raise ValueError(f"不支持的爬取模式: {crawl_mode}")

            return result

        finally:
            self.metrics['end_time'] = datetime.now()
            await self.cleanup()

    async def _infer_spec(self, start_url: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        """自动推断 Spec 缺失字段"""
        inferrer = SpecInferrer(self.browser)
        inferred_spec = await inferrer.infer_missing_fields(start_url, spec.copy())
        return inferred_spec

    def _apply_spec_inference(self, features: Dict[str, Any]) -> None:
        """兼容旧测试入口：基于感知特征补全 self.spec。"""
        current_spec = getattr(self, 'spec', {}) or {}
        patch = SpecInferrer().infer(features or {}, existing_spec=current_spec)
        current_spec.update(patch)
        self.spec = current_spec

    async def _navigate_url(self, url: str) -> None:
        """兼容 BrowserTool.navigate 和旧接口 goto。"""
        navigate = getattr(self.browser, 'navigate', None)
        goto = getattr(self.browser, 'goto', None)
        if callable(navigate):
            ret = navigate(url)
            if inspect.isawaitable(ret):
                await ret
            return
        if callable(goto):
            ret = goto(url)
            if inspect.isawaitable(ret):
                await ret
            return

    async def _run_single_or_multi(self, start_url: str, spec: Any):
        """单页或多页爬取（复用 pipeline.py 中的单页循环）"""
        from .pipeline import run_single_page_pipeline

        if isinstance(spec, str):
            return await self._run_single_or_multi_compat(start_url, spec)
        else:
            spec = spec or getattr(self, 'spec', {}) or {}

        # 准备初始上下文
        llm = getattr(self, 'llm_client', None)
        if not getattr(self, '_llm_available', False):
            llm = None  # 无有效 key 时不传 llm_client，避免无效请求
        initial_context = {
            'start_url': start_url,
            'spec': spec,
            'browser': self.browser,
            'llm_client': llm,
            'vision_llm_client': getattr(self, 'vision_llm_client', None),
            'sandbox': getattr(self, 'sandbox', None),
            'agent_pool': self.agent_pool,
        }

        # 爬取模式相关参数
        crawl_mode = spec.get('crawl_mode', 'full_site')
        max_pages = spec.get('max_pages', 1)
        max_iterations = spec.get('max_iterations', 1 if crawl_mode == 'single_page' else max_pages)
        max_iterations = max(1, int(max_iterations))

        all_results = []
        extracted_data = []
        current_url = start_url
        page_count = 0
        iterations = 0
        spec_inferred = False

        while page_count < max_pages and iterations < max_iterations:
            print(f"正在处理第 {page_count + 1} 页: {current_url}")

            # 跳转到当前 URL
            await self._navigate_url(current_url)

            # 运行单页流水线
            page_result = await run_single_page_pipeline(initial_context)
            iterations += 1
            page_records = page_result.get('extracted_data', [])
            for rec in page_records:
                if isinstance(rec, dict) and 'source_url' not in rec:
                    rec['source_url'] = current_url
            extracted_data.extend(page_records)

            all_results.append({
                'url': current_url,
                'result': page_result,
                'page_number': page_count + 1
            })

            # 风险监控检查
            risk_metrics = {
                'total_items': len(extracted_data),
                'failed_items': 0 if page_result.get('success') else 1,
                'consecutive_errors': sum(1 for r in all_results if not r.get('result', {}).get('success')),
                'iteration_count': iterations,
            }
            alerts = self.risk_monitor.check_metrics(risk_metrics)
            if self.risk_monitor.has_critical_risk():
                critical = self.risk_monitor.get_high_risk_alerts()
                logger.warning(f"风险监控触发终止: {[a.message for a in critical]}")
                return {
                    'success': False,
                    'crawl_mode': crawl_mode,
                    'results': all_results,
                    'extracted_data': extracted_data,
                    'iterations': iterations,
                    'quality_score': page_result.get('verification_result', {}).get('quality_score', 0.0),
                    'error': 'terminated_by_risk_monitor',
                    'risk_alerts': [{'level': a.level.value, 'message': a.message} for a in critical],
                }

            # 仅在首次迭代做一次 spec 推断（兼容旧行为）
            if not spec_inferred:
                features = page_result.get('page_structure', {}).get('features', {})
                self._apply_spec_inference(features)
                spec_inferred = True

            # Meta-Controller: 记录结果 + 自主策略调整
            self.meta_controller.record_outcome(current_url, page_result)
            adjustment = self.meta_controller.evaluate()
            if adjustment:
                print(f"[MetaController] L{adjustment.level.value} 调整: {adjustment.action} — {adjustment.reason}")
                overrides = self.meta_controller.get_context_overrides()
                if 'max_page_retries' in overrides:
                    initial_context['spec'] = {**initial_context['spec'],
                                                'max_page_retries': overrides['max_page_retries']}
                if overrides.get('hint_selectors'):
                    initial_context.setdefault('reflect_hints', {})['suggested_selectors'] = overrides['hint_selectors']
            self.meta_controller.reset_escalation()

            judge_decision = page_result.get('decision', {}).get('decision', 'complete')
            if judge_decision == 'terminate':
                return {
                    'success': False,
                    'crawl_mode': crawl_mode,
                    'results': all_results,
                    'extracted_data': extracted_data,
                    'iterations': iterations,
                    'quality_score': page_result.get('verification_result', {}).get('quality_score', 0.0),
                    'error': 'terminated_by_judge',
                }
            if judge_decision == 'complete':
                return {
                    'success': True,
                    'crawl_mode': crawl_mode,
                    'results': all_results,
                    'extracted_data': extracted_data,
                    'iterations': iterations,
                    'quality_score': page_result.get('verification_result', {}).get('quality_score', 0.0),
                    'total_pages_processed': len(all_results),
                    'summary': self._summarize_results(all_results)
                }

            # 如果是单页模式，则只处理一次
            if crawl_mode == 'single_page':
                break

            # 如果有多页，则尝试获取下一页 URL
            current_url = self._get_next_page_url(current_url, page_count + 2, page_result)
            if not current_url:
                print("无法找到下一页，结束分页爬取")
                break

            page_count += 1

        if iterations >= max_iterations:
            # 聚合已收集的质量分数
            qs = [r.get('result', {}).get('verification_result', {}).get('quality_score', 0.0)
                  for r in all_results if r.get('result', {}).get('success')]
            return {
                'success': False,
                'crawl_mode': crawl_mode,
                'results': all_results,
                'extracted_data': extracted_data,
                'iterations': iterations,
                'quality_score': round(sum(qs) / len(qs), 4) if qs else 0.0,
                'error': 'max_iterations_reached'
            }

        qs = [r.get('result', {}).get('verification_result', {}).get('quality_score', 0.0)
              for r in all_results if r.get('result', {}).get('success')]
        extracted_data = self._deduplicate_records(extracted_data)
        return {
            'success': True,
            'crawl_mode': crawl_mode,
            'results': all_results,
            'extracted_data': extracted_data,
            'iterations': iterations,
            'quality_score': round(sum(qs) / len(qs), 4) if qs else 0.0,
            'total_pages_processed': len(all_results),
            'summary': self._summarize_results(all_results)
        }

    async def _run_single_or_multi_compat(self, start_url: str, crawl_mode: str) -> Dict[str, Any]:
        """兼容旧测试路径：直接按字符串能力调用 agent_pool。"""
        spec = dict(getattr(self, 'spec', {}) or {})
        spec['crawl_mode'] = crawl_mode
        max_iterations = max(1, int(spec.get('max_iterations', 1)))
        iterations = 0
        inferred = False
        extracted_data: List[Dict[str, Any]] = []

        while iterations < max_iterations:
            iterations += 1
            await self._navigate_url(start_url)

            sense = await self.agent_pool.execute_capability('sense', {'browser': self.browser, 'spec': spec})
            if not inferred:
                self._apply_spec_inference(sense.get('features', {}) if isinstance(sense, dict) else {})
                inferred = True

            await self.agent_pool.execute_capability('plan', {'spec': spec, 'page_structure': sense})
            act = await self.agent_pool.execute_capability('act', {'browser': self.browser, 'spec': spec})
            page_items = act.get('extracted_data', []) if isinstance(act, dict) else []
            if isinstance(page_items, list):
                extracted_data.extend(page_items)

            verify = await self.agent_pool.execute_capability(
                'verify', {'extracted_data': page_items, 'spec': spec}
            )
            judge = await self.agent_pool.execute_capability(
                'judge',
                {
                    'quality_score': (verify or {}).get('quality_score', 0.0),
                    'errors': (verify or {}).get('verification_result', {}).get('issues', []),
                    'extracted_data': page_items,
                    'iteration': iterations,
                    'max_iterations': max_iterations,
                    'spec': spec,
                }
            )
            decision = (judge or {}).get('decision', 'complete')

            if decision == 'terminate':
                return {
                    'success': False,
                    'crawl_mode': crawl_mode,
                    'extracted_data': extracted_data,
                    'iterations': iterations,
                    'quality_score': (verify or {}).get('quality_score', 0.0),
                    'error': 'terminated_by_judge',
                }
            if decision == 'complete':
                return {
                    'success': True,
                    'crawl_mode': crawl_mode,
                    'extracted_data': extracted_data,
                    'iterations': iterations,
                    'quality_score': (verify or {}).get('quality_score', 0.0),
                }

        return {
            'success': False,
            'crawl_mode': crawl_mode,
            'extracted_data': extracted_data,
            'iterations': max_iterations,
            'quality_score': 0.0,
            'error': 'max_iterations_reached',
        }

    def _get_next_page_url(self, current_url: str, next_page_num: int, page_result: Dict) -> Optional[str]:
        """从结果中提取下一页 URL 或尝试自动推断"""
        # 如果 page_result 中包含分页信息
        if 'next_url' in page_result and page_result['next_url']:
            return page_result['next_url']

        # 尝试 URL 推断
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

        parsed = urlparse(current_url)
        params = parse_qs(parsed.query, keep_blank_values=True)

        # 页码类参数推断
        page_params = ['page', 'p', 'pageNum', 'pn', 'pg']
        for param in page_params:
            if param in params:
                params[param] = [str(next_page_num)]
                new_query = urlencode(params, doseq=True)
                return urlunparse(parsed._replace(query=new_query))

        # 路径类分页推断
        path = parsed.path
        import re
        page_path_match = re.search(r'(/page/)(\d+)', path)
        if page_path_match:
            new_path = path[:page_path_match.start(2)] + str(next_page_num) + path[page_path_match.end(2):]
            return urlunparse(parsed._replace(path=new_path))

        # 默认情况下返回 None（表示没有更多页面）
        return None

    async def _run_full_site(self, start_url: str, spec: Optional[Dict[str, Any]] = None):
        """全站爬取模式"""
        from .pipeline import run_single_page_pipeline
        from .agents.base import ExploreAgent

        print("启动全站爬取模式...")
        spec = spec or getattr(self, 'spec', {}) or {}
        max_pages = spec.get('max_pages', 200)
        max_depth = spec.get('max_depth', 5)

        # object.__new__ 场景下兜底初始化
        if not hasattr(self, 'frontier') or self.frontier is None:
            self.frontier = CrawlFrontier(base_url=start_url, max_depth=max_depth, max_pages=max_pages)
        else:
            self.frontier.reset()
            self.frontier.base_url = start_url
            self.frontier.max_depth = max_depth
            self.frontier.max_pages = max_pages
        if not hasattr(self, 'meta_controller') or self.meta_controller is None:
            self.meta_controller = MetaController()
        if not hasattr(self, 'risk_monitor') or self.risk_monitor is None:
            self.risk_monitor = RiskMonitor()

        # 初始化爬取前沿
        self.frontier.push(start_url, depth=0, priority=100)

        all_results = []
        visited_urls = set()
        explore_agent = ExploreAgent()
        # 跨页面传递 reflect 改进建议
        reflect_hints = {}

        while len(visited_urls) < max_pages and not self.frontier.is_empty():
            # 获取下一个要处理的 URL
            next_item = self.frontier.pop()
            if not next_item:
                break
            next_url = next_item.url
            next_depth = next_item.depth

            # 规范化 URL 用于去重
            canonical = canonicalize_url(next_url)
            if canonical in visited_urls:
                continue

            # Meta-Controller: 跳过已知失败 URL 模式
            if self.meta_controller.should_skip_url(next_url):
                logger.info(f"[MetaController] 跳过失败模式 URL: {next_url}")
                continue

            visited_urls.add(canonical)

            print(f"处理 URL (深度 {next_depth}): {next_url}")

            try:
                # 访问页面
                await self._navigate_url(next_url)
                if hasattr(self.browser, 'dismiss_popups'):
                    try:
                        await self.browser.dismiss_popups()
                    except Exception:
                        pass
                if hasattr(self.browser, 'smart_scroll'):
                    try:
                        await self.browser.smart_scroll(max_scrolls=6, scroll_delay=0.8, detect_new_content=True)
                    except Exception:
                        pass

                # 运行单页流水线
                llm = getattr(self, 'llm_client', None)
                if not getattr(self, '_llm_available', False):
                    llm = None
                page_result = await run_single_page_pipeline({
                    'start_url': next_url,
                    'spec': spec,
                    'browser': self.browser,
                    'llm_client': llm,
                    'vision_llm_client': getattr(self, 'vision_llm_client', None),
                    'sandbox': getattr(self, 'sandbox', None),
                    'agent_pool': self.agent_pool,
                    'reflect_hints': reflect_hints,
                })

                # 收集结果
                all_results.append({
                    'url': next_url,
                    'result': page_result,
                    'depth': next_depth
                })

                # Meta-Controller: 记录结果 + 自主策略调整
                self.meta_controller.record_outcome(next_url, page_result)
                adjustment = self.meta_controller.evaluate()
                if adjustment:
                    print(f"[MetaController] L{adjustment.level.value} 调整: "
                          f"{adjustment.action} — {adjustment.reason}")
                    overrides = self.meta_controller.get_context_overrides()
                    if overrides.get('hint_selectors'):
                        reflect_hints['suggested_selectors'] = overrides['hint_selectors']
                    if overrides.get('force_llm_plan'):
                        reflect_hints['force_llm_plan'] = True
                self.meta_controller.reset_escalation()

                # 提取 reflect 改进建议传递给下一页
                reflection = page_result.get('reflection_notes', {})
                improvements = reflection.get('improvements', {})
                if improvements.get('action') == 'change_strategy':
                    reflect_hints = {
                        'previous_reasoning': improvements.get('reasoning', ''),
                        'suggested_selectors': improvements.get('selectors', {}),
                        'suggested_strategy': improvements.get('strategy', ''),
                        'alternative_approaches': improvements.get('alternative_approaches', []),
                    }

                # 从结果中提取新链接并添加到前沿
                if page_result.get('success') and 'extracted_data' in page_result:
                    new_links = await self._extract_links_from_result(page_result)
                    for link in new_links:
                        if canonicalize_url(link) not in visited_urls and (next_depth + 1) <= max_depth:
                            self.frontier.push(link, depth=next_depth + 1, priority=50)

                # 兼容旧逻辑：调用 ExploreAgent 探索链接并推入 frontier
                await explore_agent.execute({
                    'browser': self.browser,
                    'current_url': next_url,
                    'base_url': start_url,
                    'depth': next_depth,
                    'max_depth': max_depth,
                    'frontier': self.frontier,
                })

            except Exception as e:
                print(f"处理 URL {next_url} 时发生错误: {e}")
                self.metrics['total_errors'] += 1

        extracted_data = []
        quality_scores = []
        for result in all_results:
            page_result = result.get('result', {})
            page_url = result.get('url', '')
            page_records = page_result.get('extracted_data', [])
            # 确保每条记录都有来源 URL
            for rec in page_records:
                if isinstance(rec, dict) and 'source_url' not in rec:
                    rec['source_url'] = page_url
            extracted_data.extend(page_records)
            # 收集每页的质量分数用于聚合
            page_quality = page_result.get('verification_result', {}).get('quality_score')
            if page_quality is None:
                page_quality = page_result.get('metrics', {}).get('data_quality_score')
            if page_quality is not None and isinstance(page_quality, (int, float)):
                quality_scores.append(float(page_quality))

        # 记录级去重：按内容哈希去除完全相同的记录，并过滤非目标字段记录
        target_fields = set()
        for t in spec.get('targets', []):
            for f in t.get('fields', []):
                target_fields.add(f.get('name', ''))
        extracted_data = self._deduplicate_records(extracted_data, target_fields)

        # 按页面加权平均计算整体质量（无分数时回退到 0.0）
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        return {
            'success': True,
            'crawl_mode': 'full_site',
            'results': all_results,
            'extracted_data': extracted_data,
            'urls_visited': list(visited_urls),
            'pages_visited': len(visited_urls),
            'total_pages_processed': len(visited_urls),
            'frontier_stats': self.frontier.get_stats(),
            'quality_score': round(avg_quality, 4),
            'meta_controller_stats': self.meta_controller.get_stats(),
            'summary': self._summarize_results(all_results)
        }

    async def _extract_links_from_result(self, result: Dict) -> List[str]:
        """从结果中提取新链接"""
        # 实现链接提取逻辑
        links = []
        try:
            # 从提取的数据中查找可能的链接
            extracted_data = result.get('extracted_data', [])
            for item in extracted_data:
                for key, value in item.items():
                    if isinstance(value, str) and (value.startswith('http') or value.startswith('/')):
                        # 简单的链接格式化逻辑
                        links.append(value)
        except Exception as e:
            print(f"提取链接时发生错误: {e}")

        return links

    @staticmethod
    def _deduplicate_records(records: List[Dict], target_fields: set = None) -> List[Dict]:
        """去除完全相同的记录和空/非目标记录，保持原始顺序。"""
        import hashlib
        seen = set()
        unique = []
        for rec in records:
            # 如果指定了目标字段，跳过不包含任何目标字段的记录
            if target_fields:
                has_target = any(k in target_fields for k in rec.keys())
                if not has_target:
                    continue
            # 跳过空记录：所有值为空或缺失
            non_empty_vals = [v for v in rec.values() if v and str(v).strip()]
            if not non_empty_vals:
                continue
            try:
                key = hashlib.md5(
                    json.dumps(rec, sort_keys=True, default=str).encode()
                ).hexdigest()
            except Exception:
                key = str(sorted(rec.items()))
            if key not in seen:
                seen.add(key)
                unique.append(rec)
        return unique

    def _summarize_results(self, results: List[Dict]) -> Dict[str, Any]:
        """汇总结果统计"""
        total_items = 0
        total_errors = 0

        for result in results:
            page_result = result.get('result', {})
            if page_result.get('success'):
                extracted = page_result.get('extracted_data', [])
                total_items += len(extracted)
            else:
                total_errors += 1

        return {
            'total_items_extracted': total_items,
            'total_errors': total_errors,
            'total_pages': len(results),
            'success_rate': (len(results) - total_errors) / len(results) if results else 0
        }

    async def cleanup(self):
        """清理资源"""
        try:
            if getattr(self, 'browser', None) is not None and hasattr(self.browser, 'close'):
                await self.browser.close()
            if getattr(self, 'llm_client', None) is not None and hasattr(self.llm_client, 'close'):
                await self.llm_client.close()
        except Exception as e:
            logger.error(f"清理资源时发生错误: {e}")

    def get_metrics(self):
        """获取性能指标"""
        if self.metrics['end_time']:
            duration = (self.metrics['end_time'] - self.metrics['start_time']).total_seconds()
            self.metrics['duration_seconds'] = duration

        return self.metrics
