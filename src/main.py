"""
主入口 - Full Self-Crawling Agent
完整的迭代循环实现：Sense → Plan → Act → Verify → Judge → Reflect

支持三种爬取模式（crawl_mode）：
- single_page：单页爬取（默认，向后兼容）
- multi_page：多页爬取（分页跟随）
- full_site：全站爬取（ExploreAgent + CrawlFrontier）

Spec 自动推断：当 Spec 缺失 crawl_mode / max_pages / max_depth 时，
由 SpecInferrer 根据页面特征自动补全。
"""

import asyncio
import sys
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件

# 导入日志配置
from src.core.logging import setup_logging, get_logger, TaskLogger

# 导入配置验证
from src.config.validator import ConfigValidator, LLMConfigValidator, check_requirements

# 配置日志
setup_logging(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = get_logger('main')


class SelfCrawlingAgent:
    """
    Full Self-Crawling Agent 主类

    五层架构：
    - 战略层：SpecLoader, PolicyManager, CompletionGate
    - 管理层：SmartRouter, StateManager, ContextCompressor, RiskMonitor
    - 验证层：Verifier, EvidenceCollector
    - 执行层：AgentPool, Executor
    - 工具层：BrowserTool, LLMClient

    执行流程：
    Sense → Plan → Act → Verify → Gate → Judge → Reflect
    """

    def __init__(self, spec_path: str, api_key: Optional[str] = None):
        self.spec_path = Path(spec_path)
        self.api_key = api_key
        self.task_logger: Optional[TaskLogger] = None

        # 验证配置
        config_validator = ConfigValidator()
        config_result = config_validator.validate()
        if config_result.warnings:
            for warning in config_result.warnings:
                logger.warning(warning)

        # 延迟导入，避免循环依赖
        from src.config.loader import SpecLoader
        from src.config.contracts import SpecContract, StateContract
        from src.core.policy_manager import PolicyManager
        from src.core.completion_gate import CompletionGate
        from src.core.smart_router import SmartRouter
        from src.core.state_manager import StateManager
        from src.core.risk_monitor import RiskMonitor
        from src.core.context_compressor import ContextCompressor
        from src.core.verifier import EvidenceCollector, Verifier
        from src.agents.base import AgentPool, AgentCapability
        from src.executors.executor import Executor
        from src.tools.browser import BrowserTool
        from src.tools.multi_llm_client import MultiLLMClient
        from src.tools.storage import EvidenceStorage, StateStorage

        # 初始化各层组件

        # 战略层
        self.spec_loader = SpecLoader('./specs')
        self.policy_manager = PolicyManager()
        self.completion_gate = None

        # 管理层
        self.smart_router = SmartRouter()
        self.state_manager = StateManager()
        self.risk_monitor = RiskMonitor()
        self.context_compressor = ContextCompressor()

        # 验证层
        self.evidence_collector = None
        self.verifier = None

        # 执行层
        self.agent_pool = None
        self.executor = Executor()

        # 工具层
        self.browser = None

        # 使用多提供商 LLM 客户端
        try:
            self.llm_client = MultiLLMClient.from_env()
        except ValueError as e:
            logger.warning(f"LLM 客户端初始化失败: {e}")
            self.llm_client = None

        # 存储
        self.evidence_storage = EvidenceStorage()
        self.state_storage = StateStorage()

        # 状态
        self.spec: Optional[Dict] = None
        self.state: Optional[Dict] = None
        self.task_id: Optional[str] = None

    async def initialize(self):
        """初始化"""
        from src.core.completion_gate import CompletionGate
        from src.core.verifier import Verifier, EvidenceCollector
        from src.agents.base import AgentPool
        from src.tools.browser import BrowserTool

        # 加载 Spec 契约
        self.spec = self.spec_loader.load_spec(str(self.spec_path))
        self.task_id = self.spec['task_id']

        # 初始化任务日志器
        self.task_logger = TaskLogger(self.task_id)

        # 创建状态
        self.state = await self.state_manager.create_initial_state(self.task_id, self.spec)

        # 初始化完成门禁
        self.completion_gate = CompletionGate()

        # 初始化验证器
        self.verifier = Verifier(self.spec)

        # 初始化智能体池
        self.agent_pool = AgentPool(self.llm_client)
        self.agent_pool.set_verifier(self.verifier)

        # 初始化浏览器
        headless = self.spec.get('headless', True)
        self.browser = BrowserTool(headless=headless)

        # 初始化证据收集器
        self.evidence_collector = EvidenceCollector(f'./evidence/{self.task_id}')

        # 创建证据目录
        self.evidence_storage.create_task_dir(self.task_id)

        logger.info(f"已初始化任务: {self.spec['task_name']} ({self.task_id})")
        if self.llm_client:
            stats = self.llm_client.get_stats()
            logger.info(f"LLM 客户端: {stats.get('provider', 'unknown')} - {stats['model']}")

    async def run(self) -> Dict[str, Any]:
        """
        运行爬取任务 - 支持三种爬取模式

        crawl_mode 控制流程：
        - single_page: 对单个 URL 执行 Sense→Plan→Act→Verify→Gate→Judge→Reflect 循环
        - multi_page:  同 single_page，但 ActAgent 自动跟随分页，直到无下一页或达上限
        - full_site:   使用 ExploreAgent + CrawlFrontier 广度优先爬取多个页面，
                       每页执行完整的 Sense→Plan→Act→Verify 流程

        Spec 自动推断：
        若 Spec 中未提供 crawl_mode / max_pages / max_depth，在首次感知后
        由 SpecInferrer 根据页面特征自动补全。
        """
        await self.initialize()

        # 初始化状态
        self.state_manager.update_state_sync({'status': 'running'})

        try:
            # 启动浏览器
            await self.browser.start()

            # 导航到起始 URL
            start_url = self.spec.get('start_url') or self.spec.get('target_url')
            logger.info(f"访问: {start_url}")
            await self.browser.navigate(start_url)

            # 解析爬取模式（spec 中尚未自动推断时先用默认值）
            crawl_mode = self.spec.get('crawl_mode', 'single_page')
            logger.info(f"爬取模式: {crawl_mode}")

            if crawl_mode == 'full_site':
                return await self._run_full_site(start_url)
            else:
                # single_page 和 multi_page 共用相同的迭代循环；
                # multi_page 下 ActAgent 内部会跟随分页。
                return await self._run_single_or_multi(start_url, crawl_mode)

        except Exception as e:
            logger.error(f"错误: {str(e)}")
            logger.exception("详细错误信息:")
            self.state_manager.add_error_sync(str(e))
            return {
                'success': False,
                'error': str(e)
            }

        finally:
            # 关闭浏览器
            if self.browser:
                await self.browser.stop()

            # 保存最终状态
            final_state = self.state_manager.get_state()
            self.state_storage.save_state(self.task_id, final_state)

            # 保存证据索引
            if self.evidence_collector:
                self.evidence_collector.save_index()

            logger.info("="*50)
            logger.info("任务结束")

    # ------------------------------------------------------------------
    # 内部循环实现
    # ------------------------------------------------------------------

    async def _run_single_or_multi(
        self, start_url: str, crawl_mode: str
    ) -> Dict[str, Any]:
        """
        single_page / multi_page 模式的主迭代循环。

        执行流程：Sense → Plan → Act → Verify → Gate → Judge → Reflect
        multi_page 模式下 ActAgent 负责自动跟随分页（内部处理），
        此处主循环结构与 single_page 完全相同。
        """
        max_iterations = self.spec.get('max_iterations', 10)
        extracted_data: List[Any] = []
        errors: List[str] = []
        spec_inferred = False

        for iteration in range(max_iterations):
            logger.info(f"{'='*20} 迭代 {iteration + 1}/{max_iterations} {'='*20}")
            self.state_manager.update_state_sync({
                'iteration': iteration,
                'stage': 'sensing'
            })

            iteration_errors: List[str] = []

            # 1. Sense: 感知页面
            logger.info("[1/6] 感知页面结构...")
            try:
                sense_result = await self.agent_pool.execute_capability(
                    'sense',
                    {
                        'browser': self.browser,
                        'spec': self.spec,
                        'llm_client': self.llm_client
                    }
                )

                if not sense_result.get('success'):
                    logger.warning(f"感知失败: {sense_result.get('error', '未知错误')}")
                    iteration_errors.append(f"sense_error: {sense_result.get('error', '未知')}")

                structure = sense_result.get('structure', {})
                features = sense_result.get('features', {})
                self.state_manager.update_state_sync({
                    'html_snapshot': sense_result.get('html_snapshot', '')[:50000],
                    'sense_analysis': structure,
                    'features': features,
                })

                # 保存证据
                if sense_result.get('html_snapshot'):
                    self.evidence_storage.save_html(
                        sense_result['html_snapshot'][:100000],
                        'page_snapshot.html'
                    )
                if sense_result.get('screenshot'):
                    try:
                        import base64
                        img_data = base64.b64decode(sense_result['screenshot'])
                        self.evidence_storage.save_screenshot(img_data, 'page.png')
                    except Exception:
                        pass

                # 统一字段日志（page_type / pagination_type / main_content_selector）
                logger.debug(
                    f"page_type={structure.get('page_type', 'unknown')} "
                    f"pagination_type={structure.get('pagination_type', 'none')} "
                    f"main_content_selector={structure.get('main_content_selector')} "
                    f"complexity={structure.get('complexity', 'unknown')} "
                    f"anti_bot={'yes' if sense_result.get('anti_bot_detected') else 'no'}"
                )

                # Spec 自动推断（仅首次迭代且尚未推断过）
                if not spec_inferred and features:
                    self._apply_spec_inference(features)
                    spec_inferred = True
                    # 更新 crawl_mode（推断后可能已变更）
                    crawl_mode = self.spec.get('crawl_mode', crawl_mode)

            except Exception as e:
                logger.error(f"感知异常: {str(e)}")
                iteration_errors.append(f"sense_exception: {str(e)}")
                sense_result = {'success': False, 'structure': {}, 'features': {}}

            # 2. Plan: 规划策略
            logger.info("[2/6] 规划提取策略...")
            self.state_manager.update_state_sync({'stage': 'planning'})

            try:
                plan_result = await self.agent_pool.execute_capability(
                    'plan',
                    {
                        'page_structure': sense_result.get('structure', {}),
                        'spec': self.spec,
                        'llm_client': self.llm_client
                    }
                )

                if not plan_result.get('success'):
                    logger.warning(f"规划失败: {plan_result.get('error', '未知错误')}")
                    iteration_errors.append(f"plan_error: {plan_result.get('error', '未知')}")

                self.state_manager.update_state_sync({
                    'generated_code': plan_result.get('generated_code'),
                    'routing_decision': plan_result.get('strategy', {})
                })

                if plan_result.get('generated_code'):
                    self.evidence_collector.collect_plan(
                        plan_result['generated_code'],
                        str(plan_result.get('strategy', {}))
                    )

                logger.debug(
                    f"strategy_type={plan_result.get('strategy', {}).get('strategy_type', 'css')} "
                    f"selectors={len(plan_result.get('selectors', {}))}"
                )

            except Exception as e:
                logger.error(f"规划异常: {str(e)}")
                iteration_errors.append(f"plan_exception: {str(e)}")
                plan_result = {'success': False, 'selectors': {}, 'strategy': {}}

            # 3. Act: 执行提取
            logger.info("[3/6] 执行数据提取...")
            self.state_manager.update_state_sync({'stage': 'acting'})

            try:
                act_result = await self.agent_pool.execute_capability(
                    'act',
                    {
                        'browser': self.browser,
                        'selectors': plan_result.get('selectors', {}),
                        'strategy': plan_result.get('strategy', {}),
                        'generated_code': plan_result.get('generated_code'),
                        'crawl_mode': crawl_mode,
                        'max_pages': self.spec.get('max_pages', 1),
                    }
                )

                if not act_result.get('success'):
                    logger.warning(f"执行失败: {act_result.get('error', '未知错误')}")
                    iteration_errors.append(f"act_error: {act_result.get('error', '未知')}")

                extracted_data = act_result.get('extracted_data', [])

                self.state_manager.update_state_sync({
                    'sample_data': extracted_data[:10] if extracted_data else [],
                    'execution_result': act_result
                })

                logger.info(f"提取数据: {len(extracted_data)} 条")

            except Exception as e:
                logger.error(f"执行异常: {str(e)}")
                iteration_errors.append(f"act_exception: {str(e)}")
                extracted_data = []
                act_result = {'extracted_data': [], 'extraction_metrics': {}}

            # 4. Verify: 验证数据
            logger.info("[4/6] 验证数据质量...")
            self.state_manager.update_state_sync({'stage': 'verifying'})
            verify_result: Dict[str, Any] = {'quality_score': 0}
            quality_score = 0.0

            try:
                verify_result = await self.agent_pool.execute_capability(
                    'verify',
                    {
                        'extracted_data': extracted_data,
                        'spec': self.spec,
                        'extraction_metrics': act_result.get('extraction_metrics', {})
                    }
                )

                quality_score = verify_result.get('quality_score', 0)

                self.state_manager.update_state_sync({
                    'quality_score': quality_score,
                    'verification_result': verify_result
                })

                logger.info(
                    f"质量分数: {quality_score:.2f} "
                    f"有效数据: {verify_result.get('valid_items', 0)}/{len(extracted_data)}"
                )
                if verify_result.get('verification_result', {}).get('issues'):
                    logger.warning(f"问题: {verify_result['verification_result']['issues'][:3]}")

            except Exception as e:
                logger.error(f"验证异常: {str(e)}")
                iteration_errors.append(f"verify_exception: {str(e)}")

            # 5. Gate: 门禁检查
            logger.info("[5/6] 门禁检查...")
            current_state = self.state_manager.get_state()

            risk_metrics = {
                'iteration_count': iteration + 1,
                'consecutive_errors': len(iteration_errors),
                'total_items': len(extracted_data),
                'failed_items': len([e for e in iteration_errors if 'error' in e]),
                'quality_score': quality_score,
            }
            risk_alerts = self.risk_monitor.check_metrics(risk_metrics)
            if risk_alerts:
                from src.core.risk_monitor import RiskLevel
                critical = [a for a in risk_alerts
                            if a.level in (RiskLevel.HIGH, RiskLevel.CRITICAL)]
                if critical:
                    logger.warning(f"风险监控告警: {[a.message for a in critical]}")

            gate_passed = self.completion_gate.check(current_state, self.spec)

            if gate_passed:
                logger.info("门禁通过!")
                self.evidence_storage.save_data(extracted_data)
                return {
                    'success': True,
                    'crawl_mode': crawl_mode,
                    'extracted_data': extracted_data,
                    'verification': verify_result,
                    'iterations': iteration + 1,
                    'quality_score': quality_score,
                }

            logger.warning(f"门禁未通过: {self.completion_gate.get_failed_gates()}")

            # 6. Judge: 决策
            logger.info("[6/6] 决策下一步...")
            errors.extend(iteration_errors)
            self.state_manager.update_state_sync({'errors': errors})

            try:
                judge_result = await self.agent_pool.execute_capability(
                    'judge',
                    {
                        'quality_score': quality_score,
                        'iteration': iteration,
                        'max_iterations': max_iterations,
                        'errors': errors,
                        'spec': self.spec,
                        'llm_client': self.llm_client,
                        'extracted_data': extracted_data
                    }
                )

                decision = judge_result.get('decision', 'terminate')
                reasoning = judge_result.get('reasoning', '')
                suggestions = judge_result.get('suggestions', [])

                logger.info(f"决策: {decision}")
                logger.debug(f"原因: {reasoning}")
                if suggestions:
                    logger.info(f"建议: {suggestions[:2]}")

            except Exception as e:
                logger.error(f"决策异常: {str(e)}")
                decision = 'reflect_and_retry' if (
                    quality_score >= 0.5 and iteration < max_iterations - 1
                ) else 'terminate'
                reasoning = str(e)

            if decision == 'complete':
                self.evidence_storage.save_data(extracted_data)
                return {
                    'success': True,
                    'crawl_mode': crawl_mode,
                    'extracted_data': extracted_data,
                    'verification': verify_result,
                    'iterations': iteration + 1,
                    'quality_score': quality_score,
                }

            elif decision == 'reflect_and_retry':
                # 7. Reflect: 反思并重试
                logger.info("[反思] 分析并优化...")
                self.state_manager.update_state_sync({'stage': 'reflecting'})

                try:
                    reflect_result = await self.agent_pool.execute_capability(
                        'reflect',
                        {
                            'execution_history': self.state_manager.get_history(),
                            'errors': errors,
                            'quality_score': quality_score,
                            'spec': self.spec,
                            'llm_client': self.llm_client
                        }
                    )

                    new_selectors = reflect_result.get('new_selectors', {})
                    if new_selectors:
                        plan_result['selectors'].update(new_selectors)
                        logger.info(f"更新选择器: {list(new_selectors.keys())}")

                    new_strategy = reflect_result.get('new_strategy')
                    if new_strategy:
                        logger.info(f"新策略: {new_strategy}")

                    self.state_manager.update_state_sync({'last_reflection': reflect_result})

                except Exception as e:
                    logger.error(f"反思异常: {str(e)}")

                continue

            else:  # terminate
                logger.warning("任务终止")
                return {
                    'success': False,
                    'crawl_mode': crawl_mode,
                    'error': reasoning,
                    'extracted_data': extracted_data,
                    'quality_score': quality_score,
                    'iterations': iteration + 1,
                }

        # 超过最大迭代次数
        logger.warning(f"达到最大迭代次数 {max_iterations}")
        return {
            'success': False,
            'crawl_mode': crawl_mode,
            'error': '超过最大迭代次数',
            'extracted_data': extracted_data,
            'quality_score': quality_score,
            'iterations': max_iterations,
        }

    async def _run_full_site(self, start_url: str) -> Dict[str, Any]:
        """
        full_site 模式：使用 ExploreAgent + CrawlFrontier 广度优先爬取。

        流程：
        1. 初始化 CrawlFrontier，将 start_url 入队
        2. 循环：pop URL → navigate → Sense → Plan → Act → Verify →
                 Explore（发现子链接推入 Frontier）
        3. 直到 Frontier 为空、达到 max_pages 或 max_depth 上限
        """
        from src.core.crawl_frontier import CrawlFrontier
        from src.agents.base import ExploreAgent

        max_pages = self.spec.get('max_pages', 100)
        max_depth = self.spec.get('max_depth', 3)
        url_patterns = self.spec.get('url_patterns') or []

        frontier = CrawlFrontier(
            base_url=start_url,
            max_depth=max_depth,
            max_pages=max_pages,
            url_patterns=url_patterns,
            same_domain_only=True,
        )
        frontier.push(start_url, depth=0)

        explore_agent = ExploreAgent()
        all_extracted: List[Any] = []
        pages_visited = 0
        spec_inferred = False

        logger.info(
            f"[full_site] max_pages={max_pages} max_depth={max_depth} "
            f"url_patterns={url_patterns}"
        )

        while not frontier.is_empty() and pages_visited < max_pages:
            item = frontier.pop()
            if item is None:
                break

            current_url = item.url
            current_depth = item.depth
            frontier.mark_visited(current_url)
            pages_visited += 1

            logger.info(
                f"[full_site] 页面 {pages_visited}/{max_pages} "
                f"depth={current_depth} url={current_url}"
            )
            self.state_manager.update_state_sync({
                'stage': 'navigating',
                'current_url': current_url,
                'pages_visited': pages_visited,
            })

            try:
                await self.browser.navigate(current_url)
            except Exception as e:
                logger.warning(f"导航失败 {current_url}: {e}")
                continue

            # Sense
            try:
                sense_result = await self.agent_pool.execute_capability(
                    'sense',
                    {
                        'browser': self.browser,
                        'spec': self.spec,
                        'llm_client': self.llm_client
                    }
                )
                structure = sense_result.get('structure', {})
                features = sense_result.get('features', {})

                logger.debug(
                    f"page_type={structure.get('page_type', 'unknown')} "
                    f"pagination_type={structure.get('pagination_type', 'none')} "
                    f"main_content_selector={structure.get('main_content_selector')} "
                    f"url={current_url}"
                )

                # Spec 自动推断（仅首页且尚未推断）
                if not spec_inferred and features:
                    self._apply_spec_inference(features)
                    spec_inferred = True

            except Exception as e:
                logger.warning(f"感知失败 {current_url}: {e}")
                sense_result = {'success': False, 'structure': {}, 'features': {}}
                structure = {}

            # Plan
            try:
                plan_result = await self.agent_pool.execute_capability(
                    'plan',
                    {
                        'page_structure': structure,
                        'spec': self.spec,
                        'llm_client': self.llm_client
                    }
                )
            except Exception as e:
                logger.warning(f"规划失败 {current_url}: {e}")
                plan_result = {'success': False, 'selectors': {}, 'strategy': {}}

            # Act
            try:
                act_result = await self.agent_pool.execute_capability(
                    'act',
                    {
                        'browser': self.browser,
                        'selectors': plan_result.get('selectors', {}),
                        'strategy': plan_result.get('strategy', {}),
                        'generated_code': plan_result.get('generated_code'),
                        'crawl_mode': 'full_site',
                        'max_pages': 1,
                    }
                )
                page_data = act_result.get('extracted_data', [])
                all_extracted.extend(page_data)
                logger.info(
                    f"[full_site] 提取 {len(page_data)} 条（累计 {len(all_extracted)} 条）"
                )
            except Exception as e:
                logger.warning(f"提取失败 {current_url}: {e}")

            # Explore（发现子链接 → 推入 Frontier）
            if current_depth < max_depth:
                try:
                    await explore_agent.execute({
                        'browser': self.browser,
                        'current_url': current_url,
                        'base_url': start_url,
                        'depth': current_depth,
                        'max_depth': max_depth,
                        'frontier': frontier,
                    })
                    logger.debug(
                        f"[full_site] frontier queue={frontier.queue_size()} "
                        f"visited={frontier.visited_count()}"
                    )
                except Exception as e:
                    logger.warning(f"探索失败 {current_url}: {e}")

        stats = frontier.get_stats()
        logger.info(
            f"[full_site] 完成: pages_visited={pages_visited} "
            f"total_extracted={len(all_extracted)} "
            f"frontier_stats={stats}"
        )

        self.evidence_storage.save_data(all_extracted)
        return {
            'success': True,
            'crawl_mode': 'full_site',
            'extracted_data': all_extracted,
            'pages_visited': pages_visited,
            'frontier_stats': stats,
            'quality_score': 1.0 if all_extracted else 0.0,
        }

    # ------------------------------------------------------------------
    # Spec 自动推断
    # ------------------------------------------------------------------

    def _apply_spec_inference(self, features: Dict[str, Any]) -> None:
        """
        使用 SpecInferrer 将推断字段合并到 self.spec（仅补充缺失字段）。
        """
        from src.core.spec_inferrer import SpecInferrer
        inferrer = SpecInferrer()
        patch = inferrer.infer(features, existing_spec=self.spec)
        if patch:
            self.spec.update(patch)
            logger.info(f"Spec 自动推断补充字段: {list(patch.keys())}")

    async def stop(self):
        """停止"""
        if self.browser:
            await self.browser.stop()
        if self.llm_client:
            await self.llm_client.close()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'task_id': self.task_id,
            'task_name': self.spec.get('task_name', '') if self.spec else '',
            'llm_stats': self.llm_client.get_stats() if self.llm_client else {},
            'cache_stats': self.llm_client.get_cache_stats() if self.llm_client else {},
            'evidence_summary': self.evidence_storage.get_task_summary(self.task_id) if self.task_id else {}
        }


# ==================== CLI 入口 ====================

def main():
    """CLI 入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Full Self-Crawling Agent - 智能网页数据爬取'
    )
    parser.add_argument('spec_file', help='Spec 契约文件路径')
    parser.add_argument('--api-key', help='智谱 API Key')
    parser.add_argument('--model', help='模型名称（默认: glm-4）')
    parser.add_argument('--api-base', help='自定义 API 端点')
    parser.add_argument('--headless', action='store_true', default=True,
                        help='无头模式（默认）')
    parser.add_argument('--debug', action='store_true', help='调试模式')

    args = parser.parse_args()

    # 设置调试日志
    if args.debug:
        setup_logging(level='DEBUG')

    # 多 LLM 提供商客户端现在从环境变量自动加载
    # 支持 DEEPSEEK_API_KEY (推理任务) 和 ZHIPU_API_KEY (编码任务)

    # 运行
    agent = SelfCrawlingAgent(args.spec_file)

    try:
        result = asyncio.run(agent.run())

        if result['success']:
            logger.info("任务成功完成")
            logger.info(f"已提取 {len(result.get('extracted_data', []))} 条数据")
            logger.info(f"迭代次数: {result.get('iterations', 'N/A')}")
            logger.info(f"质量分数: {result.get('quality_score', 0):.2f}")

            if args.debug:
                logger.debug(f"详细信息: {result.get('verification', {})}")
        else:
            logger.error(f"任务失败: {result.get('error', '未知错误')}")
            if result.get('extracted_data'):
                logger.info(f"部分数据: {len(result['extracted_data'])} 条")

    except KeyboardInterrupt:
        logger.warning("任务被用户中断")

    except Exception as e:
        logger.error(f"任务失败: {str(e)}")
        logger.exception("详细错误信息:")


if __name__ == '__main__':
    main()