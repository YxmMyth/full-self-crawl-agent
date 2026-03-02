"""
Pipeline - 单页处理流水线
包含完整的 Sense → Plan → Act → Verify → Judge → Reflect 循环
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .agents.base import AgentCapability
from .core.smart_router import SmartRouter
from .tools.vision_browser import VisionBrowser, PageBlocker

logger = logging.getLogger(__name__)


async def run_single_page_pipeline(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行单页处理流水线
    完整的 Sense → Route → [Plan → Act → Verify → Judge → Reflect]* 循环
    内循环支持 intra-page retry：Judge 说 retry → Reflect → 重跑 Plan→Act→Verify
    """
    browser = context['browser']
    spec = context['spec']
    agent_pool = context['agent_pool']
    max_retries = spec.get('max_page_retries', 3)

    # 初始化上下文
    pipeline_context = {
        'start_url': context['start_url'],
        'spec': spec,
        'browser': browser,
        'llm_client': context.get('llm_client'),
        'sandbox': context.get('sandbox'),
        'agent_pool': agent_pool,
        'reflect_hints': context.get('reflect_hints', {}),

        # 流水线中间结果
        'page_structure': {},
        'selectors': {},
        'strategy': {},
        'extracted_data': [],
        'verification_result': {},
        'decision': {},
        'reflection_notes': {},
    }

    try:
        # 1. Sense (感知) - 分析页面结构（只做一次）
        print("1. 正在感知页面结构...")
        sense_result = await agent_pool.execute_capability(
            AgentCapability.SENSE,
            {**pipeline_context, 'stage': 'sense'}
        )

        if not sense_result.get('success'):
            return {'success': False, 'error': f"感知阶段失败: {sense_result.get('error')}"}

        pipeline_context['page_structure'] = sense_result.get('structure', {})
        pipeline_context['sense_features'] = sense_result.get('features', {})
        pipeline_context['html_snapshot'] = sense_result.get('html_snapshot', '')
        print("   感知完成")

        # 1.5 SmartRouter 路由决策（三层：程序→规则→LLM）
        routing_decision = {}
        try:
            print("1.5 正在进行智能路由...")
            smart_router = SmartRouter(llm_client=context.get('llm_client'))
            routing_decision = await smart_router.route(
                url=context['start_url'],
                goal=spec.get('description', spec.get('goal', 'data_extraction')),
                html=pipeline_context['html_snapshot'],
                use_llm=bool(context.get('llm_client'))
            )
            pipeline_context['routing_decision'] = routing_decision
            print(f"   路由完成: 策略={routing_decision.get('strategy', '?')}, "
                  f"复杂度={routing_decision.get('complexity', '?')}, "
                  f"页面类型={routing_decision.get('page_type', '?')}")
        except Exception as e:
            logger.debug(f"SmartRouter 路由失败（降级为默认流程）: {e}")
            routing_decision = {}

        # 1.8 Vision-LLM 页面阻断检测与自主处理
        try:
            vision = VisionBrowser(browser, llm_client=context.get('llm_client'))
            visual_analysis = await vision.analyze_page(use_vision_llm=bool(context.get('llm_client')))
            pipeline_context['visual_analysis'] = {
                'page_type': visual_analysis.page_type,
                'blocker': visual_analysis.blocker.value,
                'data_regions': len(visual_analysis.data_regions),
                'has_meaningful_content': visual_analysis.has_meaningful_content,
                'analysis_source': visual_analysis.analysis_source,
            }
            if visual_analysis.blocker != PageBlocker.NONE:
                print(f"   ⚠ 检测到页面阻断: {visual_analysis.blocker.value}")
                handled = await vision.handle_blocker(visual_analysis.blocker)
                if handled:
                    print(f"   ✓ 已自主处理: {visual_analysis.blocker.value}")
                else:
                    print(f"   ✗ 无法绕过: {visual_analysis.blocker.value}，继续尝试提取")
        except Exception as e:
            logger.debug(f"Vision 分析失败（降级为标准流程）: {e}")

        # 内循环：Plan → Act → Verify → Judge → Reflect，支持重试
        reflect_hints = context.get('reflect_hints', {})
        best_result = None
        best_quality = -1.0

        for attempt in range(max_retries):
            iteration_label = f"[尝试 {attempt + 1}/{max_retries}] " if max_retries > 1 else ""

            # 2. Plan (规划)
            print(f"{iteration_label}2. 正在规划提取策略...")
            plan_context = {**pipeline_context,
                            'page_structure': sense_result.get('structure', {}),
                            'sense_features': sense_result.get('features', {}),
                            'reflect_hints': reflect_hints,
                            'routing_decision': routing_decision}
            if routing_decision:
                plan_context['routing_guidance'] = {
                    'strategy': routing_decision.get('strategy', ''),
                    'page_type': routing_decision.get('page_type', ''),
                    'complexity': routing_decision.get('complexity', ''),
                    'special_requirements': routing_decision.get('special_requirements', []),
                    'execution_params': routing_decision.get('execution_params', {}),
                }
            if reflect_hints.get('suggested_selectors'):
                plan_context['previous_selectors'] = reflect_hints['suggested_selectors']
            if reflect_hints.get('previous_reasoning'):
                plan_context['previous_reflect_reasoning'] = reflect_hints['previous_reasoning']
            plan_result = await agent_pool.execute_capability(
                AgentCapability.PLAN,
                plan_context
            )

            if not plan_result.get('success'):
                return {'success': False, 'error': f"规划阶段失败: {plan_result.get('error')}"}

            pipeline_context.update({
                'selectors': plan_result.get('selectors', {}),
                'strategy': plan_result.get('strategy', {}),
                'generated_code': plan_result.get('generated_code', '')
            })
            print(f"   规划完成")

            # 3. Act (执行)
            print(f"{iteration_label}3. 正在执行数据提取...")
            sense_features = sense_result.get('features', {})
            is_spa = sense_features.get('is_spa', False)
            extraction_method = 'standard'

            if is_spa and hasattr(browser, 'wait_for_content'):
                try:
                    container_sel = plan_result.get('strategy', {}).get('container_selector')
                    await browser.wait_for_content(
                        min_elements=2,
                        container_selector=container_sel,
                        timeout=8000
                    )
                except Exception:
                    pass

            act_context = {**pipeline_context,
                          'selectors': plan_result.get('selectors', {}),
                          'strategy': plan_result.get('strategy', {}),
                          'generated_code': plan_result.get('generated_code', ''),
                          'crawl_mode': spec.get('crawl_mode', 'single_page'),
                          'max_pages': spec.get('max_pages', 1),
                          'sense_features': sense_features,
                          'html_snapshot': sense_result.get('html_snapshot', '')}

            act_result = await agent_pool.execute_capability(
                AgentCapability.ACT,
                act_context
            )

            # SPA 自适应回退
            std_data = act_result.get('extracted_data', []) if act_result.get('success') else []
            if is_spa and len(std_data) < 3:
                try:
                    print("   SPA 检测到且标准提取不足，尝试 SPA 提取...")
                    spa_result = await run_spa_extraction({
                        'browser': browser,
                        'spec': spec,
                        'llm_client': context.get('llm_client'),
                    })
                    spa_data = spa_result.get('extracted_data', [])
                    if len(spa_data) > len(std_data):
                        act_result = spa_result
                        extraction_method = spa_result.get('method_used', 'spa')
                        print(f"   SPA 提取更优 ({len(spa_data)} 条)")
                except Exception as e:
                    logger.debug(f"SPA 提取回退失败: {e}")

            if not act_result.get('success'):
                if attempt < max_retries - 1:
                    print(f"   执行阶段失败，准备重试...")
                    reflect_hints = {'previous_reasoning': f"Act 失败: {act_result.get('error')}"}
                    continue
                return {'success': False, 'error': f"执行阶段失败: {act_result.get('error')}"}

            pipeline_context['extracted_data'] = act_result.get('extracted_data', [])
            source_url = context.get('start_url', '')
            for rec in pipeline_context['extracted_data']:
                if isinstance(rec, dict) and 'source_url' not in rec:
                    rec['source_url'] = source_url
            pipeline_context['extraction_method'] = extraction_method
            print(f"   提取完成 ({len(act_result.get('extracted_data', []))} 条记录, 方法: {extraction_method})")

            # 4. Verify (验证)
            print(f"{iteration_label}4. 正在验证数据质量...")
            verify_context = {**pipeline_context,
                             'extracted_data': act_result.get('extracted_data', []),
                             'expected_fields': spec.get('targets', [])}

            verify_result = await agent_pool.execute_capability(
                AgentCapability.VERIFY,
                verify_context
            )

            pipeline_context['verification_result'] = verify_result
            cur_quality = verify_result.get('quality_score', 0.0)
            print(f"   验证完成 (质量: {cur_quality:.2f})")

            # 跟踪最佳结果
            if cur_quality > best_quality:
                best_quality = cur_quality
                best_result = {
                    'extracted_data': list(pipeline_context['extracted_data']),
                    'verification_result': dict(verify_result),
                    'selectors': dict(pipeline_context['selectors']),
                    'strategy': dict(pipeline_context['strategy']),
                    'extraction_method': extraction_method,
                }

            # 5. Judge (判断)
            print(f"{iteration_label}5. 正在评估当前状态...")
            verify_issues = verify_result.get('verification_result', {}).get('issues', [])
            judge_context = {**pipeline_context,
                             'verification_result': verify_result,
                             'quality_score': cur_quality,
                             'errors': verify_issues,
                             'iteration': attempt,
                             'max_iterations': max_retries,
                             'iteration_complete': True}

            judge_result = await agent_pool.execute_capability(
                AgentCapability.JUDGE,
                judge_context
            )

            pipeline_context['decision'] = judge_result
            decision = judge_result.get('decision', 'complete')
            print(f"   评估完成: {decision}")

            # 如果 Judge 说完成或终止，跳出重试循环
            if decision in ('complete', 'terminate'):
                break

            # 6. Reflect (反思) — 只在需要重试时执行
            if attempt < max_retries - 1 and decision == 'reflect_and_retry':
                print(f"{iteration_label}6. 正在反思优化策略...")
                reflect_context = {**pipeline_context,
                                  'previous_decision': judge_result,
                                  'errors': verify_issues,
                                  'quality_score': cur_quality,
                                  'html_snapshot': sense_result.get('html_snapshot', ''),
                                  'experience_log': []}

                reflect_result = await agent_pool.execute_capability(
                    AgentCapability.REFLECT,
                    reflect_context
                )

                pipeline_context['reflection_notes'] = reflect_result

                # 提取 Reflect 的改进建议，注入到下次 Plan 的 hints 中
                improvements = reflect_result.get('improvements', {})
                reflect_hints = {
                    'previous_reasoning': improvements.get('reasoning', ''),
                    'suggested_selectors': improvements.get('selectors') or reflect_result.get('new_selectors', {}),
                    'suggested_strategy': improvements.get('strategy', ''),
                    'alternative_approaches': improvements.get('alternative_approaches', []),
                }
                print(f"   反思完成，将改进注入下次规划")
                continue

        # 循环结束后，如果没有 Reflect（complete/terminate），补做一次 Reflect 收尾
        if 'reflection_notes' not in pipeline_context or not pipeline_context['reflection_notes']:
            print("6. 正在反思优化策略...")
            reflect_context = {**pipeline_context,
                              'previous_decision': judge_result,
                              'experience_log': []}
            reflect_result = await agent_pool.execute_capability(
                AgentCapability.REFLECT,
                reflect_context
            )
            pipeline_context['reflection_notes'] = reflect_result
            print("   反思完成")

        # 使用最佳结果（可能不是最后一次尝试的）
        if best_result and best_quality > verify_result.get('quality_score', 0.0):
            pipeline_context['extracted_data'] = best_result['extracted_data']
            pipeline_context['verification_result'] = best_result['verification_result']
            pipeline_context['selectors'] = best_result['selectors']
            pipeline_context['strategy'] = best_result['strategy']
            pipeline_context['extraction_method'] = best_result['extraction_method']
            verify_result = best_result['verification_result']

        # 返回完整结果
        return {
            'success': True,
            'page_structure': pipeline_context['page_structure'],
            'selectors': pipeline_context['selectors'],
            'strategy': pipeline_context['strategy'],
            'extracted_data': pipeline_context['extracted_data'],
            'verification_result': pipeline_context['verification_result'],
            'decision': pipeline_context['decision'],
            'reflection_notes': pipeline_context['reflection_notes'],
            'metrics': {
                'total_items_extracted': len(pipeline_context['extracted_data']),
                'data_quality_score': verify_result.get('quality_score', 0.0),
                'extraction_method': pipeline_context.get('extraction_method', 'standard'),
                'retry_attempts': attempt + 1,
                'routing_strategy': routing_decision.get('strategy', 'default'),
                'routing_complexity': routing_decision.get('complexity', 'unknown'),
                'processing_time': datetime.now().timestamp()
            }
        }

    except Exception as e:
        error_msg = f"单页流水线执行失败: {str(e)}"
        logger.error(error_msg)
        return {'success': False, 'error': error_msg}


async def run_spa_extraction(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    专门针对SPA页面的提取流程
    """
    browser = context['browser']
    spec = context['spec']

    try:
        # 导入SPA处理器
        from .agents.spa_handler import SPAHandler
        spa_handler = SPAHandler(context.get('llm_client'))

        # 开始拦截API响应
        spa_handler.start_intercept(browser.page)

        # 等待页面加载
        await browser.page.wait_for_load_state('networkidle')
        await browser.page.wait_for_timeout(2000)  # 额外等待时间确保JS执行完成

        # 获取API数据
        api_data = spa_handler.get_best_list_data()
        # 统一为 dict 格式以便后续判断
        api_data = {'list_data': api_data} if api_data else None

        # 如果没有通过拦截获得数据，尝试DOM提取
        if not api_data or not api_data.get('list_data'):
            html = await browser.get_html()
            dom_data = spa_handler.extract_from_dom(html)
            return {
                'success': True,
                'extracted_data': dom_data,
                'method_used': 'dom_extraction',
                'api_data_available': bool(api_data)
            }
        else:
            return {
                'success': True,
                'extracted_data': api_data['list_data'],
                'method_used': 'api_interception',
                'api_details': api_data
            }

    except Exception as e:
        return {
            'success': False,
            'error': f"SPA提取失败: {str(e)}",
            'extracted_data': []
        }
