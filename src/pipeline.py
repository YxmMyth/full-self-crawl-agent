"""
Pipeline - 单页处理流水线
包含完整的 Sense → Plan → Act → Verify → Judge → Reflect 循环
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .agents.base import AgentCapability

logger = logging.getLogger(__name__)


async def run_single_page_pipeline(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行单页处理流水线
    完整的 Sense → Plan → Act → Verify → Judge → Reflect 循环
    """
    browser = context['browser']
    spec = context['spec']
    agent_pool = context['agent_pool']

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
        # 1. Sense (感知) - 分析页面结构
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

        # 2. Plan (规划) - 规划提取策略
        print("2. 正在规划提取策略...")
        reflect_hints = context.get('reflect_hints', {})
        plan_context = {**pipeline_context,
                        'page_structure': sense_result.get('structure', {}),
                        'sense_features': sense_result.get('features', {}),
                        'reflect_hints': reflect_hints}
        # 如果前一页的 reflect 建议了新选择器，注入到规划上下文
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
        print("   规划完成")

        # 3. Act (执行) - 执行数据提取（含自适应策略选择）
        print("3. 正在执行数据提取...")
        sense_features = sense_result.get('features', {})
        is_spa = sense_features.get('is_spa', False)
        extraction_method = 'standard'

        # SPA 页面：等待动态内容加载后再提取
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

        # 自适应：标准提取结果太少时尝试 SPA 提取
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
            return {'success': False, 'error': f"执行阶段失败: {act_result.get('error')}"}

        pipeline_context['extracted_data'] = act_result.get('extracted_data', [])
        # 为每条记录注入来源 URL
        source_url = context.get('start_url', '')
        for rec in pipeline_context['extracted_data']:
            if isinstance(rec, dict) and 'source_url' not in rec:
                rec['source_url'] = source_url
        pipeline_context['extraction_method'] = extraction_method
        print(f"   提取完成 ({len(act_result.get('extracted_data', []))} 条记录, 方法: {extraction_method})")

        # 4. Verify (验证) - 验证数据质量
        print("4. 正在验证数据质量...")
        verify_context = {**pipeline_context,
                         'extracted_data': act_result.get('extracted_data', []),
                         'expected_fields': spec.get('targets', [])}

        verify_result = await agent_pool.execute_capability(
            AgentCapability.VERIFY,
            verify_context
        )

        pipeline_context['verification_result'] = verify_result
        print("   验证完成")

        # 5. Judge (判断) - 评估当前状态并决定是否继续
        print("5. 正在评估当前状态...")
        verify_issues = verify_result.get('verification_result', {}).get('issues', [])
        judge_context = {**pipeline_context,
                         'verification_result': verify_result,
                         'quality_score': verify_result.get('quality_score', 0.0),
                         'errors': verify_issues,
                         'iteration_complete': True}  # 表示这是单页迭代

        judge_result = await agent_pool.execute_capability(
            AgentCapability.JUDGE,
            judge_context
        )

        pipeline_context['decision'] = judge_result
        print("   评估完成")

        # 6. Reflect (反思) - 总结经验教训
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
