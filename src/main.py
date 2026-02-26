"""
主入口 - Full Self-Crawling Agent
完整的迭代循环实现：Sense → Plan → Act → Verify → Judge → Reflect
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件


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
        from src.tools.llm_client import CachedLLMClient
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
        llm_api_base = os.getenv('LLM_API_BASE')

        self.llm_client = CachedLLMClient(
            api_key=api_key,
            model=os.getenv('LLM_MODEL', 'glm-4'),
            api_base=llm_api_base
        ) if api_key else None

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
        from src.core.verifier import Verifier
        from src.agents.base import AgentPool
        from src.tools.browser import BrowserTool

        # 加载 Spec 契约
        self.spec = self.spec_loader.load_spec(str(self.spec_path))
        self.task_id = self.spec['task_id']

        # 创建状态
        self.state = self.state_manager.create_initial_state(self.task_id, self.spec)

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

        print(f"OK 已初始化任务: {self.spec['task_name']} ({self.task_id})")
        if self.llm_client:
            stats = self.llm_client.get_stats()
            print(f"  LLM 客户端: {stats.get('provider', 'unknown')} - {stats['model']}")

    async def run(self) -> Dict[str, Any]:
        """
        运行爬取任务 - 完整迭代循环

        执行流程：
        1. Sense: 感知页面结构
        2. Plan: 规划提取策略
        3. Act: 执行数据提取
        4. Verify: 验证数据质量
        5. Gate: 门禁检查
        6. Judge: 决策是否继续
        7. Reflect: 反思并优化（如需要）
        """
        await self.initialize()

        # 初始化状态
        self.state_manager.update_state({'status': 'running'})

        try:
            # 启动浏览器
            await self.browser.start()

            # 导航到起始 URL
            start_url = self.spec.get('start_url') or self.spec.get('target_url')
            print(f"-> 访问: {start_url}")
            await self.browser.navigate(start_url)

            max_iterations = self.spec.get('max_iterations', 10)
            extracted_data = []
            errors = []

            # ========== 主迭代循环 ==========
            for iteration in range(max_iterations):
                print(f"\n{'='*20} 迭代 {iteration + 1}/{max_iterations} {'='*20}")
                self.state_manager.update_state({
                    'iteration': iteration,
                    'stage': 'sensing'
                })

                iteration_errors = []

                # 1. Sense: 感知页面
                print("-> [1/6] 感知页面结构...")
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
                        print(f"   感知失败: {sense_result.get('error', '未知错误')}")
                        iteration_errors.append(f"sense_error: {sense_result.get('error', '未知')}")

                    self.state_manager.update_state({
                        'html_snapshot': sense_result.get('html_snapshot', '')[:50000],
                        'sense_analysis': sense_result.get('structure', {}),
                        'features': sense_result.get('features', {})
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

                    print(f"   页面类型: {sense_result.get('structure', {}).get('type', 'unknown')}")
                    print(f"   复杂度: {sense_result.get('structure', {}).get('complexity', 'unknown')}")
                    print(f"   反爬检测: {'是' if sense_result.get('anti_bot_detected') else '否'}")

                except Exception as e:
                    print(f"   感知异常: {str(e)}")
                    iteration_errors.append(f"sense_exception: {str(e)}")
                    sense_result = {'success': False, 'structure': {}}

                # 2. Plan: 规划策略
                print("-> [2/6] 规划提取策略...")
                self.state_manager.update_state({'stage': 'planning'})

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
                        print(f"   规划失败: {plan_result.get('error', '未知错误')}")
                        iteration_errors.append(f"plan_error: {plan_result.get('error', '未知')}")

                    self.state_manager.update_state({
                        'generated_code': plan_result.get('generated_code'),
                        'routing_decision': plan_result.get('strategy', {})
                    })

                    # 保存生成的代码
                    if plan_result.get('generated_code'):
                        self.evidence_collector.collect_plan(
                            plan_result['generated_code'],
                            str(plan_result.get('strategy', {}))
                        )

                    print(f"   策略类型: {plan_result.get('strategy', {}).get('strategy_type', 'css')}")
                    print(f"   选择器数量: {len(plan_result.get('selectors', {}))}")

                except Exception as e:
                    print(f"   规划异常: {str(e)}")
                    iteration_errors.append(f"plan_exception: {str(e)}")
                    plan_result = {'success': False, 'selectors': {}, 'strategy': {}}

                # 3. Act: 执行提取
                print("-> [3/6] 执行数据提取...")
                self.state_manager.update_state({'stage': 'acting'})

                try:
                    act_result = await self.agent_pool.execute_capability(
                        'act',
                        {
                            'browser': self.browser,
                            'selectors': plan_result.get('selectors', {}),
                            'strategy': plan_result.get('strategy', {}),
                            'generated_code': plan_result.get('generated_code')
                        }
                    )

                    if not act_result.get('success'):
                        print(f"   执行失败: {act_result.get('error', '未知错误')}")
                        iteration_errors.append(f"act_error: {act_result.get('error', '未知')}")

                    extracted_data = act_result.get('extracted_data', [])

                    self.state_manager.update_state({
                        'sample_data': extracted_data[:10] if extracted_data else [],
                        'execution_result': act_result
                    })

                    print(f"   提取数据: {len(extracted_data)} 条")

                except Exception as e:
                    print(f"   执行异常: {str(e)}")
                    iteration_errors.append(f"act_exception: {str(e)}")
                    extracted_data = []

                # 4. Verify: 验证数据
                print("-> [4/6] 验证数据质量...")
                self.state_manager.update_state({'stage': 'verifying'})

                try:
                    verify_result = await self.agent_pool.execute_capability(
                        'verify',
                        {
                            'extracted_data': extracted_data,
                            'spec': self.spec
                        }
                    )

                    quality_score = verify_result.get('quality_score', 0)

                    self.state_manager.update_state({
                        'quality_score': quality_score,
                        'verification_result': verify_result
                    })

                    print(f"   质量分数: {quality_score:.2f}")
                    print(f"   有效数据: {verify_result.get('valid_items', 0)}/{len(extracted_data)}")
                    if verify_result.get('verification_result', {}).get('issues'):
                        print(f"   问题: {verify_result['verification_result']['issues'][:3]}")

                except Exception as e:
                    print(f"   验证异常: {str(e)}")
                    iteration_errors.append(f"verify_exception: {str(e)}")
                    verify_result = {'quality_score': 0}
                    quality_score = 0

                # 5. Gate: 门禁检查
                print("-> [5/6] 门禁检查...")
                current_state = self.state_manager.get_state()
                gate_passed = self.completion_gate.check(current_state, self.spec)

                if gate_passed:
                    # 成功完成
                    print("   门禁通过!")
                    self.evidence_storage.save_data(extracted_data)
                    return {
                        'success': True,
                        'extracted_data': extracted_data,
                        'verification': verify_result,
                        'iterations': iteration + 1,
                        'quality_score': quality_score
                    }

                print(f"   门禁未通过: {self.completion_gate.get_failed_gates()}")

                # 6. Judge: 决策
                print("-> [6/6] 决策下一步...")
                errors.extend(iteration_errors)
                self.state_manager.update_state({'errors': errors})

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

                    print(f"   决策: {decision}")
                    print(f"   原因: {reasoning}")
                    if suggestions:
                        print(f"   建议: {suggestions[:2]}")

                except Exception as e:
                    print(f"   决策异常: {str(e)}")
                    # 降级决策
                    if quality_score >= 0.5 and iteration < max_iterations - 1:
                        decision = 'reflect_and_retry'
                    else:
                        decision = 'terminate'
                    reasoning = str(e)

                if decision == 'complete':
                    # 任务完成
                    self.evidence_storage.save_data(extracted_data)
                    return {
                        'success': True,
                        'extracted_data': extracted_data,
                        'verification': verify_result,
                        'iterations': iteration + 1,
                        'quality_score': quality_score
                    }

                elif decision == 'reflect_and_retry':
                    # 7. Reflect: 反思并重试
                    print("-> [反思] 分析并优化...")
                    self.state_manager.update_state({'stage': 'reflecting'})

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

                        # 应用改进建议
                        new_selectors = reflect_result.get('new_selectors', {})
                        if new_selectors:
                            plan_result['selectors'].update(new_selectors)
                            print(f"   更新选择器: {list(new_selectors.keys())}")

                        new_strategy = reflect_result.get('new_strategy')
                        if new_strategy:
                            print(f"   新策略: {new_strategy}")

                        self.state_manager.update_state({
                            'last_reflection': reflect_result
                        })

                    except Exception as e:
                        print(f"   反思异常: {str(e)}")

                    # 继续下一次迭代
                    continue

                else:  # terminate
                    print("   任务终止")
                    return {
                        'success': False,
                        'error': reasoning,
                        'extracted_data': extracted_data,
                        'quality_score': quality_score,
                        'iterations': iteration + 1
                    }

            # 超过最大迭代次数
            print(f"\n达到最大迭代次数 {max_iterations}")
            return {
                'success': False,
                'error': '超过最大迭代次数',
                'extracted_data': extracted_data,
                'quality_score': quality_score,
                'iterations': max_iterations
            }

        except Exception as e:
            print(f"X 错误: {str(e)}")
            import traceback
            traceback.print_exc()
            self.state_manager.add_error(str(e))
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

            print("\n" + "="*50)
            print("任务结束")

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

    # 从环境变量或参数读取配置
    api_key = args.api_key or os.getenv('ZHIPU_API_KEY')
    model = args.model or os.getenv('LLM_MODEL', 'glm-4')
    api_base = args.api_base or os.getenv('LLM_API_BASE')

    if not api_key:
        print("! 未提供 API Key，将使用降级模式运行")
        print("  如需使用 AI 功能，设置 ZHIPU_API_KEY 环境变量")

    # 运行
    agent = SelfCrawlingAgent(args.spec_file, api_key)

    try:
        result = asyncio.run(agent.run())

        if result['success']:
            print(f"\nOK 任务成功完成")
            print(f"  - 已提取 {len(result.get('extracted_data', []))} 条数据")
            print(f"  - 迭代次数: {result.get('iterations', 'N/A')}")
            print(f"  - 质量分数: {result.get('quality_score', 0):.2f}")

            if args.debug:
                print(f"\n  详细信息:")
                print(f"  验证结果: {result.get('verification', {})}")
        else:
            print(f"\nX 任务失败")
            print(f"  - 错误: {result.get('error', '未知错误')}")
            if result.get('extracted_data'):
                print(f"  - 部分数据: {len(result['extracted_data'])} 条")

    except KeyboardInterrupt:
        print("\nX 任务被用户中断")

    except Exception as e:
        print(f"\nX 任务失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()