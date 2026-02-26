"""
主入口 - Full Self-Crawling Agent
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

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
        from src.agents.base import AgentPool
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
        self.evidence_collector = EvidenceCollector('./evidence')
        self.verifier = None

        # 执行层
        self.agent_pool = None
        self.executor = Executor()

        # 工具层
        self.browser = BrowserTool(headless=True)
        # 从环境变量读取配置
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
        self.spec: Optional[SpecContract] = None
        self.state: Optional[StateContract] = None
        self.task_id: Optional[str] = None

    async def initialize(self):
        """初始化"""
        # 加载 Spec 契约
        self.spec = self.spec_loader.load_spec(str(self.spec_path))
        self.task_id = self.spec['task_id']

        # 创建状态
        self.state = self.state_manager.create_initial_state(self.task_id, self.spec)

        # 初始化完成门禁
        from src.core.completion_gate import CompletionGate
        self.completion_gate = CompletionGate()

        # 初始化验证器
        from src.core.verifier import Verifier
        self.verifier = Verifier(self.spec)

        # 初始化智能体池
        from src.agents.base import AgentPool
        self.agent_pool = AgentPool(self.llm_client)

        # 创建证据目录
        self.evidence_storage.create_task_dir(self.task_id)

        print(f"OK 已初始化任务: {self.spec['task_name']} ({self.task_id})")
        if self.llm_client:
            stats = self.llm_client.get_stats()
            print(f"✓ LLM 客户端: {stats['provider']} - {stats['model']}")

    async def run(self) -> Dict[str, Any]:
        """
        运行爬取任务

        执行流程：
        1. 感知页面结构
        2. 规划提取策略
        3. 执行提取
        4. 验证数据
        5. 门禁检查
        6. 决策（继续/终止）
        """
        await self.initialize()

        # 记录开始
        self.state_manager.update_state({
            'status': 'running'
        })

        try:
            # 启动浏览器
            await self.browser.start()

            # 导航到起始 URL
            print(f"→ 访问: {self.spec.start_url}")
            await self.browser.navigate(self.spec['start_url'])

            # 感知页面
            print("→ 感知页面结构...")
            sense_result = self.agent_pool.execute_capability(
                'sense',
                {'browser': self.browser, 'spec': self.spec}
            )

            # 规划策略
            print("→ 规划提取策略...")
            page_structure = sense_result.get('structure', {})
            plan_result = self.agent_pool.execute_capability(
                'plan',
                {'page_structure': page_structure, 'spec': self.spec}
            )

            # 执行提取
            print("→ 执行数据提取...")
            selectors = plan_result.get('selectors', {})
            act_result = self.agent_pool.execute_capability(
                'act',
                {'browser': self.browser, 'selectors': selectors, 'spec': self.spec}
            )

            extracted_data = act_result.get('extracted_data', [])

            # 验证数据
            print("→ 验证数据质量...")
            verify_result = self.agent_pool.execute_capability(
                'verify',
                {'extracted_data': extracted_data, 'spec': self.spec}
            )

            # 门禁检查
            print("→ 门禁检查...")
            gate_decision = self.completion_gate.check(
                self.state_manager.get_state(),  # state
                self.spec  # spec
            )

            # 保存证据和数据
            self.evidence_storage.save_data(extracted_data)
            self.evidence_storage.save_log(
                f'提取完成: {len(extracted_data)} 条',
                level='info'
            )

            # 更新状态
            self.state_manager.add_extracted_data(extracted_data)

            # 返回结果
            return {
                'success': True,
                'extracted_data': extracted_data,
                'verification': verify_result,
                'gate_decision': gate_decision,
                'evidence_dir': str(self.evidence_storage.get_task_dir())
            }

        except Exception as e:
            print(f"✗ 错误: {str(e)}")
            self.state_manager.add_error(str(e))
            self.evidence_storage.save_log(str(e), level='error')

            return {
                'success': False,
                'error': str(e)
            }

        finally:
            # 关闭浏览器
            await self.browser.stop()

            # 保存最终状态
            final_state = self.state_manager.get_state()
            self.state_storage.save_state(self.task_id, final_state.to_dict())

            print("✓ 任务完成")

    async def stop(self):
        """停止"""
        await self.browser.stop()
        if self.llm_client:
            await self.llm_client.close()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'task_id': self.task_id,
            'task_name': self.spec.task_name if self.spec else '',
            'llm_stats': self.llm_client.get_stats() if self.llm_client else {},
            'cache_stats': self.llm_client.get_cache_stats() if self.llm_client else {},
            'evidence_summary': self.evidence_storage.get_task_summary(self.task_id)
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
    parser.add_argument('--model', help='模型名称（默认: glm-coding-plan）')
    parser.add_argument('--api-base', help='自定义 API 端点')
    parser.add_argument('--headless', action='store_true', default=True,
                        help='无头模式（默认）')
    parser.add_argument('--debug', action='store_true', help='调试模式')

    args = parser.parse_args()

    # 从环境变量或参数读取配置
    api_key = args.api_key or os.getenv('ZHIPU_API_KEY')
    model = args.model or os.getenv('LLM_MODEL', 'glm-coding-plan')
    api_base = args.api_base or os.getenv('LLM_API_BASE')

    if not api_key:
        print("⚠ 未提供 API Key，将使用降级模式运行")
        print("  如需使用 AI 功能，设置 ZHIPU_API_KEY 环境变量")

    # 运行
    agent = SelfCrawlingAgent(args.spec_file, api_key)

    try:
        result = asyncio.run(agent.run())

        if result['success']:
            print(f"\n✓ 任务成功完成")
            print(f"  - 已提取 {len(result['extracted_data'])} 条数据")
            print(f"  - 证据目录: {result['evidence_dir']}")

            if args.debug:
                print(f"\n  详细信息:")
                print(f"  验证结果: {result['verification']}")
                print(f"  门禁决策: {result['gate_decision']}")
        else:
            print(f"\n✗ 任务失败")
            print(f"  - 错误: {result['error']}")

    except KeyboardInterrupt:
        print("\n✗ 任务被用户中断")

    except Exception as e:
        print(f"\n✗ 任务失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
