"""
简化版端到端测试 - 验证 PlanAgent 重试架构
使用 arXiv Spec，但只测试 Plan 阶段
"""

import asyncio
import sys
import os
import logging

# 配置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


async def test_plan_agent_retry():
    """测试 PlanAgent 重试机制"""
    logger.info("="*80)
    logger.info("🧪 开始端到端测试 - PlanAgent 重试机制")
    logger.info("="*80)
    logger.info("")

    # 1. 加载 Spec
    logger.info("1️⃣ 步骤 1: 加载 arXiv Spec...")
    spec_path = os.path.join(project_root, 'specs/test_sites/site_05_arxiv.yaml')

    from src.config.loader import SpecLoader
    loader = SpecLoader('./specs')
    spec = loader.load_spec(spec_path)
    logger.info(f"   ✅ Spec 加载成功: {spec['task_name']}")
    logger.info("")

    # 2. 初始化 LLM 客户端
    logger.info("2️⃣ 步骤 2: 初始化 LLM 客户端...")
    from src.tools.multi_llm_client import MultiLLMClient
    llm_client = MultiLLMClient.from_env()
    stats = llm_client.get_stats()
    logger.info(f"   ✅ LLM 客户端初始化成功")
    logger.info(f"      Provider: {stats.get('provider', 'N/A')}")
    logger.info(f"      Model: {stats.get('model', 'N/A')}")
    logger.info("")

    # 3. 初始化 Browser 并获取页面
    logger.info("3️⃣ 步骤 3: 启动浏览器并访问 arXiv...")
    from src.tools.browser import BrowserTool
    browser = BrowserTool(headless=True)

    await browser.start()
    await browser.navigate('https://arxiv.org/list/cs/recent')

    # 获取 HTML
    html = await browser.get_html()
    logger.info(f"   ✅ 页面加载成功")
    logger.info(f"      HTML 长度: {len(html)} 字符")
    logger.info("")

    # 4. 模拟 SenseAgent 结果
    logger.info("4️⃣ 步骤 4: 模拟页面感知结果...")

    # 使用 LLM 分析页面结构
    from src.agents.base import SenseAgent
    sense_agent = SenseAgent(llm_client=llm_client)

    sense_result = await sense_agent.execute({
        'browser': browser,
        'llm_client': llm_client,
        'html_snapshot': html[:100000]  # 只取前100k
    })

    logger.info(f"   ✅ 页面感知完成")
    logger.info(f"      页面类型: {sense_result.get('structure', {}).get('type', 'unknown')}")
    logger.info(f"      复杂度: {sense_result.get('structure', {}).get('complexity', 'unknown')}")
    logger.info("")

    # 5. 执行 PlanAgent (重试机制在此触发!)
    logger.info("5️⃣ 步骤 5: 执行 PlanAgent (触发重试机制!)...")

    from src.agents.base import PlanAgent
    plan_agent = PlanAgent(llm_client=llm_client)

    start_time = asyncio.get_event_loop().time()

    plan_result = await plan_agent.execute({
        'page_structure': sense_result.get('structure', {}),
        'spec': spec,
        'llm_client': llm_client,
        'html_snapshot': html[:200000]
    })

    duration = asyncio.get_event_loop().time() - start_time

    logger.info("")
    logger.info("🎉 PlanAgent 执行完成!")
    logger.info(f"   ⏱️  耗时: {duration:.2f} 秒")
    logger.info(f"   ✅ 结果: {'成功' if plan_result.get('success') else '失败'}")

    if plan_result.get('success'):
        selectors = plan_result.get('selectors', {})
        logger.info(f"   📊 生成的选择器数量: {len(selectors)}")
        logger.info(f"   🎯 选择器列表:")
        for name, selector in selectors.items():
            logger.info(f"      - {name}: {selector}")

        code_preview = plan_result.get('generated_code', '')[:200]
        logger.info(f"   💻 生成代码预览: {code_preview}...")
    else:
        logger.error(f"   ❌ 错误: {plan_result.get('error', '未知错误')}")

    logger.info("")

    # 6. 清理
    logger.info("6️⃣ 步骤 6: 清理资源...")
    await browser.stop()
    logger.info("   ✅ 资源清理完成")
    logger.info("")

    # 7. 总结
    logger.info("="*80)
    logger.info("📊 测试总结")
    logger.info("="*80)
    logger.info(f"✅ 测试状态: {'通过' if plan_result.get('success') else '失败'}")
    logger.info(f"⏱️  总耗时: {duration:.2f} 秒")
    logger.info(f"🎯 选择器数量: {len(plan_result.get('selectors', {}))}")
    logger.info("="*80)

    return plan_result.get('success')


async def main():
    """主函数"""
    success = await test_plan_agent_retry()

    print("")
    if success:
        print("✅ 测试通过!")
    else:
        print("❌ 测试失败!")

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    asyncio.run(main())
