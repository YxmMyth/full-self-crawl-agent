"""
快速端到端测试 - 验证 PlanAgent 重试机制
测试 3 个网站来快速验证新架构
"""

import asyncio
import sys
import os
from datetime import datetime

# 配置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.tools.browser import BrowserTool
from src.tools.multi_llm_client import MultiLLMClient
from src.agents.base import AgentPool
from src.config.loader import load_spec
from src.executors.executor import CrawlExecutor


async def quick_test():
    """快速测试"""
    print("\n🚀 快速端到端测试 - 验证 PlanAgent 重试架构")
    print(f"   测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 测试用例 (选择3个有代表性的网站)
    test_cases = [
        {
            'name': 'TechCrunch (科技新闻)',
            'spec': 'specs/test_sites/site_01_techcrunch.yaml',
            'url': 'https://techcrunch.com/'
        },
        {
            'name': 'arXiv (学术论文) - 目标驱动模式',
            'spec': 'specs/test_sites/site_05_arxiv.yaml',
            'url': 'https://arxiv.org/list/cs/recent'
        },
        {
            'name': 'GitHub Trending',
            'spec': 'specs/test_sites/site_02_github_trending.yaml',
            'url': 'https://github.com/trending'
        },
    ]

    # 初始化
    print("🔧 初始化环境...")
    llm_client = MultiLLMClient()
    browser = BrowserTool()
    print("✅ 环境初始化完成\n")

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] 🧪 测试: {test_case['name']}")
        print(f"   URL: {test_case['url']}")

        start_time = datetime.now()

        try:
            # 加载 Spec
            spec_path = os.path.join(project_root, test_case['spec'])
            spec = load_spec(spec_path)

            if not spec:
                print(f"   ❌ Spec 加载失败")
                results.append({'success': False, 'error': 'Spec加载失败'})
                continue

            print(f"   ✅ Spec: {spec.name}")

            # 执行爬取
            agent_pool = AgentPool(llm_client=llm_client)
            executor = CrawlExecutor(agent_pool=agent_pool, llm_client=llm_client)

            print(f"   🕷️  开始爬取...")
            result = await executor.execute(spec, test_case['url'])

            duration = (datetime.now() - start_time).total_seconds()

            if result and result.get('success'):
                data = result.get('data', [])
                attempts = result.get('plan_attempts', 1)

                print(f"   ✅ 爬取成功! ({duration:.1f}s)")
                print(f"   📊 数据量: {len(data)} 条")
                print(f"   🔄 PlanAgent 尝试次数: {attempts}")

                results.append({
                    'success': True,
                    'data_count': len(data),
                    'attempts': attempts,
                    'duration': duration
                })
            else:
                error = result.get('error', 'Unknown error') if result else 'No result'
                print(f"   ❌ 爬取失败: {error}")
                results.append({'success': False, 'error': error, 'duration': duration})

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            print(f"   ❌ 异常: {e}")
            results.append({'success': False, 'error': str(e), 'duration': duration})

        print()

    # 清理
    await browser.close()

    # 生成报告
    print(f"{'='*80}")
    print("📊 快速测试总结")
    print(f"{'='*80}")

    total = len(results)
    passed = sum(1 for r in results if r.get('success'))
    failed = total - passed

    print(f"\n✅ 通过: {passed}/{total}")
    print(f"❌ 失败: {failed}/{total}")
    print(f"🎯 成功率: {passed/total*100:.1f}%")

    print(f"\n📝 详细结果:")
    for i, (test_case, result) in enumerate(zip(test_cases, results), 1):
        status = "✅" if result.get('success') else "❌"
        name = test_case['name']
        attempts = result.get('attempts', 'N/A')
        data_count = result.get('data_count', 0)
        duration = result.get('duration', 0)

        print(f"  {status} [{i}] {name}")
        if result.get('success'):
            print(f"     重试: {attempts} 次 | 数据: {data_count} 条 | 耗时: {duration:.1f}s")
        else:
            error = result.get('error', 'Unknown')
            print(f"     错误: {error[:80]}")

    print(f"\n{'='*80}\n")

    return passed == total


async def main():
    success = await quick_test()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    asyncio.run(main())
