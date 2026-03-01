"""
端到端测试 - 10个真实网站测试 PlanAgent 重试架构
"""

import asyncio
import sys
import os

# 添加src到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

print(f"Python path: {sys.path}")

from src.tools.multi_llm_client import MultiLLMClient
from src.agents.base import AgentPool
from src.config.loader import load_spec
from src.executors.executor import CrawlExecutor


async def run_test(site_name: str, spec_file: str, url: str):
    """运行单个网站测试"""
    print(f"\n{'='*80}")
    print(f"🧪 测试网站: {site_name}")
    print(f"📄 Spec: {spec_file}")
    print(f"🌐 URL: {url}")
    print(f"{'='*80}")

    try:
        # 1. 初始化 LLM 客户端
        llm_client = MultiLLMClient()

        # 2. 加载 Spec
        spec = load_spec(spec_file)
        if not spec:
            print(f"❌ Spec 加载失败: {spec_file}")
            return False

        print(f"✓ Spec 加载成功: {spec.name}")

        # 3. 初始化智能体池
        agent_pool = AgentPool(llm_client=llm_client)

        # 4. 初始化执行器
        executor = CrawlExecutor(agent_pool=agent_pool, llm_client=llm_client)

        # 5. 执行爬取
        print(f"⏳ 开始爬取...")
        result = await executor.execute(spec, url)

        # 6. 验证结果
        if result and result.get('success'):
            data = result.get('data', [])
            print(f"✅ 爬取成功!")
            print(f"📊 提取数据量: {len(data)} 条")

            if data:
                print(f"🔍 示例数据:")
                sample = data[0]
                for key, value in list(sample.items())[:5]:
                    print(f"   {key}: {value}")

            # 检查 PlanAgent 日志
            if 'plan_attempts' in result:
                print(f"🔄 PlanAgent 重试次数: {result['plan_attempts']}")

            return True
        else:
            error = result.get('error', 'Unknown error') if result else 'No result'
            print(f"❌ 爬取失败: {error}")
            return False

    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """运行所有测试"""
    print("\n🚀 启动端到端测试 - 10个真实网站")
    print("Testing PlanAgent retry architecture...")

    test_cases = [
        {
            'name': 'TechCrunch (科技新闻)',
            'spec': 'specs/test_sites/site_01_techcrunch.yaml',
            'url': 'https://techcrunch.com/'
        },
        {
            'name': 'arXiv (学术论文)',
            'spec': 'specs/test_sites/site_05_arxiv.yaml',
            'url': 'https://arxiv.org/'
        },
        {
            'name': 'GitHub Trending',
            'spec': 'specs/test_sites/site_02_github_trending.yaml',
            'url': 'https://github.com/trending'
        },
        {
            'name': 'Hacker News',
            'spec': 'specs/test_sites/site_03_hackernews.yaml',
            'url': 'https://news.ycombinator.com/'
        },
        {
            'name': 'Reddit r/programming',
            'spec': 'specs/test_sites/site_04_reddit_programming.yaml',
            'url': 'https://www.reddit.com/r/programming/'
        },
        {
            'name': 'Product Hunt',
            'spec': 'specs/test_sites/site_06_product_hunt.yaml',
            'url': 'https://www.producthunt.com/'
        },
        {
            'name': 'Medium 专栏',
            'spec': 'specs/test_sites/site_07_medium.yaml',
            'url': 'https://medium.com/'
        },
        {
            'name': 'Stack Overflow',
            'spec': 'specs/test_sites/site_08_stackoverflow.yaml',
            'url': 'https://stackoverflow.com/'
        },
        {
            'name': 'Lobsters',
            'spec': 'specs/test_sites/site_09_lobsters.yaml',
            'url': 'https://lobste.rs/'
        },
        {
            'name': 'InfoQ',
            'spec': 'specs/test_sites/site_10_infoq.yaml',
            'url': 'https://www.infoq.com/'
        },
    ]

    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] ", end="")
        success = await run_test(
            test_case['name'],
            test_case['spec'],
            test_case['url']
        )
        results.append({
            'index': i,
            'name': test_case['name'],
            'success': success
        })

    # 总结报告
    print(f"\n{'='*80}")
    print("📊 测试总结报告")
    print(f"{'='*80}")

    total = len(results)
    passed = sum(1 for r in results if r['success'])
    failed = total - passed

    print(f"✅ 通过: {passed}/{total}")
    print(f"❌ 失败: {failed}/{total}")
    print(f"🎯 成功率: {passed/total*100:.1f}%")

    print(f"\n📝 详细结果:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"  {status} [{result['index']}] {result['name']}")

    print(f"\n{'='*80}")

    return passed == total


if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
