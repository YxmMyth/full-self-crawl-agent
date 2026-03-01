"""
端到端测试运行器 - 测试 PlanAgent 重试架构
运行 10 个真实网站的完整测试
"""

import asyncio
import sys
import os
from datetime import datetime
import json

# 配置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.tools.browser import BrowserTool
from src.tools.multi_llm_client import MultiLLMClient
from src.agents.base import AgentPool
from src.config.loader import load_spec
from src.executors.executor import CrawlExecutor


class E2ETestRunner:
    """端到端测试运行器"""

    def __init__(self):
        self.results = []
        self.start_time = None
        self.llm_client = None
        self.browser = None

    async def setup(self):
        """初始化测试环境"""
        print("🔧 初始化测试环境...")
        self.llm_client = MultiLLMClient()
        self.browser = BrowserTool()
        self.start_time = datetime.now()
        print("✅ 环境初始化完成\n")

    async def cleanup(self):
        """清理测试环境"""
        print("\n🧹 清理测试环境...")
        if self.browser:
            await self.browser.close()
        print("✅ 环境清理完成")

    async def test_site(self, name: str, spec_path: str, url: str):
        """测试单个网站"""
        print(f"{'='*80}")
        print(f"🧪 [{len(self.results)+1}/10] 测试: {name}")
        print(f"📄 Spec: {spec_path}")
        print(f"🌐 URL: {url}")
        print(f"{'='*80}")

        result = {
            'name': name,
            'spec': spec_path,
            'url': url,
            'start_time': datetime.now().isoformat(),
            'success': False,
            'error': None,
            'data_count': 0,
            'attempts': 0
        }

        try:
            # 1. 加载 Spec
            print("   📖 加载 Spec...")
            spec = load_spec(spec_path)
            if not spec:
                raise Exception(f"Spec 加载失败: {spec_path}")

            print(f"   ✅ Spec: {spec.name}")

            # 2. 初始化执行器
            print("   🤖 初始化智能体...")
            agent_pool = AgentPool(llm_client=self.llm_client)
            executor = CrawlExecutor(agent_pool=agent_pool, llm_client=self.llm_client)

            # 3. 执行爬取
            print("   🕷️  开始爬取...")
            crawl_result = await executor.execute(spec, url)

            # 4. 验证结果
            if crawl_result and crawl_result.get('success'):
                data = crawl_result.get('data', [])
                result['success'] = True
                result['data_count'] = len(data)
                result['attempts'] = crawl_result.get('plan_attempts', 1)

                print(f"   ✅ 爬取成功!")
                print(f"   📊 数据量: {len(data)} 条")
                print(f"   🔄 PlanAgent 尝试次数: {result['attempts']}")

                if data:
                    print(f"   🔍 首条数据预览:")
                    sample = data[0]
                    for key, value in list(sample.items())[:3]:
                        preview = str(value)[:100]
                        print(f"      {key}: {preview}")
            else:
                error = crawl_result.get('error', 'Unknown error') if crawl_result else 'No result'
                raise Exception(f"爬取失败: {error}")

        except Exception as e:
            result['error'] = str(e)
            print(f"   ❌ 测试失败: {e}")

        result['end_time'] = datetime.now().isoformat()
        result['duration'] = (
            datetime.fromisoformat(result['end_time']) -
            datetime.fromisoformat(result['start_time'])
        ).total_seconds()

        self.results.append(result)
        return result['success']

    async def run_all_tests(self):
        """运行所有测试"""
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

        await self.setup()

        print(f"\n🚀 开始端到端测试 - {len(test_cases)} 个网站")
        print(f"   测试时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        for test_case in test_cases:
            await self.test_site(
                test_case['name'],
                os.path.join(project_root, test_case['spec']),
                test_case['url']
            )
            print()

        await self.cleanup()
        return self.generate_report()

    def generate_report(self):
        """生成测试报告"""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()

        total = len(self.results)
        passed = sum(1 for r in self.results if r['success'])
        failed = total - passed

        report = {
            'summary': {
                'total': total,
                'passed': passed,
                'failed': failed,
                'success_rate': round(passed / total * 100, 1) if total > 0 else 0,
                'total_duration_seconds': round(total_duration, 2),
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat()
            },
            'results': self.results
        }

        # 打印报告
        print(f"\n{'='*80}")
        print("📊 端到端测试总结报告")
        print(f"{'='*80}")
        print(f"\n✅ 通过: {passed}/{total}")
        print(f"❌ 失败: {failed}/{total}")
        print(f"🎯 成功率: {report['summary']['success_rate']}%")
        print(f"⏱️  总耗时: {total_duration:.2f} 秒")
        print(f"\n📝 详细结果:")

        for result in self.results:
            status = "✅" if result['success'] else "❌"
            duration = result.get('duration', 0)
            attempts = result.get('attempts', 0)
            data_count = result.get('data_count', 0)

            print(f"\n  {status} {result['name']}")
            print(f"     耗时: {duration:.1f}s | 重试: {attempts} 次 | 数据: {data_count} 条")

            if not result['success']:
                print(f"     错误: {result['error']}")

        # 保存报告
        report_file = os.path.join(project_root, 'reports', 'e2e_test_report.json')
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n📄 详细报告已保存: {report_file}")

        return report['summary']['success_rate'] >= 80  # 80% 成功率视为通过


async def main():
    runner = E2ETestRunner()
    success = await runner.run_all_tests()

    print(f"\n{'='*80}")
    print(f"{'✅ 测试通过' if success else '❌ 测试未通过'}")
    print(f"{'='*80}\n")

    return success


if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
