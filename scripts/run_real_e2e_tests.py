"""
真实网站端到端测试脚本
批量运行 10 个网站的爬取测试，生成测试报告
"""

import asyncio
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# 导入主入口
from src.main import SelfCrawlingAgent


# 测试网站配置
TEST_SITES = [
    {
        "id": "site_01_amazon",
        "name": "Amazon 智能手机",
        "url": "https://www.amazon.com/s?k=smartphone",
        "spec_file": "specs/test_sites/site_01_amazon.yaml",
        "description": "电商产品页：名称/价格/规格/主图URL",
        "expected_fields": ["name", "price", "image_url", "product_url"],
    },
    {
        "id": "site_02_techcrunch",
        "name": "TechCrunch 科技新闻",
        "url": "https://techcrunch.com/",
        "spec_file": "specs/test_sites/site_02_techcrunch.yaml",
        "description": "新闻文章：标题/摘要/图片/视频",
        "expected_fields": ["title", "article_url", "image_url"],
    },
    {
        "id": "site_03_datawrapper",
        "name": "DataWrapper 图表库",
        "url": "https://www.datawrapper.de/",
        "spec_file": "specs/test_sites/site_03_datawrapper.yaml",
        "description": "数据可视化：SVG图表代码",
        "expected_fields": ["title", "svg_content"],
    },
    {
        "id": "site_04_indeed",
        "name": "Indeed 招聘",
        "url": "https://www.indeed.com/hire/job-description/software-engineer",
        "spec_file": "specs/test_sites/site_04_indeed.yaml",
        "description": "招聘信息：职位/薪资/公司Logo/JD HTML",
        "expected_fields": ["title", "company", "description"],
    },
    {
        "id": "site_05_arxiv",
        "name": "arXiv 学术论文",
        "url": "https://arxiv.org/list/cs/recent",
        "spec_file": "specs/test_sites/site_05_arxiv.yaml",
        "description": "学术论文：PDF下载/摘要/作者",
        "expected_fields": ["title", "authors", "pdf_url"],
    },
    {
        "id": "site_06_zillow",
        "name": "Zillow 房地产",
        "url": "https://www.zillow.com/",
        "spec_file": "specs/test_sites/site_06_zillow.yaml",
        "description": "房地产：户型图/价格/地址",
        "expected_fields": ["address", "price", "listing_url"],
    },
    {
        "id": "site_07_allrecipes",
        "name": "AllRecipes 菜谱",
        "url": "https://www.allrecipes.com/",
        "spec_file": "specs/test_sites/site_07_allrecipes.yaml",
        "description": "菜谱网站：配料JSON/步骤HTML/成品图",
        "expected_fields": ["title", "ingredients", "instructions", "image_url"],
    },
    {
        "id": "site_08_yahoo_finance",
        "name": "Yahoo Finance 股票",
        "url": "https://finance.yahoo.com/quote/AAPL/chart",
        "spec_file": "specs/test_sites/site_08_yahoo_finance.yaml",
        "description": "股票图表：K线SVG/实时价格/成交量",
        "expected_fields": ["symbol", "current_price"],
    },
    {
        "id": "site_09_medium",
        "name": "Medium 博客",
        "url": "https://medium.com/",
        "spec_file": "specs/test_sites/site_09_medium.yaml",
        "description": "博客文章：嵌套HTML+图片",
        "expected_fields": ["title", "content", "image_urls"],
    },
    {
        "id": "site_10_gov_tender",
        "name": "UK Government 招标",
        "url": "https://www.find-tender.service.gov.uk/",
        "spec_file": "specs/test_sites/site_10_gov_tender.yaml",
        "description": "政府公告：PDF+HTML表格",
        "expected_fields": ["title", "organization", "pdf_url"],
    },
]


class E2ETestRunner:
    """端到端测试运行器"""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.api_key = os.getenv("ZHIPU_API_KEY")

    async def run_single_site(self, site: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个网站的测试"""
        print(f"\n{'='*60}")
        print(f"测试站点: {site['name']}")
        print(f"URL: {site['url']}")
        print(f"目标: {site['description']}")
        print(f"{'='*60}")

        result = {
            "site_id": site["id"],
            "site_name": site["name"],
            "url": site["url"],
            "description": site["description"],
            "expected_fields": site["expected_fields"],
            "start_time": datetime.now().isoformat(),
            "status": "pending",
            "success": False,
            "extracted_count": 0,
            "quality_score": 0.0,
            "iterations": 0,
            "error": None,
            "sample_data": [],
            "extracted_fields": [],
            "missing_fields": [],
            "execution_time_seconds": 0,
        }

        start = time.time()

        try:
            # 检查 spec 文件是否存在
            spec_path = Path(site["spec_file"])
            if not spec_path.exists():
                raise FileNotFoundError(f"Spec 文件不存在: {spec_path}")

            # 创建 Agent 并运行
            agent = SelfCrawlingAgent(str(spec_path), self.api_key)
            run_result = await agent.run()

            # 收集结果
            result["status"] = "completed"
            result["success"] = run_result.get("success", False)
            result["extracted_count"] = len(run_result.get("extracted_data", []))
            result["quality_score"] = run_result.get("quality_score", 0)
            result["iterations"] = run_result.get("iterations", 0)
            result["error"] = run_result.get("error")

            # 分析提取的字段
            extracted_data = run_result.get("extracted_data", [])
            if extracted_data:
                result["sample_data"] = extracted_data[:3]  # 保存前3条作为样本

                # 检查哪些字段被成功提取
                all_keys = set()
                for item in extracted_data:
                    if isinstance(item, dict):
                        all_keys.update(item.keys())

                result["extracted_fields"] = list(all_keys)
                result["missing_fields"] = [
                    f for f in site["expected_fields"] if f not in all_keys
                ]

            print(f"\n结果: {'成功' if result['success'] else '失败'}")
            print(f"提取数据: {result['extracted_count']} 条")
            print(f"质量分数: {result['quality_score']:.2f}")
            print(f"迭代次数: {result['iterations']}")
            if result["missing_fields"]:
                print(f"缺失字段: {result['missing_fields']}")

        except FileNotFoundError as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"错误: {e}")

        except Exception as e:
            result["status"] = "error"
            result["error"] = f"{type(e).__name__}: {str(e)}"
            print(f"异常: {e}")
            import traceback
            traceback.print_exc()

        finally:
            result["execution_time_seconds"] = round(time.time() - start, 2)
            result["end_time"] = datetime.now().isoformat()

        return result

    async def run_all_sites(
        self,
        sites: Optional[List[Dict[str, Any]]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """运行所有网站测试"""
        self.start_time = datetime.now()
        sites_to_run = sites or TEST_SITES

        if limit:
            sites_to_run = sites_to_run[:limit]
            print(f"限制测试数量: {limit} 个站点")

        print(f"\n开始端到端测试")
        print(f"测试站点数: {len(sites_to_run)}")
        print(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        for i, site in enumerate(sites_to_run, 1):
            print(f"\n[{i}/{len(sites_to_run)}] ", end="")
            result = await self.run_single_site(site)
            self.results.append(result)

            # 站点间延迟，避免触发反爬
            if i < len(sites_to_run):
                delay = 3
                print(f"\n等待 {delay} 秒后继续下一个站点...")
                await asyncio.sleep(delay)

        return self.results

    def generate_report(self) -> str:
        """生成 Markdown 测试报告"""
        if not self.results:
            return "无测试结果"

        # 统计
        total = len(self.results)
        success_count = sum(1 for r in self.results if r["success"])
        failed_count = total - success_count
        total_data = sum(r["extracted_count"] for r in self.results)
        avg_quality = sum(r["quality_score"] for r in self.results) / total if total > 0 else 0
        total_time = sum(r["execution_time_seconds"] for r in self.results)

        # 生成报告
        report = f"""# 真实网站端到端测试报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 测试概览

| 指标 | 数值 |
|------|------|
| 测试站点总数 | {total} |
| 成功数 | {success_count} |
| 失败数 | {failed_count} |
| 成功率 | {(success_count/total*100):.1f}% |
| 总提取数据量 | {total_data} 条 |
| 平均质量分数 | {avg_quality:.2f} |
| 总执行时间 | {total_time:.1f} 秒 |

---

## 详细结果

"""

        # 每个站点的详细结果
        for result in self.results:
            status_icon = "✅" if result["success"] else "❌"
            report += f"""### {status_icon} {result['site_name']}

| 属性 | 值 |
|------|-----|
| URL | `{result['url']}` |
| 状态 | {result['status']} |
| 成功 | {'是' if result['success'] else '否'} |
| 提取数据量 | {result['extracted_count']} 条 |
| 质量分数 | {result['quality_score']:.2f} |
| 迭代次数 | {result['iterations']} |
| 执行时间 | {result['execution_time_seconds']} 秒 |

**数据需求**: {result['description']}

**期望字段**: {', '.join(result['expected_fields'])}
"""

            if result['extracted_fields']:
                report += f"\n**已提取字段**: {', '.join(result['extracted_fields'])}\n"

            if result['missing_fields']:
                report += f"\n**缺失字段**: {', '.join(result['missing_fields'])}\n"

            if result['error']:
                report += f"\n**错误信息**: `{result['error']}`\n"

            # 样本数据
            if result['sample_data']:
                report += "\n**数据样本**:\n```json\n"
                report += json.dumps(result['sample_data'][0], ensure_ascii=False, indent=2)[:500]
                if len(json.dumps(result['sample_data'][0], ensure_ascii=False)) > 500:
                    report += "\n... (截断)"
                report += "\n```\n"

            report += "\n---\n\n"

        # 问题总结
        report += """## 问题总结

"""
        errors = [r for r in self.results if r['error']]
        if errors:
            for e in errors:
                report += f"- **{e['site_name']}**: {e['error']}\n"
        else:
            report += "无严重错误。\n"

        # 建议
        report += """
## 改进建议

"""
        low_quality = [r for r in self.results if r['quality_score'] < 0.5 and r['success']]
        if low_quality:
            report += "### 低质量分数站点\n"
            for r in low_quality:
                report += f"- {r['site_name']}: 质量分数 {r['quality_score']:.2f}\n"
                if r['missing_fields']:
                    report += f"  - 缺失字段: {', '.join(r['missing_fields'])}\n"

        report += """
## 测试环境

- Python 版本: {python_version}
- 操作系统: {os}
- API Key: {api_key_status}
""".format(
            python_version=sys.version.split()[0],
            os=sys.platform,
            api_key_status="已配置" if self.api_key else "未配置"
        )

        return report

    def save_report(self, filename: str = "e2e_real_test_report.md"):
        """保存报告到文件"""
        report = self.generate_report()
        report_path = self.output_dir / filename

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\n报告已保存: {report_path}")
        return report_path

    def save_results_json(self, filename: str = "e2e_test_results.json"):
        """保存原始结果 JSON"""
        results_path = self.output_dir / filename

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": datetime.now().isoformat(),
                "total_sites": len(self.results),
                "results": self.results
            }, f, ensure_ascii=False, indent=2)

        print(f"JSON 结果已保存: {results_path}")
        return results_path


async def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(description="真实网站端到端测试")
    parser.add_argument("--limit", type=int, help="限制测试站点数量")
    parser.add_argument("--site", type=str, help="只测试指定站点ID (如 site_01_amazon)")
    parser.add_argument("--no-report", action="store_true", help="不生成报告")

    args = parser.parse_args()

    runner = E2ETestRunner()

    # 确定要运行的站点
    sites = TEST_SITES
    if args.site:
        sites = [s for s in TEST_SITES if s["id"] == args.site]
        if not sites:
            print(f"未找到站点: {args.site}")
            print(f"可用站点: {[s['id'] for s in TEST_SITES]}")
            return

    # 运行测试
    await runner.run_all_sites(sites=sites, limit=args.limit)

    # 保存结果
    if not args.no_report:
        runner.save_report()
        runner.save_results_json()

    # 打印总结
    success = sum(1 for r in runner.results if r["success"])
    print(f"\n{'='*60}")
    print(f"测试完成: {success}/{len(runner.results)} 成功")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())