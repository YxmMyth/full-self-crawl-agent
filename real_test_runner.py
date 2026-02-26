#!/usr/bin/env python3
"""
真实运行测试样例 - 根据test_assets/table.csv创建对应的Spec并运行
"""

import asyncio
import csv
import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime

from src.main import SelfCrawlingAgent


def create_spec_for_test(test_id, description, url, key_elements):
    """为测试创建Spec配置文件"""
    # 根据描述生成相应的spec配置
    targets = []

    if "产品页" in description or "手机" in description or "价格" in description:
        # 电商产品页
        targets = [{
            "name": "products",
            "fields": [
                {"name": "title", "type": "text", "selector": "[data-component='title'], h1, .product-title, .product-name, [data-testid='product-title'], #title, .a-size-large", "required": True, "description": "产品标题"},
                {"name": "price", "type": "number", "selector": "[data-component='price'], .price, .cost, .a-price, .a-offscreen", "required": True, "description": "价格"},
                {"name": "image_url", "type": "url", "selector": "img[data-a-dynamic-image], img[src*='media-amazon'], .a-dynamic-image img, #landingImage", "attribute": "src", "required": False, "description": "图片URL"},
                {"name": "description", "type": "text", "selector": ".a-size-medium, .a-spacing-small, .product-description, #feature-bullets", "required": False, "description": "产品描述"}
            ]
        }]
    elif "新闻文章" in description or "HTML片段" in description:
        # 新闻文章
        targets = [{
            "name": "articles",
            "fields": [
                {"name": "title", "type": "text", "selector": "h1, .article-title, .headline, .post-title", "required": True, "description": "标题"},
                {"name": "content", "type": "html", "selector": ".article-body, .content, .post-content, .entry-content", "required": True, "description": "HTML内容"},
                {"name": "author", "type": "text", "selector": ".author, .byline, .post-author", "required": False, "description": "作者"},
                {"name": "publish_date", "type": "text", "selector": ".date, .publish-date, .post-date", "required": False, "description": "发布日期"}
            ]
        }]
    elif "图表" in description and "SVG" in description:
        # SVG图表
        targets = [{
            "name": "charts",
            "fields": [
                {"name": "svg_code", "type": "text", "selector": "svg", "required": True, "description": "SVG代码"},
                {"name": "title", "type": "text", "selector": ".chart-title, h2, .title, caption", "required": False, "description": "图表标题"},
                {"name": "chart_data", "type": "text", "selector": "[data-chart], script[type='application/json'], .data-wrapper", "required": False, "description": "图表数据"}
            ]
        }]
    elif "招聘" in description:
        # 招聘信息
        targets = [{
            "name": "jobs",
            "fields": [
                {"name": "title", "type": "text", "selector": ".job-title, h1, .position-title, [data-testid='job-title']", "required": True, "description": "职位名称"},
                {"name": "salary", "type": "text", "selector": ".salary, .pay, .compensation, .salary-snippet", "required": False, "description": "薪资"},
                {"name": "company_name", "type": "text", "selector": ".company-name, .employer-name, [data-testid='company-name']", "required": False, "description": "公司名称"},
                {"name": "jd_html", "type": "html", "selector": ".job-description, .description, .job-snippet", "required": False, "description": "职位描述HTML"}
            ]
        }]
    elif "学术论文" in description:
        # 学术论文
        targets = [{
            "name": "papers",
            "fields": [
                {"name": "title", "type": "text", "selector": ".title, h1, .abs-title, .arxiv-title", "required": True, "description": "论文标题"},
                {"name": "abstract", "type": "text", "selector": ".abstract, .abs-abstract, .abstract-text", "required": True, "description": "摘要"},
                {"name": "authors", "type": "text", "selector": ".authors, .abs-authors", "required": False, "description": "作者列表"},
                {"name": "pdf_link", "type": "url", "selector": "a[href*='/pdf'], a[title*='Download'], .abs-button.download-pdf", "attribute": "href", "required": True, "description": "PDF下载链接"}
            ]
        }]
    elif "房地产" in description:
        # 房地产
        targets = [{
            "name": "properties",
            "fields": [
                {"name": "title", "type": "text", "selector": ".property-title, h1, .listing-title, .street-address", "required": True, "description": "房源标题"},
                {"name": "price", "type": "number", "selector": ".price, .cost, .list-price", "required": True, "description": "价格"},
                {"name": "address", "type": "text", "selector": ".address, .addr, .location", "required": False, "description": "地址"},
                {"name": "image_urls", "type": "url", "selector": ".media-container img, .slide img, .main-photo img", "attribute": "src", "required": False, "description": "图片列表"}
            ]
        }]
    elif "菜谱" in description:
        # 菜谱
        targets = [{
            "name": "recipes",
            "fields": [
                {"name": "title", "type": "text", "selector": ".recipe-title, h1, .headline, .title", "required": True, "description": "菜谱标题"},
                {"name": "ingredients", "type": "json", "selector": "[data-ingredients], .ingredients, .ingredient-group", "required": True, "description": "配料JSON"},
                {"name": "instructions", "type": "html", "selector": ".instructions, .steps, .directions", "required": True, "description": "步骤HTML"},
                {"name": "image", "type": "url", "selector": ".recipe-image img, .hero-image img, .lead-media-img", "attribute": "src", "required": True, "description": "成品图片URL"}
            ]
        }]
    elif "股票" in description or "K线" in description:
        # 股票图表
        targets = [{
            "name": "stock_charts",
            "fields": [
                {"name": "symbol", "type": "text", "selector": ".symbol, .ticker", "required": True, "description": "股票代码"},
                {"name": "current_price", "type": "number", "selector": ".price, .current-price, .last-sale", "required": True, "description": "当前价格"},
                {"name": "change", "type": "number", "selector": ".change, .percent-change", "required": False, "description": "涨跌额"},
                {"name": "volume", "type": "number", "selector": ".volume, .trade-volume", "required": False, "description": "成交量"}
            ]
        }]
    elif "博客" in description or "CMS" in description:
        # 博客/CMS
        targets = [{
            "name": "posts",
            "fields": [
                {"name": "title", "type": "text", "selector": "h1, .post-title, .article-title, .entry-title", "required": True, "description": "文章标题"},
                {"name": "content_html", "type": "html", "selector": ".post-content, .article-body, .entry-content", "required": True, "description": "嵌套HTML内容"},
                {"name": "author", "type": "text", "selector": ".author, .byline, .post-author", "required": False, "description": "作者"},
                {"name": "tags", "type": "text", "selector": ".tags, .tag, .post-tag", "required": False, "description": "标签"}
            ]
        }]
    elif "政府" in description or "招标" in description:
        # 政府公告
        targets = [{
            "name": "announcements",
            "fields": [
                {"name": "title", "type": "text", "selector": ".announcement-title, h1, .tender-title, .notice-title", "required": True, "description": "公告标题"},
                {"name": "summary", "type": "text", "selector": ".summary, .brief, .description", "required": False, "description": "摘要"},
                {"name": "deadline", "type": "text", "selector": ".deadline, .closing-date, .due-date", "required": False, "description": "截止日期"},
                {"name": "document_links", "type": "url", "selector": "a[href*='.pdf'], a[download], .document-link", "attribute": "href", "required": False, "description": "文档链接"}
            ]
        }]
    else:
        # 默认模板
        targets = [{
            "name": "data",
            "fields": [
                {"name": "title", "type": "text", "selector": "h1, h2, .title, .headline", "required": True, "description": "标题"},
                {"name": "content", "type": "text", "selector": ".content, .main, .container", "required": True, "description": "内容"}
            ]
        }]

    # 创建spec配置
    spec = {
        "task_id": f"test_{test_id}_{int(datetime.now().timestamp())}",
        "task_name": f"测试样例{test_id}: {description}",
        "created_at": datetime.now().isoformat(),
        "version": "1.0",
        "extraction_type": "single_page",
        "targets": targets,
        "start_url": url,
        "max_pages": 1,
        "depth_limit": 1,
        "validation_rules": {},
        "anti_bot": {
            "random_delay": {"min": 1, "max": 3},
            "user_agent_rotation": True
        },
        "completion_criteria": {
            "min_items": 1,
            "quality_threshold": 0.5
        }
    }

    return spec


async def run_single_test(test_id, description, url, key_elements):
    """运行单个测试"""
    print(f"[{test_id}/10] {description[:60]}...")
    print(f"     URL: {url}")
    print(f"     关键元素: {key_elements[:60]}...")

    # 为该测试创建spec文件
    spec_data = create_spec_for_test(test_id, description, url, key_elements)
    spec_file = f"temp_test_spec_{test_id}.yaml"

    try:
        # 写入spec文件
        with open(spec_file, 'w', encoding='utf-8') as f:
            yaml.dump(spec_data, f, default_flow_style=False, allow_unicode=True)

        print(f"     -> 创建Spec配置: {spec_file}")

        # 创建爬取代理（在无API密钥模式下运行，这将使用降级模式）
        agent = SelfCrawlingAgent(spec_file)

        print(f"     -> 启动爬取代理...")

        # 运行爬取任务
        result = await agent.run()

        if result['success']:
            print(f"     -> [OK] 测试成功完成")
            print(f"        提取数据: {len(result.get('extracted_data', []))} 条")
            return True, result
        else:
            print(f"     -> [FAIL] 测试失败: {result.get('error', '未知错误')}")
            return False, result

    except Exception as e:
        print(f"     -> [FAIL] 异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, {'error': str(e)}

    finally:
        # 清理临时spec文件
        if os.path.exists(spec_file):
            os.remove(spec_file)
            print(f"     -> 清理临时文件: {spec_file}")


async def run_all_tests():
    """运行所有测试"""
    print("="*80)
    print("开始运行10个测试样例需求验证...")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # 读取测试样例
    test_cases = []
    with open('test_assets/table.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

        for i, row in enumerate(rows):
            if i == 0:  # 跳过表头
                continue
            test_id = i
            description = row[1]
            url = row[2]
            key_elements = row[3]

            test_cases.append((test_id, description, url, key_elements))

    print(f"共识别到 {len(test_cases)} 个测试用例")
    print()

    results = []

    for test_id, description, url, key_elements in test_cases:
        success, result = await run_single_test(test_id, description, url, key_elements)
        results.append((test_id, description, success, result))

        # 添加延迟避免过于频繁的请求
        if test_id < len(test_cases):  # 不在最后一个测试后等待
            print(f"     -> 等待2秒后继续下一个测试...")
            await asyncio.sleep(2)
        print()

    # 输出总结
    print("="*80)
    print("测试结果汇总")
    print("="*80)

    successful_tests = 0
    for test_id, description, success, result in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"{status} [{test_id}] {description[:60]}...")

        if not success:
            print(f"      错误: {result.get('error', '未知错误')}")

        if success:
            successful_tests += 1

    print(f"\n总计: {len(results)} 个测试用例，成功 {successful_tests} 个，失败 {len(results)-successful_tests} 个")

    if successful_tests == len(results):
        print(f"\n[PASS] 所有测试用例均已成功验证！")
        print(f"[OK] 项目已成功处理全部10个测试样例需求")
    else:
        print(f"\n[WARN]  {len(results)-successful_tests} 个测试用例失败，需要进一步排查")

    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    return successful_tests == len(results)


async def main():
    """主函数"""
    print("Full Self-Crawling Agent - 10个测试样例需求真实验证")
    print()

    # 检查必要文件
    required_files = [
        'test_assets/table.csv',
        'src/main.py',
        'src/config/loader.py'
    ]

    missing_files = []
    for f in required_files:
        if not os.path.exists(f):
            missing_files.append(f)

    if missing_files:
        print(f"缺少必要文件: {missing_files}")
        return False

    print("环境检查完成 OK")
    print("开始运行真实测试...")
    print()

    success = await run_all_tests()

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)