#!/usr/bin/env python3
"""
测试脚本：使用10个真实测试样例验证重构后的项目
根据 test_assets/table.csv 中的数据运行测试
"""

import asyncio
import csv
import os
import sys
from pathlib import Path
import shutil
from src.main import SelfCrawlingAgent

async def run_single_test_case(test_id, description, url, key_elements):
    """运行单个测试用例"""
    print(f"\n{'='*60}")
    print(f"测试用例 {test_id}: {description}")
    print(f"URL: {url}")
    print(f"关键元素: {key_elements}")
    print(f"{'='*60}")

    # 根据描述生成对应的spec文件
    spec_content = generate_spec_for_description(test_id, description, url)

    # 创建临时spec文件
    spec_file = f"temp_spec_{test_id}.yaml"
    with open(spec_file, 'w', encoding='utf-8') as f:
        f.write(spec_content)

    try:
        # 创建代理实例（无API密钥，降级模式运行）
        agent = SelfCrawlingAgent(spec_file)

        # 运行测试
        result = await agent.run()

        print(f"\n结果: {'成功' if result['success'] else '失败'}")
        if not result['success']:
            print(f"错误: {result.get('error', '未知错误')}")
        else:
            print(f"提取数据: {len(result.get('extracted_data', []))} 条")

        return result

    except Exception as e:
        print(f"异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

    finally:
        # 清理临时文件
        if os.path.exists(spec_file):
            os.remove(spec_file)


def generate_spec_for_description(test_id, description, url):
    """根据测试描述生成相应的spec内容"""

    # 根据描述确定要提取的目标字段
    targets = []

    if "产品页" in description or "手机" in description or "价格" in description:
        # 电商产品页
        targets = [{
            "name": "products",
            "fields": [
                {"name": "title", "type": "text", "selector": "[data-component='title'], h1, .product-title, .product-name", "required": True, "description": "产品标题"},
                {"name": "price", "type": "number", "selector": "[data-component='price'], .price, .cost, .amount", "required": True, "description": "价格"},
                {"name": "image_url", "type": "url", "selector": "img[src], .product-image img", "attribute": "src", "required": False, "description": "图片URL"},
                {"name": "description", "type": "text", "selector": ".description, .product-desc, .detail", "required": False, "description": "产品描述"}
            ]
        }]
    elif "新闻文章" in description or "HTML片段" in description:
        # 新闻文章
        targets = [{
            "name": "articles",
            "fields": [
                {"name": "title", "type": "text", "selector": "h1, .article-title, .headline", "required": True, "description": "标题"},
                {"name": "content", "type": "html", "selector": ".article-body, .content, .post-content", "required": True, "description": "HTML内容"},
                {"name": "images", "type": "url", "selector": ".article-body img, .content img", "attribute": "src", "required": False, "description": "图片URL列表"},
                {"name": "videos", "type": "url", "selector": ".article-body video, .content video, iframe", "attribute": "src", "required": False, "description": "视频URL列表"}
            ]
        }]
    elif "图表" in description and "SVG" in description:
        # SVG图表
        targets = [{
            "name": "charts",
            "fields": [
                {"name": "svg_code", "type": "text", "selector": "svg", "required": True, "description": "SVG代码"},
                {"name": "title", "type": "text", "selector": ".chart-title, h2, .title", "required": False, "description": "图表标题"},
                {"name": "chart_data", "type": "text", "selector": "[data-chart], script[type='application/json']", "required": False, "description": "图表数据"}
            ]
        }]
    elif "招聘" in description:
        # 招聘信息
        targets = [{
            "name": "jobs",
            "fields": [
                {"name": "title", "type": "text", "selector": ".job-title, h1, .position-title", "required": True, "description": "职位名称"},
                {"name": "salary", "type": "text", "selector": ".salary, .pay, .compensation", "required": False, "description": "薪资"},
                {"name": "company_logo", "type": "url", "selector": ".company-logo img, .employer-logo img", "attribute": "src", "required": False, "description": "公司Logo URL"},
                {"name": "jd_html", "type": "html", "selector": ".job-description, .description", "required": False, "description": "职位描述HTML"}
            ]
        }]
    elif "学术论文" in description:
        # 学术论文
        targets = [{
            "name": "papers",
            "fields": [
                {"name": "title", "type": "text", "selector": ".paper-title, h1, .title", "required": True, "description": "论文标题"},
                {"name": "abstract", "type": "text", "selector": ".abstract, .summary", "required": True, "description": "摘要"},
                {"name": "pdf_link", "type": "url", "selector": "a[href*='.pdf'], .pdf-link", "attribute": "href", "required": True, "description": "PDF下载链接"}
            ]
        }]
    elif "房地产" in description:
        # 房地产
        targets = [{
            "name": "properties",
            "fields": [
                {"name": "title", "type": "text", "selector": ".property-title, h1, .listing-title", "required": True, "description": "房源标题"},
                {"name": "price", "type": "number", "selector": ".price, .cost", "required": True, "description": "价格"},
                {"name": "floor_plan_svg", "type": "text", "selector": "svg.floor-plan, svg.layout", "required": False, "description": "户型图SVG"},
                {"name": "floor_plan_pdf", "type": "url", "selector": "a[href*='.pdf'].floor-plan", "attribute": "href", "required": False, "description": "户型图PDF链接"}
            ]
        }]
    elif "菜谱" in description:
        # 菜谱
        targets = [{
            "name": "recipes",
            "fields": [
                {"name": "title", "type": "text", "selector": ".recipe-title, h1, .title", "required": True, "description": "菜谱标题"},
                {"name": "ingredients", "type": "json", "selector": "[data-ingredients], .ingredients", "required": True, "description": "配料JSON"},
                {"name": "instructions", "type": "html", "selector": ".instructions, .steps", "required": True, "description": "步骤HTML"},
                {"name": "image", "type": "url", "selector": ".recipe-image img, .hero-image img", "attribute": "src", "required": True, "description": "成品图片URL"}
            ]
        }]
    elif "股票" in description or "K线" in description:
        # 股票图表
        targets = [{
            "name": "stock_charts",
            "fields": [
                {"name": "svg_chart", "type": "text", "selector": "svg.chart, svg.stock-chart", "required": True, "description": "K线SVG图表"},
                {"name": "current_price", "type": "number", "selector": ".price, .current-price", "required": True, "description": "当前价格"},
                {"name": "volume", "type": "number", "selector": ".volume, .trade-volume", "required": False, "description": "成交量"}
            ]
        }]
    elif "博客" in description or "CMS" in description:
        # 博客/CMS
        targets = [{
            "name": "posts",
            "fields": [
                {"name": "title", "type": "text", "selector": "h1, .post-title, .article-title", "required": True, "description": "文章标题"},
                {"name": "content_html", "type": "html", "selector": ".post-content, .article-body", "required": True, "description": "嵌套HTML内容"},
                {"name": "embedded_images", "type": "url", "selector": ".post-content img, .article-body img", "attribute": "src", "required": False, "description": "嵌入图片URL"}
            ]
        }]
    elif "政府" in description or "招标" in description:
        # 政府公告
        targets = [{
            "name": "announcements",
            "fields": [
                {"name": "title", "type": "text", "selector": ".announcement-title, h1, .tender-title", "required": True, "description": "公告标题"},
                {"name": "pdf_links", "type": "url", "selector": "a[href*='.pdf'], .document-link", "attribute": "href", "required": False, "description": "PDF链接"},
                {"name": "table_html", "type": "html", "selector": ".data-table, table", "required": False, "description": "HTML表格"}
            ]
        }]
    else:
        # 默认模板
        targets = [{
            "name": "data",
            "fields": [
                {"name": "title", "type": "text", "selector": "h1, h2, .title", "required": True, "description": "标题"},
                {"name": "content", "type": "text", "selector": ".content, .main", "required": True, "description": "内容"}
            ]
        }]

    # 创建spec
    spec = {
        "task_id": f"test_{test_id}",
        "task_name": f"测试样例 {test_id}: {description}",
        "created_at": "2026-02-26T00:00:00",
        "version": "1.0",
        "extraction_type": "single_page",
        "targets": targets,
        "start_url": url,
        "max_pages": 1,
        "depth_limit": 1,
        "validation_rules": {},
        "anti_bot": {
            "random_delay": {"min": 1, "max": 2},
            "user_agent_rotation": True
        },
        "completion_criteria": {
            "min_items": 1,
            "quality_threshold": 0.5
        }
    }

    # 转换为YAML格式
    import yaml
    return yaml.dump(spec, default_flow_style=False, allow_unicode=True)


async def main():
    """主函数：运行所有测试用例"""
    print("开始运行10个测试样例验证...")

    # 读取测试用例
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

    results = []

    # 逐个运行测试用例
    for test_id, description, url, key_elements in test_cases:
        result = await run_single_test_case(test_id, description, url, key_elements)
        results.append((test_id, description, result))

        # 暂停一会儿避免过于频繁请求
        await asyncio.sleep(2)

    # 输出总结
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")

    successful_tests = 0
    for test_id, description, result in results:
        status = "OK 成功" if result['success'] else "FAIL 失败"
        print(f"{test_id}. {description[:30]}... - {status}")
        if not result['success']:
            print(f"   错误: {result.get('error', '未知错误')}")
        else:
            successful_tests += 1

    print(f"\n总计: {len(results)} 个测试用例，成功 {successful_tests} 个，失败 {len(results)-successful_tests} 个")

    return successful_tests == len(results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)