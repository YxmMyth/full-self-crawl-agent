"""
快速开始示例 - 使用 Full Self-Crawling Agent
"""

import asyncio
import os
from src.main import SelfCrawlingAgent


async def example_basic():
    """基础示例"""
    print("=" * 60)
    print("📋 快速开始示例")
    print("=" * 60)

    # 1. 创建 Agent
    api_key = os.getenv('ZHIPU_API_KEY', 'your_api_key_here')
    
    agent = SelfCrawlingAgent(
        spec_path='specs/example_ecommerce.json',
        api_key=api_key
    )

    try:
        # 2. 运行任务
        print("\n🚀 正在运行爬虫任务...")
        result = await agent.run()

        # 3. 处理结果
        if result['success']:
            print(f"\n✅ 任务成功!")
            print(f"📊 已提取 {len(result['extracted_data'])} 条数据")
            print(f"📂 证据目录: {result['evidence_dir']}")

            # 显示部分数据
            print("\n🔍 部分数据预览:")
            for i, item in enumerate(result['extracted_data'][:3]):
                print(f"\n  [{i+1}] {item}")

        else:
            print(f"\n❌ 任务失败: {result['error']}")

    finally:
        # 4. 关闭
        await agent.stop()


async def example_custom():
    """自定义示例 - 使用自己的配置"""
    print("\n" + "=" * 60)
    print("🛠️  自定义示例")
    print("=" * 60)

    # 自定义 Spec 配置
    custom_spec = {
        "version": "v1",
        "freeze": True,
        "created_at": "2026-02-25T12:00:00",
        "updated_at": "2026-02-25T12:00:00",
        "task_id": "custom_task_001",
        "task_name": "自定义任务",
        "goal": "爬取自定义页面",
        "target_url": "https://example.com",
        "max_execution_time": 300,
        "max_retries": 3,
        "max_iterations": 10,
        "targets": [
            {
                "name": "items",
                "fields": [
                    {
                        "name": "title",
                        "type": "text",
                        "selector": ".item-title",
                        "required": True
                    },
                    {
                        "name": "price",
                        "type": "number",
                        "selector": ".item-price",
                        "required": True
                    }
                ]
            }
        ],
        "completion_gate": [
            "html_snapshot_exists",
            "sense_analysis_valid",
            "execution_success"
        ],
        "evidence": {
            "required": ["spec.json", "extracted_data.json"],
            "optional": []
        },
        "capabilities": ["sense", "plan", "act", "verify"],
        "start_url": "https://example.com",
        "max_pages": 3,
        "depth_limit": 1
    }

    # 保存到临时文件
    import json
    with open('specs/custom_temp.json', 'w', encoding='utf-8') as f:
        json.dump(custom_spec, f, indent=2, ensure_ascii=False)

    # 运行
    api_key = os.getenv('ZHIPU_API_KEY', 'your_api_key_here')
    agent = SelfCrawlingAgent(
        spec_path='specs/custom_temp.json',
        api_key=api_key
    )

    try:
        result = await agent.run()

        if result['success']:
            print(f"\n✅ 自定义任务成功!")
            print(f"📊 已提取 {len(result['extracted_data'])} 条数据")
        else:
            print(f"\n❌ 自定义任务失败: {result['error']}")

    finally:
        await agent.stop()


async def example_with_debug():
    """调试示例 - 获取详细统计"""
    print("\n" + "=" * 60)
    print("🔍 调试示例")
    print("=" * 60)

    api_key = os.getenv('ZHIPU_API_KEY', 'your_api_key_here')
    agent = SelfCrawlingAgent(
        spec_path='specs/example_ecommerce.json',
        api_key=api_key
    )

    try:
        await agent.initialize()
        result = await agent.run()

        # 获取统计信息
        stats = agent.get_stats()

        print("\n📊 系统统计:")
        print(f"  📝 任务ID: {stats['task_id']}")
        print(f"  📋 任务名称: {stats['task_name']}")

        if 'llm_stats' in stats:
            print(f"  🤖 LLM调用次数: {stats['llm_stats'].get('call_count', 0)}")
            print(f"  🤖 LLM使用Token: {stats['llm_stats'].get('total_tokens', 0)}")

        if 'cache_stats' in stats:
            print(f"  💾 缓存命中: {stats['cache_stats'].get('total_hits', 0)}")
            print(f"  💾 缓存大小: {stats['cache_stats'].get('cache_size', 0)}")

        if 'evidence_summary' in stats:
            summary = stats['evidence_summary']
            print(f"  📸 截图数量: {summary.get('screenshots', 0)}")
            print(f"  📄 HTML快照: {summary.get('html_snapshots', 0)}")
            print(f"  📊 数据文件: {summary.get('data_files', 0)}")

    finally:
        await agent.stop()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Full Self-Crawling Agent - 快速开始")
    print("=" * 60)

    # 运行基础示例
    print("\n[1/3] 运行基础示例...")
    asyncio.run(example_basic())

    # 运行自定义示例
    print("\n[2/3] 运行自定义示例...")
    asyncio.run(example_custom())

    # 运行调试示例
    print("\n[3/3] 运行调试示例...")
    asyncio.run(example_with_debug())

    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成!")
    print("=" * 60)

