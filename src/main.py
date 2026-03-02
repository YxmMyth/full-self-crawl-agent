"""
主入口 - Full Self-Crawling Agent
CLI 入口点，委托给 Orchestrator 处理核心逻辑
"""

import asyncio
import sys
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件

# 导入日志配置
from src.core.logging import setup_logging, get_logger
from src.orchestrator import SelfCrawlingAgent  # 向后兼容：测试/外部代码从 src.main 导入

# 配置日志
setup_logging(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = get_logger('main')


def main():
    """CLI 入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Full Self-Crawling Agent - 智能网页数据爬取'
    )
    parser.add_argument('spec_file', nargs='?', default=None,
                        help='Spec 文件路径')
    parser.add_argument('--container', action='store_true',
                        help='容器模式（从 TASK_SPEC 环境变量读取任务）')
    parser.add_argument('--api-key', help='智谱 API Key')
    parser.add_argument('--model', help='模型名称（默认: glm-4）')
    parser.add_argument('--api-base', help='自定义 API 端点')
    parser.add_argument('--headless', action='store_true', default=True,
                        help='无头模式（默认）')
    parser.add_argument('--debug', action='store_true', help='调试模式')

    args = parser.parse_args()

    # 设置调试日志
    if args.debug:
        setup_logging(level='DEBUG')

    if args.container:
        from src.run_mode import build_run_context
        ctx = build_run_context()
        agent = SelfCrawlingAgent()
        # Apply context-specific configurations if needed
    else:
        if not args.spec_file:
            parser.error("本地模式需要指定 spec 文件路径")

        # Load spec from file
        import json
        with open(args.spec_file, 'r', encoding='utf-8') as f:
            spec = json.load(f)

        # Extract URL and continue with agent
        start_url = spec.get('start_url') or spec.get('url')
        if not start_url:
            parser.error("Spec 文件必须包含 start_url 或 url 字段")

        agent = SelfCrawlingAgent()

    try:
        if args.container:
            from src.run_mode import build_run_context
            ctx = build_run_context()
            # Process using context
            result = asyncio.run(agent.run(ctx.get('start_url'), ctx.get('spec', {})))
        else:
            # Load spec from file
            import json
            with open(args.spec_file, 'r', encoding='utf-8') as f:
                spec = json.load(f)

            start_url = spec.get('start_url') or spec.get('url')
            result = asyncio.run(agent.run(start_url, spec))

        if result['success']:
            logger.info("任务成功完成")
            logger.info(f"已提取 {len(result.get('extracted_data', []))} 条数据")
            logger.info(f"迭代次数: {result.get('iterations', 'N/A')}")
            logger.info(f"质量分数: {result.get('quality_score', 0):.2f}")

            if args.debug:
                logger.debug(f"详细信息: {result.get('verification', {})}")
        else:
            logger.error(f"任务失败: {result.get('error', '未知错误')}")
            if result.get('extracted_data'):
                logger.info(f"部分数据: {len(result['extracted_data'])} 条")

    except KeyboardInterrupt:
        logger.warning("任务被用户中断")

    except Exception as e:
        logger.error(f"任务失败: {str(e)}")
        logger.exception("详细错误信息:")


if __name__ == '__main__':
    main()
