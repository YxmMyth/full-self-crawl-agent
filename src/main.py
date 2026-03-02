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


def _save_results(spec: dict, result: dict):
    """持久化爬取结果到 evidence 目录"""
    try:
        import json as _json
        from src.tools.storage import EvidenceStorage

        task_id = spec.get('task_id', 'unnamed_task')
        storage = EvidenceStorage('./evidence')
        storage.create_task_dir(task_id)

        # 保存提取数据
        extracted = result.get('extracted_data', [])
        if extracted:
            storage.save_data(extracted, 'extracted_data.json')
            logger.info(f"数据已保存到 evidence/{task_id}/data/extracted_data.json")

        # 保存完整结果摘要（过滤不可序列化内容）
        def _safe(obj):
            if isinstance(obj, bytes):
                return f'<bytes:{len(obj)}>'
            if isinstance(obj, set):
                return list(obj)
            return str(obj)

        summary = {
            'success': result.get('success'),
            'quality_score': result.get('quality_score', 0),
            'pages_visited': result.get('pages_visited', 0),
            'urls_visited': result.get('urls_visited', []),
            'total_records': len(extracted),
            'crawl_mode': result.get('crawl_mode', 'unknown'),
            'summary': result.get('summary', {}),
        }
        summary_path = storage.current_task_dir / 'data' / 'run_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            _json.dump(summary, f, ensure_ascii=False, indent=2, default=_safe)

        # 保存每页详细结果
        for i, page_r in enumerate(result.get('results', [])):
            page_data = {
                'url': page_r.get('url', ''),
                'depth': page_r.get('depth', 0),
                'extracted_data': page_r.get('result', {}).get('extracted_data', []),
                'quality_score': page_r.get('result', {}).get('verification_result', {}).get('quality_score'),
                'reflection': page_r.get('result', {}).get('reflection_notes', {}).get('improvements', {}),
            }
            page_path = storage.current_task_dir / 'data' / f'page_{i}.json'
            with open(page_path, 'w', encoding='utf-8') as f:
                _json.dump(page_data, f, ensure_ascii=False, indent=2, default=_safe)

    except Exception as e:
        logger.warning(f"保存结果失败: {e}")


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

            # 持久化结果到 evidence 目录
            _save_results(spec, result)

            if args.debug:
                logger.debug(f"详细信息: {result.get('verification', {})}")
        else:
            logger.error(f"任务失败: {result.get('error', '未知错误')}")
            if result.get('extracted_data'):
                logger.info(f"部分数据: {len(result['extracted_data'])} 条")
                _save_results(spec, result)

    except KeyboardInterrupt:
        logger.warning("任务被用户中断")

    except Exception as e:
        logger.error(f"任务失败: {str(e)}")
        logger.exception("详细错误信息:")


if __name__ == '__main__':
    main()
