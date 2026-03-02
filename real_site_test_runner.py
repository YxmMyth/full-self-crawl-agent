#!/usr/bin/env python3
"""
real_site_test_runner.py — 真实站点测试脚本
从 specs/test_sites/*.yaml 加载规格，执行爬取，记录详细日志和结果
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import yaml

# 确保项目根目录在 sys.path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.orchestrator import SelfCrawlingAgent
from src.core.logging import setup_logging


def load_yaml_spec(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_file_logger(task_id: str, reports_dir: Path) -> logging.FileHandler:
    """为单个测试任务配置文件级日志"""
    log_path = reports_dir / f"{task_id}.log"
    handler = logging.FileHandler(log_path, encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    logging.getLogger().addHandler(handler)
    return handler


async def run_one_site(spec_path: str, reports_dir: Path) -> dict:
    """执行单个站点测试，返回详细结果"""
    spec = load_yaml_spec(spec_path)
    task_id = spec.get('task_id', Path(spec_path).stem)
    start_url = spec.get('start_url', '')

    # 设置日志
    handler = setup_file_logger(task_id, reports_dir)

    record = {
        'task_id': task_id,
        'spec_file': str(spec_path),
        'start_url': start_url,
        'extraction_type': spec.get('extraction_type', 'unknown'),
        'start_time': datetime.now().isoformat(),
        'end_time': None,
        'duration_seconds': 0,
        'success': False,
        'error': None,
        'total_records': 0,
        'quality_score': 0,
        'pages_visited': 0,
        'crawl_mode': None,
        'extracted_sample': [],
        'warnings': [],
        'completion_criteria': spec.get('completion_criteria', {}),
        'criteria_met': {},
    }

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"🚀 开始测试: {task_id}")
    print(f"   URL: {start_url}")
    print(f"   类型: {spec.get('extraction_type')}")
    print(f"{'='*60}")

    try:
        agent = SelfCrawlingAgent()
        result = await agent.run(start_url, spec)

        record['success'] = result.get('success', False)
        record['crawl_mode'] = result.get('crawl_mode', 'unknown')
        record['pages_visited'] = result.get('pages_visited', 0)
        record['quality_score'] = result.get('quality_score', 0)

        extracted = result.get('extracted_data', [])
        record['total_records'] = len(extracted)
        # 保存前 3 条记录作为样本
        record['extracted_sample'] = extracted[:3] if extracted else []

        # 评估是否满足完成标准
        criteria = spec.get('completion_criteria', {})
        min_items = criteria.get('min_items', 0)
        quality_thresh = criteria.get('quality_threshold', 0)
        max_err = criteria.get('max_error_rate', 1.0)

        record['criteria_met'] = {
            'min_items': len(extracted) >= min_items if min_items else True,
            'quality_threshold': record['quality_score'] >= quality_thresh if quality_thresh else True,
        }

    except Exception as e:
        record['error'] = f"{type(e).__name__}: {str(e)}"
        record['warnings'].append(traceback.format_exc())
        print(f"   ❌ 异常: {record['error']}")

    finally:
        record['end_time'] = datetime.now().isoformat()
        record['duration_seconds'] = round(time.time() - t0, 2)

        # 移除文件 handler
        logging.getLogger().removeHandler(handler)
        handler.close()

    # 打印摘要
    status = "✅ 成功" if record['success'] else "❌ 失败"
    print(f"\n   {status} | 耗时 {record['duration_seconds']}s")
    print(f"   记录数: {record['total_records']} | 质量分: {record['quality_score']:.2f}")
    print(f"   页面数: {record['pages_visited']} | 模式: {record['crawl_mode']}")
    if record['error']:
        print(f"   错误: {record['error']}")
    criteria_ok = all(record['criteria_met'].values()) if record['criteria_met'] else False
    print(f"   完成标准: {'✅ 全部满足' if criteria_ok else '⚠️ 未满足'} {record['criteria_met']}")

    # 保存结果 JSON
    result_path = reports_dir / f"{task_id}_result.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)

    return record


async def main():
    # 解析参数
    sites = sys.argv[1:] if len(sys.argv) > 1 else []
    if not sites:
        print("用法: python real_site_test_runner.py site_05_arxiv site_07_allrecipes ...")
        sys.exit(1)

    # 配置日志
    setup_logging(level='DEBUG')

    # 准备报告目录
    reports_dir = Path('reports') / f"real_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    reports_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 报告目录: {reports_dir}")

    specs_dir = Path('specs/test_sites')
    all_results = []

    for site_name in sites:
        # 支持 site_05_arxiv 或完整路径
        if not site_name.endswith('.yaml'):
            spec_path = specs_dir / f"{site_name}.yaml"
        else:
            spec_path = Path(site_name)

        if not spec_path.exists():
            print(f"⚠️ 跳过不存在的 spec: {spec_path}")
            continue

        result = await run_one_site(str(spec_path), reports_dir)
        all_results.append(result)

    # 汇总报告
    print(f"\n{'='*60}")
    print("📊 测试汇总")
    print(f"{'='*60}")
    total = len(all_results)
    passed = sum(1 for r in all_results if r['success'])
    print(f"总计: {total} | 成功: {passed} | 失败: {total - passed}")
    print(f"\n{'任务ID':<25} {'状态':<8} {'记录数':<8} {'质量分':<8} {'耗时':<8}")
    print('-' * 60)
    for r in all_results:
        status = "✅" if r['success'] else "❌"
        print(f"{r['task_id']:<25} {status:<8} {r['total_records']:<8} {r['quality_score']:<8.2f} {r['duration_seconds']:<8.1f}s")

    # 保存汇总
    summary_path = reports_dir / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'test_time': datetime.now().isoformat(),
            'total': total,
            'passed': passed,
            'failed': total - passed,
            'results': all_results,
        }, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n📄 详细报告: {reports_dir}")
    return all_results


if __name__ == '__main__':
    asyncio.run(main())
