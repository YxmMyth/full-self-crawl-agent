#!/usr/bin/env python3
"""
real_site_test_runner.py — 真实站点测试脚本

从 test_assets/table.csv 读取测试用例（URL + 自然语言需求），
仅向 Agent 提供 start_url 和 description，测试完全自主推断能力。

用法:
    python real_site_test_runner.py 5          # 运行第5个测试（arXiv）
    python real_site_test_runner.py 5 7 8      # 运行第5、7、8个
    python real_site_test_runner.py all         # 运行全部10个
"""

import asyncio
import csv
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 sys.path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.orchestrator import SelfCrawlingAgent
from src.core.logging import setup_logging

CSV_PATH = ROOT / 'test_assets' / 'table.csv'


def load_test_cases() -> list[dict]:
    """从 CSV 加载测试用例，返回 [{id, description, url, notes}]"""
    cases = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头
        for row in reader:
            if len(row) >= 3:
                cases.append({
                    'id': int(row[0].strip()),
                    'description': row[1].strip(),
                    'url': row[2].strip(),
                    'notes': row[3].strip() if len(row) > 3 else '',
                })
    return cases


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


async def run_one_test(case: dict, reports_dir: Path) -> dict:
    """执行单个测试：仅传入 URL + 自然语言描述"""
    task_id = f"test_{case['id']:02d}"
    start_url = case['url']
    description = case['description']

    handler = setup_file_logger(task_id, reports_dir)

    # 最小化 spec：只有 URL 和自然语言描述，其余全部由 Agent 推断
    spec = {
        'start_url': start_url,
        'description': description,
    }

    record = {
        'task_id': task_id,
        'case_id': case['id'],
        'description': description,
        'start_url': start_url,
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
    }

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"🚀 测试 #{case['id']}: {description}")
    print(f"   URL: {start_url}")
    print(f"   输入: 仅 URL + 自然语言描述（全自主推断）")
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
        record['extracted_sample'] = extracted[:3] if extracted else []

    except Exception as e:
        record['error'] = f"{type(e).__name__}: {str(e)}"
        record['warnings'].append(traceback.format_exc())
        print(f"   ❌ 异常: {record['error']}")

    finally:
        record['end_time'] = datetime.now().isoformat()
        record['duration_seconds'] = round(time.time() - t0, 2)
        logging.getLogger().removeHandler(handler)
        handler.close()

    # 打印摘要
    status = "✅ 成功" if record['success'] else "❌ 失败"
    print(f"\n   {status} | 耗时 {record['duration_seconds']}s")
    print(f"   记录数: {record['total_records']} | 质量分: {record['quality_score']:.2f}")
    print(f"   页面数: {record['pages_visited']} | 模式: {record['crawl_mode']}")
    if record['error']:
        print(f"   错误: {record['error']}")

    # 保存结果 JSON
    result_path = reports_dir / f"{task_id}_result.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)

    return record


async def main():
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    if not args:
        print("用法:")
        print("  python real_site_test_runner.py 5          # 运行第5个测试")
        print("  python real_site_test_runner.py 5 7 8      # 运行多个")
        print("  python real_site_test_runner.py all         # 运行全部")
        sys.exit(1)

    cases = load_test_cases()
    if not cases:
        print(f"❌ 无法从 {CSV_PATH} 加载测试用例")
        sys.exit(1)

    # 解析要运行的测试编号
    if 'all' in args:
        selected = cases
    else:
        ids = set()
        for a in args:
            try:
                ids.add(int(a))
            except ValueError:
                print(f"⚠️ 忽略无效参数: {a}")
        selected = [c for c in cases if c['id'] in ids]

    if not selected:
        print("❌ 没有匹配的测试用例")
        print(f"   可用编号: {[c['id'] for c in cases]}")
        sys.exit(1)

    # 配置日志
    setup_logging(level='DEBUG')

    # 准备报告目录
    reports_dir = Path('reports') / f"real_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    reports_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 报告目录: {reports_dir}")
    print(f"📋 将运行 {len(selected)} 个测试")

    all_results = []
    for case in selected:
        result = await run_one_test(case, reports_dir)
        all_results.append(result)

    # 汇总报告
    print(f"\n{'='*60}")
    print("📊 测试汇总")
    print(f"{'='*60}")
    total = len(all_results)
    passed = sum(1 for r in all_results if r['success'])
    print(f"总计: {total} | 成功: {passed} | 失败: {total - passed}")
    print(f"\n{'#':<4} {'描述':<30} {'状态':<4} {'记录':<6} {'质量':<6} {'耗时':<8}")
    print('-' * 62)
    for r in all_results:
        status = "✅" if r['success'] else "❌"
        desc = r['description'][:28]
        print(f"{r['case_id']:<4} {desc:<30} {status:<4} {r['total_records']:<6} {r['quality_score']:<6.2f} {r['duration_seconds']:<8.1f}s")

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
