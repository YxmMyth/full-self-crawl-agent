"""
监控系统测试脚本
用于验证监控和进度追踪系统是否正常工作
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.metrics_collector import MetricsCollector
from src.monitoring.progress_tracker import ProgressTracker, TaskStage, ProgressStatus
from src.core.state_manager import StateManager


async def test_monitoring_system():
    """测试监控系统功能"""
    print("开始测试监控系统...")

    # 创建监控组件
    metrics_collector = MetricsCollector()
    progress_tracker = ProgressTracker()
    state_manager = StateManager()

    print("1. 测试指标收集器...")
    # 测试LLM指标记录
    await metrics_collector.record_llm_call_async('zhipu', 1200.5, True, 150)
    await metrics_collector.record_llm_call_async('alibaba', 950.2, False)

    # 测试页面加载指标记录
    await metrics_collector.record_page_load_async('https://example.com', 800.3, True)

    # 测试代码执行指标记录
    await metrics_collector.record_code_execution_async(45.6, True)
    await metrics_collector.record_code_execution_async(120.1, False, 'Timeout error')

    # 获取指标摘要
    metrics_summary = metrics_collector.get_metrics_summary()
    print(f"   LLM调用总数: {metrics_summary['llm_calls']['total']}")
    print(f"   页面加载成功数: {metrics_summary['page_loads']['successes']}")
    print(f"   代码执行失败数: {metrics_summary['code_executions']['failures']}")

    print("\n2. 测试进度追踪器...")
    # 测试进度更新
    task_id = "test_task_001"
    await progress_tracker.update_progress(
        task_id,
        TaskStage.SENSE,
        0.3,
        {'message': '正在感知页面'},
        ProgressStatus.RUNNING
    )

    await progress_tracker.update_progress(
        task_id,
        TaskStage.PLAN,
        0.6,
        {'message': '正在规划策略'},
        ProgressStatus.RUNNING
    )

    await progress_tracker.update_progress(
        task_id,
        TaskStage.ACT,
        0.9,
        {'message': '正在执行提取'},
        ProgressStatus.RUNNING
    )

    # 获取任务进度
    task_progress = progress_tracker.get_task_progress(task_id)
    print(f"   任务 {task_id} 总体进度: {task_progress['overall_progress']:.2f}")
    print(f"   当前阶段: {task_progress['current_stage']}")

    print("\n3. 测试状态管理器（集成监控）...")
    # 更新状态管理器中的进度
    await state_manager.update_progress(
        task_id,
        TaskStage.VERIFY,
        0.75,
        {'message': '正在验证数据'},
        ProgressStatus.RUNNING
    )

    # 获取任务进度
    state_task_progress = state_manager.get_task_progress(task_id)
    print(f"   通过状态管理器获取进度: {state_task_progress['overall_progress']:.2f}")

    print("\n4. 测试指标收集器...")
    # 通过状态管理器记录指标
    await state_manager.record_llm_call('deepseek', 1500.8, True, 200)
    await state_manager.record_page_load('https://another-example.com', 650.4, True)
    await state_manager.record_code_execution(89.7, True)

    # 获取指标摘要
    state_metrics = state_manager.get_metrics_summary()
    print(f"   LLM调用总数(通过状态管理器): {state_metrics['llm_calls']['total']}")

    print("\n监控系统测试完成！所有组件正常工作。")


if __name__ == "__main__":
    asyncio.run(test_monitoring_system())