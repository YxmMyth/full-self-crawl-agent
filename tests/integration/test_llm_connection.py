#!/usr/bin/env python3
"""
测试两个LLM（DeepSeek和GLM）连接的脚本
"""

import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.tools.multi_llm_client import MultiLLMClient
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_llm_connections():
    """测试两个LLM的连接"""
    print("正在初始化多LLM客户端...")

    try:
        # 从环境变量加载配置
        client = MultiLLMClient.from_env()

        print("✅ 多LLM客户端初始化成功")
        print(f"推理客户端状态: {'已配置' if client.reasoning_client else '未配置'}")
        print(f"编码客户端状态: {'已配置' if client.coding_client else '未配置'}")
        print(f"默认客户端状态: {'已配置' if client.default_client else '未配置'}")

        # 测试推理任务 (DeepSeek)
        if client.reasoning_client:
            print("\n🧪 正在测试推理任务 (DeepSeek)...")
            try:
                reasoning_result = await client.reason(
                    "请简要分析人工智能对现代社会的影响，控制在50字以内。",
                    system_prompt="你是一个专业的AI分析专家。"
                )
                print("✅ 推理任务成功:")
                print(f"   {reasoning_result[:100]}...")
            except Exception as e:
                print(f"❌ 推理任务失败: {e}")
        else:
            print("\n⚠️ 推理客户端未配置")

        # 测试编码任务 (GLM)
        if client.coding_client:
            print("\n🧪 正在测试编码任务 (GLM)...")
            try:
                coding_result = await client.code(
                    "写一个Python函数计算斐波那契数列的前10项",
                    system_prompt="你是一个资深的Python开发工程师。"
                )
                print("✅ 编码任务成功:")
                print(f"   {coding_result[:100]}...")
            except Exception as e:
                print(f"❌ 编码任务失败: {e}")
        else:
            print("\n⚠️ 编码客户端未配置")

        # 测试通用任务
        if client.default_client:
            print("\n🧪 正在测试通用任务...")
            try:
                general_result = await client.generate(
                    "介绍一下中国的四大发明",
                    system_prompt="你是一个知识渊博的历史学家。"
                )
                print("✅ 通用任务成功:")
                print(f"   {general_result[:100]}...")
            except Exception as e:
                print(f"❌ 通用任务失败: {e}")

        # 显示统计信息
        print("\n📊 客户端统计信息:")
        stats = client.get_stats()
        print(f"   推理调用次数: {stats['reasoning_calls']}")
        print(f"   编码调用次数: {stats['coding_calls']}")
        print(f"   回退调用次数: {stats['fallback_calls']}")
        print(f"   错误次数: {stats['error_count']}")

        if 'circuit_breakers' in stats:
            print(f"   断路器状态:")
            for cb_name, cb_stats in stats['circuit_breakers'].items():
                print(f"     {cb_name}: {cb_stats['state']} (失败次数: {cb_stats['failure_count']})")

        # 关闭客户端
        await client.close()

        print("\n✅ 所有测试完成！")

    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("开始测试LLM连接...")
    asyncio.run(test_llm_connections())