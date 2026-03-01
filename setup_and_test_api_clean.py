#!/usr/bin/env python3
"""
API Gateway 配置脚本

该脚本用于配置和验证 API Gateway 设置
"""

import asyncio
import os
import sys

from src.tools.multi_llm_client import MultiLLMClient
from src.tools.api_gateway_client import APIGatewayConfig


def setup_api_gateway():
    """设置 API Gateway 配置"""
    print("Configuring API Gateway...")

    # 设置环境变量
    os.environ['API_GATEWAY_KEY'] = 'sk-j8qM9nlrE3cpu3mw1wjoGpVBxjDCzmuUGTgEMNhPlpo5Zesb'
    os.environ['API_GATEWAY_MODEL'] = 'claude-opus-4-5-20251101'
    os.environ['API_GATEWAY_API_BASE'] = 'http://45.78.224.156:3000/v1'

    print("API Gateway configuration has been set:")
    print(f"  Model: {os.environ['API_GATEWAY_MODEL']}")
    print(f"  API Base: {os.environ['API_GATEWAY_API_BASE']}")

    # 验证配置
    config = APIGatewayConfig.from_env()
    if config:
        print(f"Configuration verification successful, Provider: {config.provider}")
    else:
        print("Configuration verification failed")

    return config is not None


async def test_multi_llm_client():
    """测试 MultiLLMClient 与 API Gateway 的集成"""
    print("\\nTesting MultiLLMClient integration with API Gateway...")

    try:
        # 创建客户端
        client = MultiLLMClient.from_env()
        print("MultiLLMClient initialization successful")

        # 测试推理任务
        print("\\nTesting reasoning task...")
        reasoning_result = await client.reason(
            prompt="请解释一下什么是人工智能？",
            max_tokens=150
        )
        print(f"Reasoning task completed: {len(reasoning_result)} characters response")

        # 测试编码任务
        print("\\nTesting coding task...")
        coding_result = await client.code(
            prompt="用Python写一个简单的快速排序算法",
            max_tokens=300
        )
        print(f"Coding task completed: {len(coding_result)} characters response")

        # 测试通用任务
        print("\\nTesting general task...")
        general_result = await client.generate(
            prompt="请用一句话总结人工智能的现状",
            max_tokens=100
        )
        print(f"General task completed: {len(general_result)} characters response")

        # 显示统计信息
        stats = client.get_stats()
        print(f"\\nClient statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")

        # 关闭客户端
        await client.close()
        print("\\nAll tests completed!")
        return True

    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主函数"""
    print("Starting API Gateway configuration and testing...")

    # 设置 API Gateway
    if not setup_api_gateway():
        print("Configuration failed, exiting...")
        return

    # 测试 MultiLLMClient
    success = await test_multi_llm_client()

    if success:
        print("\\nAll configurations and tests completed successfully!")
        print("\\nUsage instructions:")
        print("  1. API Gateway has been configured as the primary LLM provider")
        print("  2. Model: claude-opus-4-5-20251101")
        print("  3. All LLM calls will prioritize this API gateway")
        print("  4. If the API gateway is unavailable, it will fall back to other configured providers")
    else:
        print("\\nTests failed, please check configuration")


if __name__ == "__main__":
    asyncio.run(main())