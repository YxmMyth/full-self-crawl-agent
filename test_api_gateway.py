#!/usr/bin/env python3
"""
API Gateway 配置测试脚本

该脚本用于验证 API Gateway 配置是否正确
"""

import asyncio
import os
from src.tools.api_gateway_client import APIGatewayClient, APIGatewayConfig


async def test_api_gateway():
    """测试 API Gateway 连接"""
    print("开始测试 API Gateway 配置...")

    # 设置环境变量（在实际部署时，这些应作为环境变量设置）
    os.environ['API_GATEWAY_KEY'] = 'sk-j8qM9nlrE3cpu3mw1wjoGpVBxjDCzmuUGTgEMNhPlpo5Zesb'
    os.environ['API_GATEWAY_MODEL'] = 'claude-opus-4-5-20251101'
    os.environ['API_GATEWAY_API_BASE'] = 'http://45.78.224.156:3000/v1'

    # 从环境变量加载配置
    config = APIGatewayConfig.from_env()
    if not config:
        print("❌ 无法加载 API Gateway 配置，请检查环境变量设置")
        return False

    print(f"✅ 加载配置成功:")
    print(f"   模型: {config.model}")
    print(f"   提供商: {config.provider}")
    print(f"   API 地址: {config.api_base}")

    try:
        # 创建客户端并测试连接
        client = APIGatewayClient(config)

        print("\n🔄 正在测试 API 连接...")
        response = await client.generate(
            prompt="你好，请简单介绍一下自己。",
            system_prompt="你是一个有用的AI助手。",
            max_tokens=100
        )

        print(f"✅ 测试成功! 响应: {response[:100]}...")

        # 显示统计信息
        stats = client.get_stats()
        print(f"\n📊 统计信息:")
        print(f"   调用次数: {stats['call_count']}")
        print(f"   总tokens: {stats['total_tokens']}")
        print(f"   错误次数: {stats['error_count']}")

        await client.close()
        return True

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_api_gateway())
    if success:
        print("\n🎉 API Gateway 配置测试通过!")
    else:
        print("\n💥 API Gateway 配置测试失败!")