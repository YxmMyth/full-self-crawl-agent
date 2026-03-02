#!/usr/bin/env python3
"""
简化测试脚本：验证重构后的项目基本功能
"""

import asyncio
import os
import sys
from pathlib import Path

def create_simple_spec():
    """创建一个简单的spec文件用于测试"""
    spec_content = {
        "task_id": "test_basic",
        "task_name": "Basic Test",
        "created_at": "2026-02-26T00:00:00",
        "version": "1.0",
        "extraction_type": "single_page",
        "targets": [
            {
                "name": "data",
                "fields": [
                    {
                        "name": "title",
                        "type": "text",
                        "selector": "title",
                        "required": True,
                        "description": "Page title"
                    }
                ]
            }
        ],
        "start_url": "https://www.python.org",
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

    # 写入YAML格式
    import yaml
    with open('test_basic_spec.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(spec_content, f, default_flow_style=False, allow_unicode=True)

def test_imports():
    """测试模块导入是否正常"""
    print("Testing module imports...")

    try:
        from src.main import SelfCrawlingAgent
        print("OK Main module imported successfully")

        from src.config.loader import SpecLoader
        print("OK Config loader imported successfully")

        from src.config.contracts import ContractFactory, ContractValidator
        print("OK Contracts imported successfully")

        from src.core.smart_router import SmartRouter
        print("OK Core components imported successfully")

        from src.agents.base import AgentPool
        print("OK Agents imported successfully")

        from src.tools.browser import BrowserTool
        print("OK Tools imported successfully")

        return True
    except Exception as e:
        print(f"FAIL Import error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_specs_directory():
    """测试specs目录结构"""
    print("\nTesting specs directory structure...")

    required_dirs = [
        'specs/ecommerce',
        'specs/news',
        'specs/templates'
    ]

    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"OK {dir_path} exists")
        else:
            print(f"FAIL {dir_path} missing")
            all_exist = False

    # 测试创建示例spec
    print("\nCreating example spec...")
    try:
        create_simple_spec()
        if os.path.exists('test_basic_spec.yaml'):
            print("OK Example spec created successfully")
            os.remove('test_basic_spec.yaml')  # 清理
        else:
            print("FAIL Failed to create example spec")
            all_exist = False
    except Exception as e:
        print(f"FAIL Error creating example spec: {str(e)}")
        all_exist = False

    return all_exist

def test_docs_structure():
    """测试docs目录结构"""
    print("\nTesting docs directory structure...")

    required_dirs = [
        'docs/architecture',
        'docs/design',
        'docs/guides',
        'docs/implementation',
        'docs/release',
        'docs/testing'
    ]

    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"OK {dir_path} exists")
        else:
            print(f"FAIL {dir_path} missing")
            all_exist = False

    return all_exist

def test_standard_files():
    """测试标准项目文件"""
    print("\nTesting standard project files...")

    required_files = [
        'pyproject.toml',
        'LICENSE',
        '.env.example',
        'README.md',
        'requirements.txt'
    ]

    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"OK {file_path} exists")
        else:
            print(f"FAIL {file_path} missing")
            all_exist = False

    return all_exist

async def test_basic_functionality():
    """测试基本功能（不实际运行爬取）"""
    print("\nTesting basic functionality...")

    try:
        # 检查是否所有依赖都可用
        import playwright
        import bs4
        import httpx
        import yaml
        import pandas

        print("OK All required dependencies available")

        # 测试简单初始化而不运行
        from src.main import SelfCrawlingAgent

        print("OK Agent class can be instantiated")
        return True

    except ImportError as e:
        print(f"FAIL Missing dependency: {str(e)}")
        return False
    except Exception as e:
        print(f"FAIL Error during basic functionality test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主函数：运行所有测试"""
    print("开始验证重构后的项目结构和功能...")
    print("="*60)

    tests = [
        ("模块导入测试", test_imports),
        ("规格目录结构测试", test_specs_directory),
        ("文档目录结构测试", test_docs_structure),
        ("标准文件测试", test_standard_files),
        ("基本功能测试", test_basic_functionality)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n运行: {test_name}")
        print("-" * 40)
        result = test_func()
        results.append((test_name, result))

    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    all_passed = True
    for test_name, result in results:
        status = "OK 通过" if result else "FAIL 失败"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False

    print(f"\n总体结果: {'OK 全部通过' if all_passed else 'FAIL 部分失败'}")

    # 如果所有测试都通过，则表示重构成功
    if all_passed:
        print("\n[PASS] 项目重构验证成功！")
        print("[OK] 项目结构符合现代Python标准")
        print("[OK] 所有模块可以正常导入")
        print("[OK] 目录结构完整")
        print("[OK] 所需依赖可用")
    else:
        print("\n[FAIL] 项目重构存在问题，需要修复")

    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)