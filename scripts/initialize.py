"""
项目初始化脚本
设置必要的目录和依赖
"""

import os
import sys
from pathlib import Path


def create_directories():
    """创建必要目录"""
    dirs = [
        'specs',
        'config',
        'evidence',
        'states',
        'logs',
        'tests',
        'examples'
    ]

    print("→ 创建目录...")
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        print(f"  ✓ {d}")


def check_playwright():
    """检查 Playwright 并安装浏览器"""
    print("\n→ 检查 Playwright...")

    try:
        from playwright.__main__ import main as playwright_main

        print("  安装浏览器...")
        sys.argv = ['playwright', 'install', 'chromium']
        playwright_main()

        print("✓ Playwright 浏览器安装完成")

    except ImportError:
        print("✗ Playwright 未安装")
        print("  请先运行: pip install -r requirements.txt")
        return False

    except Exception as e:
        print(f"✗ Playwright 安装失败: {e}")
        return False

    return True


def check_dependencies():
    """检查 Python 依赖"""
    print("\n→ 检查依赖...")

    required_packages = [
        'playwright',
        'beautifulsoup4',
        'httpx',
        'pyyaml',
        'aiohttp'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ✗ {package} (未安装)")

    if missing:
        print(f"\n⚠ 缺少以下包: {', '.join(missing)}")
        print("  请运行: pip install -r requirements.txt")
        return False

    return True


def check_env_file():
    """检查环境配置文件"""
    print("\n→ 检查环境配置...")

    if not Path('.env').exists():
        if Path('.env.example').exists():
            print("  检测到 .env.example，建议复制为 .env 并配置 API Key")
            print("  cp .env.example .env")
        else:
            print("  警告: 没有 .env 配置文件")
    else:
        print("  ✓ .env 文件存在")


def check_config_files():
    """检查配置文件"""
    print("\n→ 检查配置文件...")

    config_files = [
        'config/settings.json',
        'config/policies.json'
    ]

    for file in config_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (缺失)")


def check_example_specs():
    """检查示例 Spec"""
    print("\n→ 检查示例契约...")

    if Path('examples').exists():
        specs = list(Path('examples').glob('*.yaml')) + list(Path('examples').glob('*.json'))
        if specs:
            print(f"  ✓ 找到 {len(specs)} 个示例契约")
            for spec in specs:
                print(f"    - {spec.name}")
        else:
            print("  ✗ examples 目录下没有示例契约")
    else:
        print("  ✗ examples 目录不存在")


def show_summary():
    """显示总结"""
    print("\n" + "="*50)
    print("✓ 项目初始化完成")
    print("="*50)

    print("\n下一步:")
    print("  1. 配置 API Key:")
    print("     cp .env.example .env")
    print("     编辑 .env 文件，填入 ZHIPU_API_KEY")

    print("\n  2. 创建或修改 Spec 契约:")
    print("     编辑 examples/example_ecommerce.yaml")

    print("\n  3. 运行任务:")
    print("     python src/main.py examples/example_ecommerce.yaml")

    print("\n  4. 查看证据:")
    print("     任务完成后查看 evidence/{task_id}/ 目录")

    print("\n详细文档请查看 README.md")
    print("="*50)


def main():
    """主函数"""
    print("="*50)
    print("Full Self-Crawling Agent - 项目初始化")
    print("="*50)

    # 创建目录
    create_directories()

    # 检查依赖
    deps_ok = check_dependencies()

    # 安装 Playwright 浏览器
    if deps_ok:
        playwright_ok = check_playwright()

    # 检查配置
    check_env_file()
    check_config_files()
    check_example_specs()

    # 显示总结
    show_summary()


if __name__ == '__main__':
    main()
