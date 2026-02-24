# 开发脚本

set -e

echo "=================="
echo "初始化项目"
echo "=================="

# 安装 Python 依赖
echo "→ 安装依赖..."
pip install -r requirements.txt

# 安装 Playwright
echo "→ 安装 Playwright 浏览器..."
python -m playwright install chromium

# 运行初始化脚本
echo "→ 运行初始化脚本..."
python scripts/initialize.py

echo ""
echo "✓ 初始化完成！"
echo ""
echo "下一步:"
echo "  1. 复制配置: cp .env.example .env"
echo "  2. 编辑 .env 文件，填入 API Key"
echo "  3. 运行示例: python src/main.py examples/example_ecommerce.yaml"
