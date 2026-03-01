#!/bin/bash
# 端到端测试脚本 - 10个网站测试

echo "🚀 启动端到端测试 - 10个真实网站"
echo "Testing PlanAgent retry architecture..."
echo ""

cd "$(dirname "$0")/.."

# 测试配置
SITES=(
    "techcrunch:specs/test_sites/site_01_techcrunch.yaml:https://techcrunch.com/"
    "arxiv:specs/test_sites/site_05_arxiv.yaml:https://arxiv.org/"
    "github_trending:specs/test_sites/site_02_github_trending.yaml:https://github.com/trending"
    "hackernews:specs/test_sites/site_03_hackernews.yaml:https://news.ycombinator.com/"
    "reddit_prog:specs/test_sites/site_04_reddit_programming.yaml:https://www.reddit.com/r/programming/"
    "product_hunt:specs/test_sites/site_06_product_hunt.yaml:https://www.producthunt.com/"
    "medium:specs/test_sites/site_07_medium.yaml:https://medium.com/"
    "stackoverflow:specs/test_sites/site_08_stackoverflow.yaml:https://stackoverflow.com/"
    "lobsters:specs/test_sites/site_09_lobsters.yaml:https://lobste.rs/"
    "infoq:specs/test_sites/site_10_infoq.yaml:https://www.infoq.com/"
)

LOG_DIR="reports"
mkdir -p "$LOG_DIR"

echo "📋 测试清单:"
for site_info in "${SITES[@]}"; do
    IFS=':' read -r name spec url <<< "$site_info"
    echo "  - $name ($url)"
done
echo ""

TOTAL=${#SITES[@]}
PASSED=0
FAILED=0

for i in "${!SITES[@]}"; do
    site_info="${SITES[$i]}"
    IFS=':' read -r name spec url <<< "$site_info"

    index=$((i + 1))
    echo "[$index/$TOTAL] 🧪 测试: $name"
    echo "    URL: $url"

    # 运行单个测试
    python -m src.main --spec "$spec" --url "$url" > "$LOG_DIR/e2e_${name}.log" 2>&1

    if [ $? -eq 0 ]; then
        echo "    ✅ 成功"
        PASSED=$((PASSED + 1))
    else
        echo "    ❌ 失败"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "{'='*80}"
echo "📊 测试总结报告"
echo "{'='*80}"
echo ""
echo "✅ 通过: $PASSED/$TOTAL"
echo "❌ 失败: $FAILED/$TOTAL"
echo "🎯 成功率: $(awk "BEGIN {printf \"%.1f\", ($PASSED/$TOTAL)*100}")%"

echo ""
echo "📝 详细日志保存在: $LOG_DIR/e2e_*.log"
