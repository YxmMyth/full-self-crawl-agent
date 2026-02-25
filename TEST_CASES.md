# 测试用例文档

## 10个真实场景测试用例

从 `need+site/table.csv` 提取的完整测试场景，涵盖各种网页数据爬取需求。

### 1. 电商产品页面
- **类型**: 电商手机产品页
- **数据字段**: 名称/价格/规格/主图URL/PDF
- **测试URL**: https://www.amazon.com/s?k=smartphone
- **匹配理由**: 海量手机列表页 + 完整产品页（价格、规格表JSON、清晰主图、规格PDF下载）

### 2. 新闻文章
- **类型**: 新闻文章富文本
- **数据字段**: 标题/HTML片段/图片/视频
- **测试URL**: https://techcrunch.com/
- **匹配理由**: 最新科技文章页（富HTML、正文含嵌入图片+视频、结构完整）

### 3. 数据可视化图表
- **类型**: 数据可视化图表站
- **数据字段**: SVG代码
- **测试URL**: https://www.datawrapper.de/
- **匹配理由**: 公开图表库，每张图都能导出/查看完整SVG代码 + 渲染预览

### 4. 招聘信息
- **类型**: 招聘网站
- **数据字段**: 职位/薪资/公司Logo图片/ JD HTML
- **测试URL**: https://www.indeed.com/hire/job-description/software-engineer
- **匹配理由**: 软件工程师职位列表 + 每条带公司Logo图片 + 完整JD HTML片段

### 5. 学术论文
- **类型**: 学术论文站点
- **数据字段**: PDF下载 + 摘要
- **测试URL**: https://arxiv.org/list/cs/recent
- **匹配理由**: AI论文列表 + 每篇都有PDF直链 + 摘要文本（arXiv经典）

### 6. 房地产信息
- **类型**: 房地产楼盘站
- **数据字段**: 户型图SVG/PDF/价格
- **测试URL**: https://www.zillow.com/
- **匹配理由**: 楼盘详情页（户型图PDF/SVG、价格、朝向，动态加载）

### 7. 菜谱网站
- **类型**: 菜谱网站
- **数据字段**: 配料JSON/步骤HTML/成品图片
- **测试URL**: https://www.allrecipes.com/
- **匹配理由**: 任意菜谱页（如Chef John's Jollof Rice）——配料表、步骤HTML、清晰成品图

### 8. 股票图表
- **类型**: 股票/基金图表站
- **数据字段**: K线SVG/价格
- **测试URL**: https://finance.yahoo.com/quote/AAPL/chart
- **匹配理由**: Apple股票K线图（动态SVG可提取、实时价格、成交量）

### 9. 博客文章
- **类型**: 博客/CMS站点
- **数据字段**: 嵌套HTML+图片
- **测试URL**: https://medium.com/
- **匹配理由**: 任意Medium文章页（富HTML嵌套、大量内嵌图片URL、标签）

### 10. 政府公告
- **类型**: 政府/企业公告站
- **数据字段**: PDF+HTML表格
- **测试URL**: https://www.find-tender.service.gov.uk/
- **匹配理由**: 英国政府招标公告（每条有PDF链接 + HTML表格中标金额等）

## 使用说明

这些测试用例可用于：
1. 验证爬虫系统能否处理各种网页类型
2. 测试不同数据格式（HTML/SVG/PDF/JSON）的提取
3. 检查动态内容加载的处理能力
4. 验证反爬机制的应对策略

运行测试：
```bash
# 创建对应的 Spec 契约文件
python scripts/create_test_specs.py

# 运行特定测试用例
python src/main.py specs/test_amazon.yaml
```
