# FeatureDetector v2 — 回滚指南与重启说明

## 问题背景

PR #2（"[WIP] Fix core issues in FeatureDetector functionality"）在仅提交了一个 **"Initial plan"** 占位 commit（无任何代码改动）的情况下被提前合并进了 `main` 分支。由于该 PR 处于 `[WIP]` 状态，所有计划中的实现项均未完成，合并属于误操作。

## 影响评估

由于 PR #2 分支上唯一的提交（`10633aad`）**不包含任何文件改动**（`git diff stats: {}`），`main` 分支的代码树在该合并前后**完全一致**。因此：

- **无需还原任何源码文件** — 代码状态已等同于合并前。
- 本次回滚仅需在 `main` 提交历史中留下一条正式的回滚记录，以防止后续基于该 WIP 合并进行错误开发。

## 回滚操作说明

### 已在本 PR 中完成的操作

本 PR（`copilot/reset-main-to-featuredetector-v2`）合并后将在 `main` 上留下正式的回滚 commit，标记 `main` 分支已恢复至 PR #2 合并前的正确状态：

- 目标基准 commit（PR #2 合并前 `main` 的最新状态）：`48af76950344c1932455bb5e466bd530b712ef6d`
  — *"Merge pull request #1 from YxmMyth/copilot/evaluate-agent-architecture"*
- 被回滚的误合并 commit：`4b74ae06e423ac549f74acc5b39cf6139ac3c064`
  — *"Merge pull request #2 from YxmMyth/copilot/fix-feature-detector-issues"*

### 如需在本地验证回滚效果

```bash
# 确认当前 main 的代码树与 48af769 一致
git diff 48af76950344c1932455bb5e466bd530b712ef6d HEAD -- src/
# 期望输出：无差异（空输出）
```

---

## 如何重新启动 FeatureDetector v2 的 PR 流程

### 步骤 1：确认 `main` 已合并本回滚 PR

确保本 PR 已被合并，`main` 已回到正确基准状态。

### 步骤 2：基于最新 `main` 创建新的功能分支

```bash
git checkout main
git pull origin main
git checkout -b feature/feature-detector-v2
```

### 步骤 3：按照 PR #2 原始需求完整实现 FeatureDetector v2

PR #2 的原始实现计划（位于 `src/core/smart_router.py`，`FeatureDetector` 类）：

- [ ] **重写 `_detect_spa(self, soup, html)`**：改为三信号判断（body 内容量、框架标记、JS bundle 数量），删除旧的硬编码字符串检测
- [ ] **重写 `_detect_pagination(self, soup, html)`**：改为优先级检测，消除 `'next' in html.lower()` 的假阳性
- [ ] **新增 `_classify_page_type_unified(self, soup, html, features)`**：合并 `FeatureDetector._classify_page_type()` 与 `SenseAgent._analyze_structure()` 两套分类体系
- [ ] **新增 `_detect_repeating_containers(self, soup)`**：基于结构相似度的列表容器检测，替代硬编码 class 名
- [ ] **新增 `_clean_class_name(self, class_name)`**：处理 CSS Modules / Styled Components 生成的 class 名
- [ ] **新增 `_compute_structure_similarity(self, elements)`**：计算元素结构相似度评分
- [ ] **新增 `_is_detail_page(self, soup)`**：基于内容特征判断详情页
- [ ] **更新 `analyze()`**：使用上述新方法，并在返回值中增加 `container_info` 字段
- [ ] **创建 `tests/unit/test_feature_detector_v2.py`**：包含 8 个测试用例，覆盖所有新方法
- [ ] **确认所有现有测试仍然通过**：`pytest tests/ -v`

### 步骤 4：提交并推送，发起新 PR

```bash
git add -A
git commit -m "feat: implement FeatureDetector v2 with improved SPA/pagination/page-type detection"
git push origin feature/feature-detector-v2
# 然后在 GitHub 上创建 PR，目标分支为 main
```

### 步骤 5：PR 合并前检查清单

- [ ] 所有 10 项实现任务均已完成（见步骤 3）
- [ ] `pytest tests/ -v` 全部通过
- [ ] PR 不再标记 `[WIP]`
- [ ] Code review 已通过

---

## 参考资料

- PR #1（已合并）：[Fix 4 silent/crash-level architectural bugs](https://github.com/YxmMyth/full-self-crawl-agent/pull/1)
- PR #2（已回滚的 WIP 合并）：[\[WIP\] Fix core issues in FeatureDetector functionality](https://github.com/YxmMyth/full-self-crawl-agent/pull/2)
- FeatureDetector 源码：`src/core/smart_router.py`
- 架构设计文档：`docs/implementation/IMPLEMENTATION.md` § 3.1
