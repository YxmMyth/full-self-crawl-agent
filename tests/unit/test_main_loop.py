"""
主循环 (SelfCrawlingAgent) 单元测试

覆盖：
- single_page 模式调用 _run_single_or_multi 而非 _run_full_site
- full_site 模式调用 _run_full_site 而非 _run_single_or_multi
- max_pages 上限：_run_full_site 不超过 max_pages
- max_depth 上限：_run_full_site 不推入超过 max_depth 的链接
- 已访问 URL 去重：同一 URL 不被重复访问
- Spec 自动推断：_apply_spec_inference 补充缺失字段
- crawl_mode 在结果中被正确包含
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


# ---------------------------------------------------------------------------
# Fixtures / 工具函数
# ---------------------------------------------------------------------------

def _make_spec(crawl_mode='single_page', max_pages=5, max_depth=2, **extra):
    spec = {
        'task_id': 'test_task',
        'task_name': '测试任务',
        'target_url': 'https://example.com/',
        'goal': '测试',
        'max_iterations': 3,
        'crawl_mode': crawl_mode,
        'max_pages': max_pages,
        'max_depth': max_depth,
        'completion_gate': [],
    }
    spec.update(extra)
    return spec


def _make_sense_result(page_type='list'):
    return {
        'success': True,
        'structure': {
            'page_type': page_type,
            'complexity': 'simple',
            'pagination_type': 'none',
            'main_content_selector': 'ul',
        },
        'features': {
            'page_type': page_type,
            'is_spa': False,
            'has_pagination': False,
            'container_info': {'found': True, 'tag': 'ul', 'count': 5, 'similarity': 0.9},
            'anti_bot_level': 'none',
        },
        'anti_bot_detected': False,
        'anti_bot_info': {'detected': False},
        'html_snapshot': '<html><body></body></html>',
        'screenshot': None,
    }


def _make_plan_result():
    return {
        'success': True,
        'selectors': {'items': 'ul li'},
        'strategy': {'strategy_type': 'css'},
        'generated_code': None,
    }


def _make_act_result(n=3):
    return {
        'success': True,
        'extracted_data': [{'title': f'item {i}'} for i in range(n)],
        'extraction_metrics': {},
    }


def _make_verify_result(score=0.9):
    return {
        'success': True,
        'quality_score': score,
        'valid_items': 3,
        'verification_result': {},
    }


def _make_judge_result(decision='complete'):
    return {
        'success': True,
        'decision': decision,
        'reasoning': 'test',
        'suggestions': [],
    }


# ---------------------------------------------------------------------------
# _apply_spec_inference 单元测试（直接测试方法）
# ---------------------------------------------------------------------------

class TestApplySpecInference:
    """直接测试 SelfCrawlingAgent._apply_spec_inference"""

    def _make_agent_with_spec(self, spec):
        """构造一个最小化的 SelfCrawlingAgent，不触发 __init__ 副作用"""
        from src.main import SelfCrawlingAgent
        agent = object.__new__(SelfCrawlingAgent)
        agent.spec = spec
        return agent

    def test_patches_missing_crawl_mode(self):
        """spec 中无 crawl_mode 时应被补充"""
        spec = {'task_id': 'x'}
        agent = self._make_agent_with_spec(spec)
        features = {
            'page_type': 'list',
            'is_spa': False,
            'has_pagination': True,
            'container_info': {'found': True, 'tag': 'ul', 'count': 5, 'similarity': 0.9},
            'anti_bot_level': 'none',
        }
        agent._apply_spec_inference(features)
        assert 'crawl_mode' in agent.spec

    def test_does_not_overwrite_existing_crawl_mode(self):
        """spec 中已有 crawl_mode 时不覆盖"""
        spec = {'task_id': 'x', 'crawl_mode': 'full_site'}
        agent = self._make_agent_with_spec(spec)
        features = {
            'page_type': 'detail',
            'is_spa': False,
            'has_pagination': False,
            'container_info': {'found': False, 'tag': None, 'count': 0, 'similarity': 0.0},
            'anti_bot_level': 'none',
        }
        agent._apply_spec_inference(features)
        assert agent.spec['crawl_mode'] == 'full_site'

    def test_patches_missing_max_pages(self):
        """spec 中无 max_pages 时应被补充"""
        spec = {'task_id': 'x'}
        agent = self._make_agent_with_spec(spec)
        features = {
            'page_type': 'detail',
            'is_spa': False,
            'has_pagination': False,
            'container_info': {'found': False, 'tag': None, 'count': 0, 'similarity': 0.0},
            'anti_bot_level': 'none',
        }
        agent._apply_spec_inference(features)
        assert 'max_pages' in agent.spec

    def test_patches_missing_max_depth(self):
        """spec 中无 max_depth 时应被补充"""
        spec = {'task_id': 'x'}
        agent = self._make_agent_with_spec(spec)
        features = {
            'page_type': 'detail',
            'is_spa': False,
            'has_pagination': False,
            'container_info': {'found': False, 'tag': None, 'count': 0, 'similarity': 0.0},
            'anti_bot_level': 'none',
        }
        agent._apply_spec_inference(features)
        assert 'max_depth' in agent.spec


# ---------------------------------------------------------------------------
# _run_full_site max_pages 和 visited 去重测试
# ---------------------------------------------------------------------------

class TestRunFullSite:
    """测试 _run_full_site 方法的关键行为"""

    def _make_agent(self, spec):
        """构造最小化 SelfCrawlingAgent"""
        from src.main import SelfCrawlingAgent
        agent = object.__new__(SelfCrawlingAgent)
        agent.spec = spec
        agent.state_manager = MagicMock()
        agent.state_manager.update_state_sync = MagicMock()
        agent.state_manager.get_state = MagicMock(return_value={})
        agent.evidence_storage = MagicMock()
        agent.evidence_storage.save_data = MagicMock()
        agent.llm_client = None

        # mock browser: navigate 成功，get_html 返回简单 HTML
        browser = MagicMock()
        browser.navigate = AsyncMock()
        browser.get_html = AsyncMock(return_value='<html><body></body></html>')
        browser.take_screenshot = AsyncMock(return_value=None)
        browser.get_current_url = AsyncMock(return_value='https://example.com/')
        browser.page = None
        agent.browser = browser

        return agent

    def _mock_agent_pool(self, agent, sense=None, plan=None, act=None):
        """为 agent 注入 mock agent_pool"""
        sense = sense or _make_sense_result()
        plan = plan or _make_plan_result()
        act = act or _make_act_result(2)

        pool = MagicMock()

        async def execute_capability(cap, ctx):
            if cap == 'sense':
                return sense
            elif cap == 'plan':
                return plan
            elif cap == 'act':
                return act
            return {'success': True}

        pool.execute_capability = execute_capability
        agent.agent_pool = pool

    async def test_respects_max_pages(self):
        """full_site 模式下 pages_visited 不超过 max_pages"""
        spec = _make_spec('full_site', max_pages=3, max_depth=5)
        agent = self._make_agent(spec)
        self._mock_agent_pool(agent)

        # ExploreAgent mock：每页返回 5 个新链接
        call_count = [0]

        async def mock_explore_execute(ctx):
            frontier = ctx.get('frontier')
            base = 'https://example.com'
            depth = ctx.get('depth', 0)
            if frontier:
                for i in range(5):
                    frontier.push(f'{base}/page/{call_count[0]}_{i}', depth=depth + 1)
            call_count[0] += 1
            return {'success': True, 'links': [], 'categorized': {}, 'count': 0,
                    'next_depth': depth + 1, 'sitemap_links': [], 'pushed_to_frontier': 5}

        with patch('src.agents.base.ExploreAgent') as MockExplore:
            mock_instance = MagicMock()
            mock_instance.execute = mock_explore_execute
            MockExplore.return_value = mock_instance

            result = await agent._run_full_site('https://example.com/')

        assert result['pages_visited'] <= 3
        assert result['crawl_mode'] == 'full_site'

    async def test_visited_dedup(self):
        """同一 URL 不被重复访问"""
        spec = _make_spec('full_site', max_pages=10, max_depth=2)
        agent = self._make_agent(spec)
        self._mock_agent_pool(agent)

        visited_urls = []

        async def track_navigate(url):
            visited_urls.append(url)

        agent.browser.navigate = track_navigate

        # ExploreAgent mock：每次推入同一批 URL（测试去重）
        async def mock_explore_execute(ctx):
            frontier = ctx.get('frontier')
            if frontier:
                # 总是推入相同的 URL
                for i in range(3):
                    frontier.push(f'https://example.com/dup/{i}', depth=1)
            return {'success': True, 'links': [], 'categorized': {}, 'count': 0,
                    'next_depth': 1, 'sitemap_links': [], 'pushed_to_frontier': 3}

        with patch('src.agents.base.ExploreAgent') as MockExplore:
            mock_instance = MagicMock()
            mock_instance.execute = mock_explore_execute
            MockExplore.return_value = mock_instance

            await agent._run_full_site('https://example.com/')

        # 每个 URL 应只被访问一次
        assert len(visited_urls) == len(set(visited_urls)), \
            f"存在重复访问: {[u for u in visited_urls if visited_urls.count(u) > 1]}"

    async def test_max_depth_honored_by_frontier(self):
        """超过 max_depth 的链接不入队（由 CrawlFrontier 过滤）"""
        spec = _make_spec('full_site', max_pages=50, max_depth=1)
        agent = self._make_agent(spec)
        self._mock_agent_pool(agent)

        pushed_depths = []

        async def mock_explore_execute(ctx):
            frontier = ctx.get('frontier')
            depth = ctx.get('depth', 0)
            if frontier:
                # 推入当前深度+1 和深度+2（应被过滤）
                r1 = frontier.push(
                    f'https://example.com/d{depth+1}_{id(ctx)}', depth=depth + 1
                )
                r2 = frontier.push(
                    f'https://example.com/d{depth+2}_{id(ctx)}', depth=depth + 2
                )
                pushed_depths.append((depth + 1, r1))
                pushed_depths.append((depth + 2, r2))
            return {'success': True, 'links': [], 'categorized': {}, 'count': 0,
                    'next_depth': depth + 1, 'sitemap_links': [], 'pushed_to_frontier': 0}

        with patch('src.agents.base.ExploreAgent') as MockExplore:
            mock_instance = MagicMock()
            mock_instance.execute = mock_explore_execute
            MockExplore.return_value = mock_instance

            await agent._run_full_site('https://example.com/')

        # 深度 > max_depth(1) 的入队应返回 False
        for depth, success in pushed_depths:
            if depth > 1:
                assert success is False, f"depth={depth} 的 URL 不应入队（max_depth=1）"

    async def test_returns_success_with_extracted_data(self):
        """_run_full_site 返回值包含 success=True 和 extracted_data"""
        spec = _make_spec('full_site', max_pages=2, max_depth=0)
        agent = self._make_agent(spec)
        self._mock_agent_pool(agent, act=_make_act_result(4))

        with patch('src.agents.base.ExploreAgent') as MockExplore:
            mock_instance = MagicMock()
            mock_instance.execute = AsyncMock(return_value={
                'success': True, 'links': [], 'categorized': {}, 'count': 0,
                'next_depth': 1, 'sitemap_links': [], 'pushed_to_frontier': 0,
            })
            MockExplore.return_value = mock_instance

            result = await agent._run_full_site('https://example.com/')

        assert result['success'] is True
        assert len(result['extracted_data']) >= 0  # 至少不崩溃
        assert 'frontier_stats' in result


# ---------------------------------------------------------------------------
# _run_single_or_multi 测试
# ---------------------------------------------------------------------------

class TestRunSingleOrMulti:
    """测试 _run_single_or_multi 的基本行为"""

    def _make_agent(self, spec):
        from src.main import SelfCrawlingAgent
        agent = object.__new__(SelfCrawlingAgent)
        agent.spec = spec
        agent.state_manager = MagicMock()
        agent.state_manager.update_state_sync = MagicMock()
        agent.state_manager.get_state = MagicMock(return_value={
            'html_snapshot': '<html></html>',
            'sense_analysis': {'page_type': 'list'},
            'quality_score': 0.9,
            'sample_data': [],
        })
        agent.state_manager.get_history = MagicMock(return_value=[])
        agent.evidence_storage = MagicMock()
        agent.evidence_storage.save_data = MagicMock()
        agent.evidence_collector = MagicMock()
        agent.evidence_collector.collect_plan = MagicMock()
        agent.risk_monitor = MagicMock()
        agent.risk_monitor.check_metrics = MagicMock(return_value=[])
        agent.completion_gate = MagicMock()
        agent.completion_gate.check = MagicMock(return_value=False)
        agent.completion_gate.get_failed_gates = MagicMock(return_value=[])
        agent.llm_client = None

        browser = MagicMock()
        browser.navigate = AsyncMock()
        browser.get_html = AsyncMock(return_value='<html><body></body></html>')
        browser.take_screenshot = AsyncMock(return_value=None)
        browser.get_current_url = AsyncMock(return_value='https://example.com/')
        browser.page = None
        agent.browser = browser

        return agent

    def _mock_agent_pool(self, agent, judge_decision='complete',
                          sense=None, plan=None, act=None, verify=None):
        sense = sense or _make_sense_result()
        plan = plan or _make_plan_result()
        act = act or _make_act_result(3)
        verify = verify or _make_verify_result(0.9)

        async def execute_capability(cap, ctx):
            if cap == 'sense':
                return sense
            elif cap == 'plan':
                return plan
            elif cap == 'act':
                return act
            elif cap == 'verify':
                return verify
            elif cap == 'judge':
                return _make_judge_result(judge_decision)
            elif cap == 'reflect':
                return {'success': True, 'new_selectors': {}}
            return {'success': True}

        pool = MagicMock()
        pool.execute_capability = execute_capability
        agent.agent_pool = pool

    async def test_single_page_returns_crawl_mode_in_result(self):
        """single_page 结果中包含 crawl_mode=single_page"""
        spec = _make_spec('single_page')
        agent = self._make_agent(spec)
        self._mock_agent_pool(agent, judge_decision='complete')

        result = await agent._run_single_or_multi('https://example.com/', 'single_page')
        assert result.get('crawl_mode') == 'single_page'

    async def test_multi_page_returns_crawl_mode_in_result(self):
        """multi_page 结果中包含 crawl_mode=multi_page"""
        spec = _make_spec('multi_page')
        agent = self._make_agent(spec)
        self._mock_agent_pool(agent, judge_decision='complete')

        result = await agent._run_single_or_multi('https://example.com/', 'multi_page')
        assert result.get('crawl_mode') == 'multi_page'

    async def test_judge_complete_returns_success(self):
        """judge 决策 complete → success=True"""
        spec = _make_spec('single_page')
        agent = self._make_agent(spec)
        self._mock_agent_pool(agent, judge_decision='complete')

        result = await agent._run_single_or_multi('https://example.com/', 'single_page')
        assert result['success'] is True

    async def test_judge_terminate_returns_failure(self):
        """judge 决策 terminate → success=False"""
        spec = _make_spec('single_page')
        agent = self._make_agent(spec)
        self._mock_agent_pool(agent, judge_decision='terminate')

        result = await agent._run_single_or_multi('https://example.com/', 'single_page')
        assert result['success'] is False

    async def test_max_iterations_limit(self):
        """超过 max_iterations 时返回 success=False 并标注原因"""
        spec = _make_spec('single_page')
        spec['max_iterations'] = 2
        agent = self._make_agent(spec)
        # judge 总是 reflect_and_retry，使循环耗尽
        self._mock_agent_pool(agent, judge_decision='reflect_and_retry')

        result = await agent._run_single_or_multi('https://example.com/', 'single_page')
        assert result['success'] is False
        assert result['iterations'] == 2

    async def test_spec_inference_applied_on_first_iteration(self):
        """首次迭代后应调用 _apply_spec_inference"""
        spec = _make_spec('single_page')
        # 移除 crawl_mode 触发推断
        del spec['crawl_mode']
        agent = self._make_agent(spec)
        self._mock_agent_pool(agent, judge_decision='complete')

        applied = []

        def fake_apply(features):
            applied.append(features)
            agent.spec['crawl_mode'] = 'single_page'

        agent._apply_spec_inference = fake_apply

        await agent._run_single_or_multi('https://example.com/', 'single_page')
        assert len(applied) == 1, "应恰好调用一次 _apply_spec_inference"

    async def test_spec_inference_not_applied_twice(self):
        """多次迭代时 _apply_spec_inference 只被调用一次"""
        spec = _make_spec('single_page')
        spec['max_iterations'] = 3
        agent = self._make_agent(spec)
        # 前两次 reflect_and_retry，第三次 complete
        decisions = ['reflect_and_retry', 'reflect_and_retry', 'complete']
        idx = [0]

        async def execute_capability(cap, ctx):
            if cap == 'sense':
                return _make_sense_result()
            elif cap == 'plan':
                return _make_plan_result()
            elif cap == 'act':
                return _make_act_result(1)
            elif cap == 'verify':
                return _make_verify_result()
            elif cap == 'judge':
                d = decisions[min(idx[0], len(decisions) - 1)]
                idx[0] += 1
                return _make_judge_result(d)
            elif cap == 'reflect':
                return {'success': True, 'new_selectors': {}}
            return {'success': True}

        pool = MagicMock()
        pool.execute_capability = execute_capability
        agent.agent_pool = pool

        applied = []

        def fake_apply(features):
            applied.append(features)

        agent._apply_spec_inference = fake_apply

        await agent._run_single_or_multi('https://example.com/', 'single_page')
        assert len(applied) == 1, "多次迭代时 _apply_spec_inference 应只被调用一次"


# ---------------------------------------------------------------------------
# run() 主入口：模式分发
# ---------------------------------------------------------------------------

class TestRunDispatch:
    """测试 run() 按 crawl_mode 正确分发到对应子方法"""

    def _make_initialized_agent(self, spec):
        """构造已初始化（跳过 initialize()）的 agent"""
        from src.main import SelfCrawlingAgent
        agent = object.__new__(SelfCrawlingAgent)
        agent.spec = spec
        agent.task_id = spec['task_id']
        agent.llm_client = None
        agent.browser = MagicMock()
        agent.browser.start = AsyncMock()
        agent.browser.stop = AsyncMock()
        agent.browser.navigate = AsyncMock()
        agent.state_manager = MagicMock()
        agent.state_manager.update_state_sync = MagicMock()
        agent.state_manager.get_state = MagicMock(return_value={})
        agent.state_storage = MagicMock()
        agent.state_storage.save_state = MagicMock()
        agent.evidence_collector = MagicMock()
        agent.evidence_collector.save_index = MagicMock()
        agent.evidence_storage = MagicMock()
        return agent

    async def test_full_site_dispatches_to_run_full_site(self):
        """crawl_mode=full_site 时 run() 应调用 _run_full_site"""
        spec = _make_spec('full_site')
        agent = self._make_initialized_agent(spec)
        agent.initialize = AsyncMock()

        called = []

        async def fake_full_site(url):
            called.append('full_site')
            return {'success': True, 'crawl_mode': 'full_site',
                    'extracted_data': [], 'pages_visited': 0,
                    'frontier_stats': {}, 'quality_score': 0.0}

        async def fake_single(url, mode):
            called.append('single')
            return {'success': True, 'crawl_mode': mode, 'extracted_data': [],
                    'quality_score': 0.0, 'iterations': 1}

        agent._run_full_site = fake_full_site
        agent._run_single_or_multi = fake_single

        await agent.run()
        assert called == ['full_site']

    async def test_single_page_dispatches_to_run_single_or_multi(self):
        """crawl_mode=single_page 时 run() 应调用 _run_single_or_multi"""
        spec = _make_spec('single_page')
        agent = self._make_initialized_agent(spec)
        agent.initialize = AsyncMock()

        called = []

        async def fake_full_site(url):
            called.append('full_site')
            return {'success': True, 'crawl_mode': 'full_site',
                    'extracted_data': [], 'pages_visited': 0,
                    'frontier_stats': {}, 'quality_score': 0.0}

        async def fake_single(url, mode):
            called.append('single')
            return {'success': True, 'crawl_mode': mode, 'extracted_data': [],
                    'quality_score': 0.0, 'iterations': 1}

        agent._run_full_site = fake_full_site
        agent._run_single_or_multi = fake_single

        await agent.run()
        assert called == ['single']

    async def test_multi_page_dispatches_to_run_single_or_multi(self):
        """crawl_mode=multi_page 时 run() 应调用 _run_single_or_multi"""
        spec = _make_spec('multi_page')
        agent = self._make_initialized_agent(spec)
        agent.initialize = AsyncMock()

        called = []

        async def fake_full_site(url):
            called.append('full_site')
            return {'success': True, 'crawl_mode': 'full_site',
                    'extracted_data': [], 'pages_visited': 0,
                    'frontier_stats': {}, 'quality_score': 0.0}

        async def fake_single(url, mode):
            called.append('single')
            return {'success': True, 'crawl_mode': mode, 'extracted_data': [],
                    'quality_score': 0.0, 'iterations': 1}

        agent._run_full_site = fake_full_site
        agent._run_single_or_multi = fake_single

        await agent.run()
        assert called == ['single']
