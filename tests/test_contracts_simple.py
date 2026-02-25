"""
契约层单元测试
"""

import pytest
import json
from datetime import datetime
import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_spec_factory():
    """测试Spec工厂"""
    from config.contracts import ContractFactory
    
    spec = ContractFactory.create_spec(
        goal="Test crawl",
        target_url="https://example.com",
        max_execution_time=300
    )
    
    assert spec['version'] == 'v1'
    assert spec['freeze'] is True
    assert spec['goal'] == 'Test crawl'


def test_spec_validation():
    """测试Spec验证"""
    from config.contracts import ContractValidator
    
    spec = {
        'version': 'v1',
        'freeze': True,
        'goal': 'Test',
        'completion_gate': ['html_snapshot_exists']
    }
    
    assert ContractValidator.validate_spec(spec) is True


def test_spec_validation_missing():
    """测试Spec验证 - 缺少必填字段"""
    from config.contracts import ContractValidator
    
    spec = {
        'version': 'v1',
        'goal': 'Test'
    }
    
    try:
        ContractValidator.validate_spec(spec)
        assert False, "应该抛出异常"
    except ValueError as e:
        assert "freeze" in str(e)


def test_state_creation():
    """测试状态创建"""
    from config.contracts import ContractFactory
    
    spec = ContractFactory.create_spec(
        goal="Test",
        target_url="https://example.com"
    )
    
    state = ContractFactory.create_initial_state(
        task_id="test_001",
        url="https://example.com",
        goal="Test",
        spec=spec
    )
    
    assert state['task_id'] == 'test_001'
    assert state['stage'] == 'initialized'


def test_routing_validation():
    """测试路由决策验证"""
    from config.contracts import ContractValidator
    
    decision = {
        'strategy': 'direct_crawl',
        'capabilities': ['sense', 'plan', 'act'],
        'expected_success_rate': 0.95
    }
    
    assert ContractValidator.validate_routing_decision(decision) is True


def test_routing_invalid_rate():
    """测试路由决策验证 - 无效的成功率"""
    from config.contracts import ContractValidator
    
    decision = {
        'strategy': 'direct_crawl',
        'capabilities': ['sense', 'plan', 'act'],
        'expected_success_rate': 1.5
    }
    
    try:
        ContractValidator.validate_routing_decision(decision)
        assert False, "应该抛出异常"
    except ValueError as e:
        assert "between 0 and 1" in str(e)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
