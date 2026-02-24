"""
策略管理器 - 战略层
管理边界和约束策略，防止越界行为
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class PolicyLevel(str, Enum):
    """策略级别"""
    CRITICAL = "critical"  # 严重违规
    WARNING = "warning"  # 警告
    INFO = "info"  # 信息


@dataclass
class PolicyRule:
    """策略规则"""
    name: str
    level: PolicyLevel
    condition: str  # 条件表达式
    action: str  # 违规时的动作
    description: str


class PolicyManager:
    """
    策略管理器

    职责：
    - 管理安全策略和边界约束
    - 检查行为是否合规
    - 防止越界行为（如无限递归、代码注入等）

    策略类型：
    - 代码安全策略
    - 资源使用策略
    - 递归深度策略
    - 网络访问策略
    - 执行时间策略
    """

    def __init__(self, config_path: Optional[str] = None):
        self.policies: Dict[str, List[PolicyRule]] = {}
        self._load_default_policies()
        if config_path:
            self._load_policies_from_config(config_path)

    def _load_default_policies(self):
        """加载默认策略"""
        # 代码安全策略
        self.policies['code_security'] = [
            PolicyRule(
                name='no_os_system',
                level=PolicyLevel.CRITICAL,
                condition='code contains "os.system"',
                action='reject',
                description='禁止使用 os.system 执行系统命令'
            ),
            PolicyRule(
                name='no_eval',
                level=PolicyLevel.CRITICAL,
                condition='code contains "eval("',
                action='reject',
                description='禁止使用 eval 执行动态代码'
            ),
            PolicyRule(
                name='no_exec',
                level=PolicyLevel.CRITICAL,
                condition='code contains "exec("',
                action='reject',
                description='禁止使用 exec 执行动态代码'
            ),
            PolicyRule(
                name='no_import_os',
                level=PolicyLevel.WARNING,
                condition='code contains "import os"',
                action='warn',
                description='谨慎使用 os 模块'
            )
        ]

        # 资源使用策略
        self.policies['resource'] = [
            PolicyRule(
                name='max_memory',
                level=PolicyLevel.CRITICAL,
                condition='memory > 512MB',
                action='terminate',
                description='内存使用不得超过 512MB'
            ),
            PolicyRule(
                name='max_cpu',
                level=PolicyLevel.WARNING,
                condition='cpu > 80%',
                action='warn',
                description='CPU 使用率超过 80%'
            )
        ]

        # 递归深度策略
        self.policies['recursion'] = [
            PolicyRule(
                name='max_depth',
                level=PolicyLevel.CRITICAL,
                condition='depth > 5',
                action='reject',
                description='递归深度不得超过 5 层'
            ),
            PolicyRule(
                name='max_iterations',
                level=PolicyLevel.CRITICAL,
                condition='iterations > 100',
                action='terminate',
                description='循环次数不得超过 100 次'
            )
        ]

        # 网络访问策略
        self.policies['network'] = [
            PolicyRule(
                name='allowed_domains',
                level=PolicyLevel.CRITICAL,
                condition='url not in allowed_domains',
                action='reject',
                description='只能访问允许的域名'
            ),
            PolicyRule(
                name='max_requests',
                level=PolicyLevel.WARNING,
                condition='requests > 1000',
                action='warn',
                description='请求次数超过 1000 次'
            )
        ]

        # 执行时间策略
        self.policies['time'] = [
            PolicyRule(
                name='max_execution_time',
                level=PolicyLevel.CRITICAL,
                condition='execution_time > 300s',
                action='terminate',
                description='单次执行时间不得超过 300 秒'
            )
        ]

    def _load_policies_from_config(self, config_path: str):
        """从配置文件加载策略"""
        # TODO: 实现从配置文件加载策略
        pass

    def check_code(self, code: str) -> Dict[str, Any]:
        """
        检查代码是否合规

        Returns:
            {
                'allowed': bool,
                'violations': List[Dict],
                'warnings': List[Dict]
            }
        """
        violations = []
        warnings = []

        # 检查代码安全策略
        security_policies = self.policies.get('code_security', [])
        for policy in security_policies:
            if self._evaluate_condition(policy.condition, code=code):
                if policy.level == PolicyLevel.CRITICAL:
                    violations.append({
                        'rule': policy.name,
                        'level': policy.level.value,
                        'description': policy.description,
                        'action': policy.action
                    })
                elif policy.level == PolicyLevel.WARNING:
                    warnings.append({
                        'rule': policy.name,
                        'level': policy.level.value,
                        'description': policy.description,
                        'action': policy.action
                    })

        return {
            'allowed': len(violations) == 0,
            'violations': violations,
            'warnings': warnings
        }

    def check_resource_usage(self, resource_data: Dict[str, Any]) -> Dict[str, Any]:
        """检查资源使用是否合规"""
        violations = []

        resource_policies = self.policies.get('resource', [])
        for policy in resource_policies:
            if self._evaluate_condition(policy.condition, **resource_data):
                violations.append({
                    'rule': policy.name,
                    'level': policy.level.value,
                    'description': policy.description
                })

        return {
            'allowed': len(violations) == 0,
            'violations': violations
        }

    def check_recursion(self, depth: int, iterations: int) -> Dict[str, Any]:
        """检查递归深度和迭代次数"""
        violations = []

        recursion_policies = self.policies.get('recursion', [])
        for policy in recursion_policies:
            context = {'depth': depth, 'iterations': iterations}
            if self._evaluate_condition(policy.condition, **context):
                violations.append({
                    'rule': policy.name,
                    'level': policy.level.value,
                    'description': policy.description
                })

        return {
            'allowed': len(violations) == 0,
            'violations': violations
        }

    def _evaluate_condition(self, condition: str, **context) -> bool:
        """
        评估条件表达式

        注意：这是一个简化的实现
        实际应用中应该使用更安全的表达式解析器
        """
        try:
            # 简单的条件评估
            # 实际应用中应该使用 ast.literal_eval 或专用表达式引擎
            if 'code contains' in condition:
                keyword = condition.split('"')[1]
                return keyword in context.get('code', '')

            # 其他条件类型...
            return False
        except Exception:
            return False

    def get_all_policies(self) -> Dict[str, List[Dict]]:
        """获取所有策略"""
        result = {}
        for category, rules in self.policies.items():
            result[category] = [self._rule_to_dict(rule) for rule in rules]
        return result

    def _rule_to_dict(self, rule: PolicyRule) -> Dict[str, Any]:
        """将规则转换为字典"""
        return {
            'name': rule.name,
            'level': rule.level.value,
            'condition': rule.condition,
            'action': rule.action,
            'description': rule.description
        }
