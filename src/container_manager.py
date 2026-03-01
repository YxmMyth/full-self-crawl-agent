"""
Containerized Task Manager
管理在Docker容器中运行的SelfCrawlingAgent任务
"""

import docker
import json
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TaskConfig:
    """任务配置"""
    task_id: str
    spec_dict: dict
    timeout: int = 3600
    cpu_shares: int = 1024
    memory_limit: str = '2g'
    max_retries: int = 3
    image: str = 'self-crawling-agent:latest'


class ContainerizedTaskManager:
    """容器化任务管理器"""

    def __init__(self, docker_client: docker.DockerClient = None):
        self.docker_client = docker_client or docker.from_env()
        self.logger = logging.getLogger(__name__)

    def run_task(self, task_config: TaskConfig) -> Dict[str, Any]:
        """在容器中运行任务"""
        spec_file = None
        output_dir = None

        try:
            # 1. 创建任务配置文件
            spec_file = self._create_spec_file(task_config.spec_dict, task_config.task_id)

            # 2. 准备输出目录
            output_dir = self._prepare_output_directory(task_config.task_id)

            # 3. 配置容器参数
            container_params = self._build_container_config(task_config, spec_file, output_dir)

            # 4. 运行容器
            self.logger.info(f"启动任务容器: {task_config.task_id}")
            container = self.docker_client.containers.run(**container_params)

            # 5. 等待完成
            self.logger.info(f"等待任务完成: {task_config.task_id}")
            result = self._wait_for_completion(container, task_config.timeout)

            # 6. 收集结果
            output_data = self._collect_output(output_dir)

            success = result['StatusCode'] == 0
            self.logger.info(f"任务完成: {task_config.task_id}, 成功: {success}")

            return {
                'success': success,
                'output': output_data,
                'exit_code': result['StatusCode'],
                'logs': container.logs().decode('utf-8'),
                'task_id': task_config.task_id,
                'execution_time': result.get('Time', time.time())
            }

        except Exception as e:
            self.logger.error(f"任务执行失败 {task_config.task_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'task_id': task_config.task_id
            }
        finally:
            # 清理资源
            self._cleanup_temp_files(spec_file, output_dir)

    def _create_spec_file(self, spec_dict: dict, task_id: str) -> str:
        """创建任务规格文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{task_id}.json', delete=False) as f:
            json.dump(spec_dict, f, ensure_ascii=False, indent=2)
            return f.name

    def _prepare_output_directory(self, task_id: str) -> str:
        """准备输出目录"""
        output_path = f'/tmp/crawler_outputs/{task_id}'
        Path(output_path).mkdir(parents=True, exist_ok=True)
        return output_path

    def _build_container_config(self, task_config: TaskConfig, spec_file: str, output_dir: str) -> dict:
        """构建容器配置"""
        return {
            'image': task_config.image,
            'volumes': {
                spec_file: {'bind': '/app/task_spec.json', 'mode': 'ro'},
                output_dir: {'bind': '/app/output', 'mode': 'rw'},
            },
            'environment': {
                'TASK_SPEC_PATH': '/app/task_spec.json',
                'TASK_ID': task_config.task_id,
                'TIMEOUT': str(task_config.timeout),
                'CPU_SHARES': str(task_config.cpu_shares),
                'MEMORY_LIMIT': task_config.memory_limit,
                'LOG_LEVEL': 'INFO'
            },
            'network_mode': 'bridge',
            'mem_limit': task_config.memory_limit,
            'cpu_shares': task_config.cpu_shares,
            'pids_limit': 50,
            'remove': True,  # 容器结束后自动删除
            'detach': False,   # 同步运行
            'stdout': True,
            'stderr': True
        }

    def _wait_for_completion(self, container, timeout: int) -> dict:
        """等待容器完成"""
        try:
            return container.wait(timeout=timeout + 60)  # 给一些额外时间
        except Exception as e:
            # 强制停止容器
            try:
                container.stop(timeout=10)
            except:
                pass
            raise e

    def _collect_output(self, output_dir: str) -> dict:
        """收集输出结果"""
        output_file = Path(output_dir) / 'results.json'
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _cleanup_temp_files(self, spec_file: str, output_dir: str):
        """清理临时文件"""
        if spec_file and Path(spec_file).exists():
            Path(spec_file).unlink()
        if output_dir:
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)

    def run_single_task(self, spec_path: str, **kwargs) -> Dict[str, Any]:
        """
        一键在容器中运行单个爬取任务
        """
        # 加载spec
        with open(spec_path, 'r', encoding='utf-8') as f:
            spec_data = json.load(f)

        # 生成任务ID
        import hashlib
        task_hash = hashlib.md5(json.dumps(spec_data, sort_keys=True).encode()).hexdigest()[:8]
        task_id = f"task_{task_hash}"

        # 设置默认参数
        task_config = TaskConfig(
            task_id=task_id,
            spec_dict=spec_data,
            timeout=kwargs.get('timeout', 3600),
            cpu_shares=kwargs.get('cpu_shares', 1024),
            memory_limit=kwargs.get('memory_limit', '2g'),
            image=kwargs.get('image', 'self-crawling-agent:latest')
        )

        print(f"启动任务: {task_config.task_id}")
        print(f"内存限制: {task_config.memory_limit}, CPU份额: {task_config.cpu_shares}")

        return self.run_task(task_config)


def run_single_task_cli(spec_path: str, **kwargs):
    """
    CLI入口函数 - 运行单个任务
    """
    manager = ContainerizedTaskManager()
    result = manager.run_single_task(spec_path, **kwargs)

    print(f"任务执行结果: {result.get('success', False)}")
    if result.get('output'):
        extracted_count = len(result['output'].get('extracted_data', []))
        print(f"提取数据: {extracted_count} 条")
    if result.get('error'):
        print(f"错误: {result['error']}")

    return result