#!/usr/bin/env python3
"""
run_task.py - 简化的一键执行脚本
支持本地执行或容器化执行
"""

import subprocess
import sys
import json
import tempfile
from pathlib import Path
import argparse
import os


def run_single_task(spec_path: str, use_docker: bool = True, **kwargs):
    """
    一键运行单个爬取任务
    """
    if use_docker:
        return run_in_docker(spec_path, **kwargs)
    else:
        return run_locally(spec_path, **kwargs)


def run_in_docker(spec_path: str, **kwargs):
    """
    在Docker中运行任务
    """
    try:
        import docker
    except ImportError:
        print("错误: 请先安装docker库: pip install docker")
        return {'success': False, 'error': 'docker not installed'}

    from src.container_manager import ContainerizedTaskManager, TaskConfig

    client = docker.from_env()
    manager = ContainerizedTaskManager(client)

    # 加载spec
    with open(spec_path, 'r', encoding='utf-8') as f:
        spec_data = json.load(f)

    # 生成任务ID
    import hashlib
    task_hash = hashlib.md5(json.dumps(spec_data, sort_keys=True).encode()).hexdigest()[:8]
    task_id = f"task_{task_hash}"

    # 创建任务配置
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

    return manager.run_task(task_config)


def run_locally(spec_path: str, **kwargs):
    """
    本地运行任务（不使用Docker）
    """
    # 使用主程序的本地执行模式
    cmd = [sys.executable, '-m', 'src.main', spec_path]

    if kwargs.get('debug'):
        cmd.append('--debug')

    print(f"本地执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        # 简单解析输出以获得结果
        success = result.returncode == 0
        output = {
            'success': success,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }

        return output
    except Exception as e:
        return {'success': False, 'error': str(e)}


def ensure_docker_image_exists(image_name: str):
    """
    确保Docker镜像存在，如果不存在则提示用户构建
    """
    try:
        import docker
        client = docker.from_env()

        try:
            client.images.get(image_name)
            print(f"镜像 {image_name} 已存在")
            return True
        except docker.errors.ImageNotFound:
            print(f"镜像 {image_name} 不存在")
            response = input(f"是否要构建镜像? (y/N): ")
            if response.lower() == 'y':
                # 构建镜像
                print("构建Docker镜像...")
                result = subprocess.run([
                    'docker', 'build', '-t', image_name, '.'
                ], check=True, capture_output=True, text=True)
                print("镜像构建完成")
                return True
            else:
                print("请先构建镜像或使用本地执行模式")
                return False
    except ImportError:
        print("Docker库未安装，无法验证镜像")
        return False


def main():
    parser = argparse.ArgumentParser(description='一键执行爬取任务')
    parser.add_argument('spec_file', help='规格文件路径')
    parser.add_argument('--local', action='store_true', help='本地执行模式（不使用Docker）')
    parser.add_argument('--docker', action='store_true', default=True, help='Docker执行模式（默认）')
    parser.add_argument('--image', default='self-crawling-agent:latest', help='Docker镜像名称')
    parser.add_argument('--memory', default='2g', help='内存限制')
    parser.add_argument('--cpu', type=int, default=1024, help='CPU份额')
    parser.add_argument('--timeout', type=int, default=3600, help='超时时间（秒）')
    parser.add_argument('--debug', action='store_true', help='调试模式')

    args = parser.parse_args()

    if not Path(args.spec_file).exists():
        print(f"错误: 规格文件不存在 - {args.spec_file}")
        sys.exit(1)

    # 如果使用Docker模式，确保镜像存在
    if args.docker and not args.local:
        if not ensure_docker_image_exists(args.image):
            sys.exit(1)

    # 执行任务
    result = run_single_task(
        spec_path=args.spec_file,
        use_docker=(args.docker and not args.local),
        image=args.image,
        memory_limit=args.memory,
        cpu_shares=args.cpu,
        timeout=args.timeout,
        debug=args.debug
    )

    print("\n执行结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()