"""
Execution environments for the crawling agent.

Provides a uniform interface for executing commands either locally
or in a Docker container.
"""

from .docker_env import DockerEnvironment, DockerEnvironmentConfig
from .local_env import LocalEnvironment, LocalEnvironmentConfig

__all__ = [
    'DockerEnvironment', 'DockerEnvironmentConfig',
    'LocalEnvironment', 'LocalEnvironmentConfig',
]
