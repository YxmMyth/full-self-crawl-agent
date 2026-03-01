#!/bin/bash
# build.sh - 构建优化的Docker镜像

set -e

IMAGE_NAME=${1:-self-crawling-agent:latest}

echo "构建Docker镜像: $IMAGE_NAME"

# 构建镜像
docker build -t "$IMAGE_NAME" .

echo "镜像构建完成: $IMAGE_NAME"

# 可选：推送镜像
if [ "$2" == "push" ]; then
    echo "推送镜像到仓库..."
    docker push "$IMAGE_NAME"
    echo "推送完成"
fi