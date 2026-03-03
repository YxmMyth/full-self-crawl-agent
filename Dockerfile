# =============================================================================
# Full Self-Crawl Agent — 统一 Dockerfile (多阶段构建)
#
# 用法:
#   开发: docker compose up dev          (挂载源码, 热更新)
#   生产: docker compose up agent        (代码打包进镜像)
#   测试: docker compose run dev pytest  (容器内跑测试)
#
# 基础镜像说明:
#   优先使用 python:3.12-bookworm (需联网拉取)
#   离线环境可改为本地 crawl-agent:latest
# =============================================================================

# ---- Stage 1: base (依赖层, 缓存友好) ----
FROM python:3.12-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DOCKER_CONTAINER=1 \
    DEBIAN_FRONTEND=noninteractive

# 系统工具 + Playwright Chromium 所需的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    jq \
    git \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxcb1 \
    libxkbcommon0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libatspi2.0-0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright + Chromium
RUN pip install playwright && playwright install chromium

# 工作空间 (agent 脚本读写用)
RUN mkdir -p /workspace/tmp /workspace/scripts

# ---- Stage 2: dev (开发模式, 源码通过 volume 挂载) ----
FROM base AS dev

RUN pip install --no-cache-dir pytest pytest-asyncio pytest-cov black ruff

RUN useradd -m -s /bin/bash crawler \
    && chown -R crawler:crawler /app /workspace

ENV BROWSER_HEADLESS=true

CMD ["bash"]

# ---- Stage 3: production (生产模式, 代码打包进镜像) ----
FROM base AS production

COPY src/ /app/src/
COPY specs/ /app/specs/
COPY config/ /app/config/

RUN useradd -m -s /bin/bash crawler \
    && chown -R crawler:crawler /app /workspace /home/crawler

RUN cp -r /root/.cache /home/crawler/.cache 2>/dev/null || true \
    && chown -R crawler:crawler /home/crawler/.cache 2>/dev/null || true

USER crawler

ENV BROWSER_HEADLESS=true

ENTRYPOINT ["python", "-m", "src.main"]