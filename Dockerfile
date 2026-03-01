FROM python:3.11-slim AS base

RUN pip install playwright && \
    playwright install chromium && \
    playwright install-deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS app

COPY src/ /app/src/
COPY specs/ /app/specs/
COPY config/ /app/config/
WORKDIR /app

ENTRYPOINT ["python", "-m", "src.main", "--container"]