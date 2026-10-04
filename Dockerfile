# syntax=docker/dockerfile:1.7
# ---------- build stage: compile wheels ----------
FROM python:3.12-slim AS build
WORKDIR /src
ARG INSTALL_ANALYTICS=true
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip wheel \
 && if [ "$INSTALL_ANALYTICS" = "true" ]; then EXTRA="[analytics]"; else EXTRA=""; fi \
 && pip wheel --no-cache-dir --wheel-dir /wheels ".${EXTRA}"

# ---------- runtime stage ----------
FROM python:3.12-slim
LABEL org.opencontainers.image.title="AI-EOS" \
      org.opencontainers.image.description="AI Executive Operating System"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EOS_CONFIG=/app/config/settings.yaml \
    EOS_AGENTS_CONFIG=/app/config/agents.yaml \
    EOS__TOOLS__SANDBOX_ROOT=/app/data/workspace \
    EOS__VECTOR__QDRANT_PATH=/app/data/qdrant
RUN groupadd --gid 10001 eos && useradd --uid 10001 --gid eos --create-home --shell /usr/sbin/nologin eos
WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY config ./config
RUN mkdir -p /app/data/workspace && chown -R eos:eos /app/data
USER eos
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=4).status==200 else 1)"
CMD ["uvicorn", "ai_eos.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
