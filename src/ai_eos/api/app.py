"""FastAPI application factory."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from importlib import resources

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from ai_eos import __version__
from ai_eos.api.routes import router
from ai_eos.config import Settings, get_settings
from ai_eos.container import Container, build_container
from ai_eos.logging_setup import configure_logging, request_id

log = logging.getLogger(__name__)
HTTP_REQUESTS = Counter("eos_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram("eos_http_request_seconds", "HTTP latency", ["route"])
_UNLIMITED = ("/health", "/metrics", "/api/events", "/static")


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or (container.settings if container else get_settings())
    configure_logging(settings.app.log_level, settings.app.log_json)
    c = container or build_container(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN202
        await c.startup()
        log.info(
            "AI-EOS %s started (%s, default model provider: %s)",
            __version__,
            settings.app.environment,
            settings.llm.default_provider,
        )
        yield
        await c.shutdown()

    app = FastAPI(
        title="AI-EOS API",
        version=__version__,
        lifespan=lifespan,
        description="AI Executive Operating System - talk to your Chief of Staff.",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.container = c
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def observability(request: Request, call_next):  # noqa: ANN001, ANN202
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request_id.set(rid)
        path = request.url.path
        if path.startswith("/api") and not path.startswith(_UNLIMITED):
            auth = request.headers.get("authorization", "") or request.headers.get("x-api-key", "")
            identity = auth[-24:] if auth else (request.client.host if request.client else "anon")
            allowed, remaining = await c.rate_limiter.allow(identity)
            if not allowed:
                return JSONResponse(
                    {"detail": "rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": "60", "X-Request-ID": rid},
                )
        start = time.perf_counter()
        response = await call_next(request)
        route = getattr(request.scope.get("route"), "path", "unmatched")
        HTTP_REQUESTS.labels(request.method, route, response.status_code).inc()
        HTTP_LATENCY.labels(route).observe(time.perf_counter() - start)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    app.include_router(router)

    @app.get("/health/live", tags=["health"])
    async def live() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/health/ready", tags=["health"])
    async def ready() -> JSONResponse:
        checks = {"database": await c.db.ping(), "cache": await c.kv.ping(), "vector": c.vectors.ping()}
        ok = all(checks.values())
        return JSONResponse({"status": "ok" if ok else "degraded", "checks": checks}, status_code=200 if ok else 503)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        html = resources.files("ai_eos.web").joinpath("index.html").read_text(encoding="utf-8")
        return HTMLResponse(
            html,
            headers={
                "Content-Security-Policy": "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'"
            },
        )

    return app


def main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run("ai_eos.api.app:create_app", factory=True, host="0.0.0.0", port=8000, proxy_headers=True)  # noqa: S104
