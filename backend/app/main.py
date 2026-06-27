import logging
import time
from datetime import datetime

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.responses import Response

from backend.app.api.routes import analytics, health, search
from backend.app.dependencies import limiter
from backend.app.config import settings
from backend.app.models.database import SessionLocal, init_db
from backend.app.services.search_index import SearchIndex

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
)
logger = structlog.get_logger()

SEARCH_REQUESTS = Counter("search_requests_total", "Total search requests")
SEARCH_LATENCY = Histogram("search_latency_seconds", "Search latency in seconds")

app = FastAPI(
    title="AI Product Search & Shopping Intelligence Engine",
    description="Production-grade hybrid search with LTR, personalization, and analytics",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
    return response


@app.on_event("startup")
def startup():
    index = SearchIndex.get_instance()
    try:
        init_db()
        db = SessionLocal()
        try:
            index.load_from_db(db)
        finally:
            db.close()
    except Exception as exc:
        logger.warning("db_unavailable_loading_from_json", error=str(exc))
        index._load_from_json()
    logger.info("search_index_loaded", product_count=len(index.products), loaded=index.loaded)


app.include_router(health.router, tags=["Health"])
app.include_router(search.router, prefix="/api/v1", tags=["Search"])
app.include_router(analytics.router, prefix="/api/v1", tags=["Analytics"])


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
