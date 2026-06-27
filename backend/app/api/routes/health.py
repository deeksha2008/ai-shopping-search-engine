from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.models.database import get_db
from backend.app.models.schemas import HealthResponse
from backend.app.services.cache_service import CacheService
from backend.app.services.search_index import SearchIndex

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    postgres_ok = False
    try:
        db.execute(text("SELECT 1"))
        postgres_ok = True
    except Exception:
        pass

    cache = CacheService()
    redis_ok = cache.ping()
    index = SearchIndex.get_instance()

    status = "healthy" if postgres_ok and index.loaded else "degraded"
    return HealthResponse(
        status=status,
        timestamp=datetime.utcnow(),
        postgres=postgres_ok,
        redis=redis_ok,
        indices_loaded=index.loaded,
    )
