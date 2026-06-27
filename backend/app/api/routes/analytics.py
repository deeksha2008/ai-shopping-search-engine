from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.models.database import get_db
from backend.app.models.schemas import AnalyticsMetrics
from backend.app.services.analytics_service import AnalyticsService
from backend.app.services.cache_service import CacheService
from backend.app.services.search_service import SearchService

router = APIRouter()
analytics_service = AnalyticsService()
search_service = SearchService()


@router.get("/analytics/metrics", response_model=AnalyticsMetrics)
def get_metrics(db: Session = Depends(get_db)):
    return analytics_service.get_metrics(db, cache_hit_ratio=search_service.cache.hit_ratio)
