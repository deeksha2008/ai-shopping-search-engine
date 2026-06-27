from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from backend.app.dependencies import limiter
from backend.app.models.database import get_db
from backend.app.models.schemas import SearchRequest, SearchResponse
from backend.app.services.search_service import SearchService

router = APIRouter()
search_service = SearchService()


@router.post("/search", response_model=SearchResponse)
@limiter.limit("60/minute")
async def search_products(request: Request, body: SearchRequest, db: Session = Depends(get_db)):
    return search_service.search(body, db)
