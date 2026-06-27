from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ParsedQuery(BaseModel):
    raw_query: str
    normalized_query: str
    corrected_query: str
    expanded_query: str
    brand: str | None = None
    category: str | None = None
    color: str | None = None
    budget: float | None = None
    entities: dict[str, Any] = Field(default_factory=dict)


class ProductResult(BaseModel):
    product_id: str
    title: str
    brand: str
    category: str
    price: float
    rating: float
    discount_pct: float
    in_stock: bool
    color: str | None = None
    relevance_score: float = 0.0
    personalization_score: float = 0.0
    business_score: float = 0.0
    final_score: float = 0.0
    rank: int = 0


class SearchRequest(BaseModel):
    query: str
    user_id: str | None = None
    limit: int = 10


class SearchResponse(BaseModel):
    query: str
    parsed_query: ParsedQuery
    results: list[ProductResult]
    total_results: int
    latency_ms: float
    cache_hit: bool = False
    zero_result_recovery: bool = False
    recovery_suggestions: list[str] = Field(default_factory=list)


class UserProfile(BaseModel):
    user_id: str
    preferred_brands: list[str] = Field(default_factory=list)
    preferred_categories: list[str] = Field(default_factory=list)
    budget_min: float | None = None
    budget_max: float | None = None
    premium_preference: bool = False


class AnalyticsMetrics(BaseModel):
    search_latency_p50_ms: float
    search_latency_p95_ms: float
    ctr: float
    ndcg_at_10: float
    mrr: float
    zero_result_rate: float
    cache_hit_ratio: float
    total_searches: int


class SearchEventCreate(BaseModel):
    query: str
    user_id: str | None = None
    results_count: int
    latency_ms: float
    cache_hit: bool
    zero_result_recovery: bool
    clicked_product_id: str | None = None
    relevance_scores: list[float] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    postgres: bool
    redis: bool
    indices_loaded: bool
