import time

from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.core.retrieval.rrf_fusion import reciprocal_rank_fusion
from backend.app.models.schemas import SearchRequest, SearchResponse
from backend.app.services.analytics_service import AnalyticsService
from backend.app.services.cache_service import CacheService
from backend.app.services.search_index import SearchIndex


class SearchService:
    def __init__(self):
        self.index = SearchIndex.get_instance()
        self.cache = CacheService()
        self.analytics = AnalyticsService()

    def search(self, request: SearchRequest, db: Session) -> SearchResponse:
        start = time.perf_counter()

        cached = self.cache.get(request.query, request.user_id)
        if cached:
            cached["cache_hit"] = True
            cached["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
            self.analytics.log_search(
                db, request.query, request.user_id,
                cached["total_results"], cached["latency_ms"], True, cached.get("zero_result_recovery", False),
            )
            return SearchResponse(**cached)

        parsed = self.index.query_pipeline.parse(request.query)
        search_query = parsed.expanded_query or parsed.corrected_query

        bm25_results = self.index.bm25.search(search_query, top_k=settings.bm25_top_k)
        semantic_results = self.index.semantic.search(search_query, top_k=settings.semantic_top_k)
        semantic_scores = dict(semantic_results)

        fused = reciprocal_rank_fusion([bm25_results, semantic_results], k=settings.rrf_k)

        zero_result_recovery = False
        recovery_suggestions: list[str] = []

        if not fused:
            fused, recovery_suggestions, zero_result_recovery = self.index.zero_result.recover(
                parsed, top_k=settings.bm25_top_k,
            )

        if parsed.budget:
            fused = [
                (pid, score) for pid, score in fused
                if self.index.products.get(pid, {}).get("price", float("inf")) <= parsed.budget
            ] or fused

        ltr_ranked = self.index.ltr.rank(
            parsed, fused, self.index.products, semantic_scores, top_k=settings.final_top_k,
        )

        profile = self.index.get_user_profile(db, request.user_id)
        results = self.index.personalization.rerank(parsed, ltr_ranked, self.index.products, profile)
        results = results[: request.limit]

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        response = SearchResponse(
            query=request.query,
            parsed_query=parsed,
            results=results,
            total_results=len(results),
            latency_ms=latency_ms,
            cache_hit=False,
            zero_result_recovery=zero_result_recovery,
            recovery_suggestions=recovery_suggestions,
        )

        self.cache.set(request.query, request.user_id, response.model_dump())
        self.analytics.log_search(
            db, request.query, request.user_id,
            len(results), latency_ms, False, zero_result_recovery,
        )

        return response
