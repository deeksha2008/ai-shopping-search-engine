import json
import math
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.models.database import SearchEvent
from backend.app.models.schemas import AnalyticsMetrics


class AnalyticsService:
    def compute_ndcg_at_k(self, relevance_scores: list[float], k: int = 10) -> float:
        if not relevance_scores:
            return 0.0

        scores = relevance_scores[:k]
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(scores))
        ideal = sorted(scores, reverse=True)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
        return dcg / idcg if idcg > 0 else 0.0

    def get_metrics(self, db: Session, cache_hit_ratio: float = 0.0) -> AnalyticsMetrics:
        try:
            events = db.query(SearchEvent).all()
        except Exception:
            events = []
        if not events:
            return AnalyticsMetrics(
                search_latency_p50_ms=0.0,
                search_latency_p95_ms=0.0,
                ctr=0.0,
                ndcg_at_10=0.0,
                mrr=0.0,
                zero_result_rate=0.0,
                cache_hit_ratio=cache_hit_ratio,
                total_searches=0,
            )

        latencies = sorted(e.latency_ms for e in events)
        p50_idx = int(len(latencies) * 0.5)
        p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)

        clicked = sum(1 for e in events if e.clicked_product_id)
        zero_results = sum(1 for e in events if e.results_count == 0)

        ndcg_scores = []
        mrr_scores = []
        for event in events:
            if event.clicked_product_id:
                mrr_scores.append(1.0)
            else:
                mrr_scores.append(0.0)

        total = len(events)
        return AnalyticsMetrics(
            search_latency_p50_ms=round(latencies[p50_idx], 2),
            search_latency_p95_ms=round(latencies[p95_idx], 2),
            ctr=round(clicked / total, 4),
            ndcg_at_10=round(sum(ndcg_scores) / max(len(ndcg_scores), 1), 4),
            mrr=round(sum(mrr_scores) / max(len(mrr_scores), 1), 4),
            zero_result_rate=round(zero_results / total, 4),
            cache_hit_ratio=round(cache_hit_ratio, 4),
            total_searches=total,
        )

    def log_search(
        self,
        db: Session,
        query: str,
        user_id: str | None,
        results_count: int,
        latency_ms: float,
        cache_hit: bool,
        zero_result_recovery: bool,
    ) -> None:
        try:
            event = SearchEvent(
                query=query,
                user_id=user_id,
                results_count=results_count,
                latency_ms=latency_ms,
                cache_hit=cache_hit,
                zero_result_recovery=zero_result_recovery,
            )
            db.add(event)
            db.commit()
        except Exception:
            db.rollback()

    def record_click(self, db: Session, event_id: int, product_id: str) -> None:
        event = db.query(SearchEvent).filter(SearchEvent.id == event_id).first()
        if event:
            event.clicked_product_id = product_id
            db.commit()
