"""Quick smoke test for the search pipeline (no server required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.config import settings
from backend.app.core.personalization.personalization_engine import PersonalizationEngine
from backend.app.core.query_understanding.pipeline import QueryUnderstandingPipeline
from backend.app.core.ranking.ltr_ranker import LTRRanker
from backend.app.core.retrieval.bm25_retriever import BM25Retriever
from backend.app.core.retrieval.rrf_fusion import reciprocal_rank_fusion
from backend.app.core.retrieval.semantic_retriever import SemanticRetriever
from backend.app.models.schemas import UserProfile
import json


def main():
    data_dir = Path(settings.data_dir) / "generated"
    with open(data_dir / "products.json") as f:
        products_list = json.load(f)

    products = {p["id"]: p for p in products_list}
    product_ids = [p["id"] for p in products_list]
    documents = [p["search_text"] for p in products_list]

    pipeline = QueryUnderstandingPipeline(settings.data_dir)
    bm25 = BM25Retriever()
    bm25.build_index(product_ids, documents)

    semantic = SemanticRetriever(settings.models_dir)
    index_path = Path(settings.data_dir) / "indices"
    semantic.load(index_path)
    if semantic.index is None:
        print("Building FAISS index (first run, may take a minute)...")
        semantic.build_index(product_ids, documents)
        semantic.save(index_path)

    ltr = LTRRanker(settings.models_dir)
    ltr.load()
    personalization = PersonalizationEngine()

    queries = [
        ("red nike shoes under 3000", None),
        ("Nik shoes", None),
        ("sports shoes", "user_a"),
        ("sports shoes", "user_b"),
    ]

    profiles = {
        "user_a": UserProfile(
            user_id="user_a",
            preferred_brands=["Nike", "Adidas", "Puma"],
            preferred_categories=["Shoes", "Sports"],
            budget_min=2000,
            budget_max=5000,
        ),
        "user_b": UserProfile(
            user_id="user_b",
            preferred_brands=["Samsung", "Sony"],
            preferred_categories=["Electronics"],
            budget_min=15000,
            budget_max=80000,
            premium_preference=True,
        ),
    }

    for query, user_id in queries:
        parsed = pipeline.parse(query)
        bm25_results = bm25.search(parsed.expanded_query, top_k=50)
        semantic_results = semantic.search(parsed.expanded_query, top_k=50)
        fused = reciprocal_rank_fusion([bm25_results, semantic_results])

        if parsed.budget:
            fused = [
                (pid, s) for pid, s in fused
                if products[pid]["price"] <= parsed.budget
            ] or fused

        semantic_scores = dict(semantic_results)
        ranked = ltr.rank(parsed, fused, products, semantic_scores, top_k=10)
        profile = profiles.get(user_id) if user_id else None
        results = personalization.rerank(parsed, ranked, products, profile)

        print(f"\n{'='*60}")
        print(f"Query: {query!r} | User: {user_id or 'anonymous'}")
        print(f"Parsed: brand={parsed.brand}, category={parsed.category}, "
              f"color={parsed.color}, budget={parsed.budget}")
        print(f"Corrected: {parsed.corrected_query}")
        for r in results[:5]:
            print(f"  {r.rank}. {r.title[:50]} — ₹{r.price} (score={r.final_score})")

    print("\n✓ Pipeline smoke test passed")


if __name__ == "__main__":
    main()
