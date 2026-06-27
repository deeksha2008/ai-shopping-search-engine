from backend.app.core.query_understanding.pipeline import QueryUnderstandingPipeline
from backend.app.core.retrieval.rrf_fusion import reciprocal_rank_fusion
from backend.app.core.retrieval.semantic_retriever import SemanticRetriever
from backend.app.core.retrieval.bm25_retriever import BM25Retriever
from backend.app.models.schemas import ParsedQuery


class ZeroResultRecovery:
    def __init__(
        self,
        query_pipeline: QueryUnderstandingPipeline,
        bm25: BM25Retriever,
        semantic: SemanticRetriever,
    ):
        self.query_pipeline = query_pipeline
        self.bm25 = bm25
        self.semantic = semantic

    def recover(
        self,
        parsed_query: ParsedQuery,
        top_k: int = 10,
    ) -> tuple[list[tuple[str, float]], list[str], bool]:
        suggestions: list[str] = []

        if parsed_query.corrected_query != parsed_query.normalized_query:
            suggestions.append(f"Did you mean: {parsed_query.corrected_query}?")

        bm25_results = self.bm25.search(parsed_query.corrected_query, top_k=top_k)
        semantic_results = self.semantic.search(parsed_query.corrected_query, top_k=top_k)

        if not bm25_results and not semantic_results:
            semantic_results = self.semantic.search(parsed_query.expanded_query, top_k=top_k)
            if semantic_results:
                suggestions.append("Showing results for similar terms")

        if not semantic_results and parsed_query.entities:
            entity_query = " ".join(str(v).lower() for v in parsed_query.entities.values())
            semantic_results = self.semantic.search(entity_query, top_k=top_k)
            if semantic_results:
                suggestions.append(f"Showing products matching: {entity_query}")

        fused = reciprocal_rank_fusion([bm25_results, semantic_results])
        recovered = len(fused) > 0

        if recovered and not suggestions:
            suggestions.append("Showing similar products based on your query")

        return fused[:top_k], suggestions, recovered
