from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self):
        self.product_ids: list[str] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

    def build_index(self, product_ids: list[str], documents: list[str]) -> None:
        self.product_ids = product_ids
        self.tokenized_corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        if not self.bm25 or not self.product_ids:
            return []

        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(
            zip(self.product_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(pid, float(score)) for pid, score in ranked[:top_k] if score > 0]
