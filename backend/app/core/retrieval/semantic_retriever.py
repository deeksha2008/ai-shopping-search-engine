import json
from pathlib import Path

import faiss
import numpy as np


class SemanticRetriever:
    def __init__(self, models_dir: str, model_name: str = "all-MiniLM-L6-v2"):
        self.models_dir = Path(models_dir)
        self.model_name = model_name
        self.model = None
        self.tfidf_vectorizer = None
        self.use_tfidf = False
        self.index: faiss.IndexFlatIP | None = None
        self.product_ids: list[str] = []
        self.dimension = 384

    def _get_model(self):
        if self.model is not None:
            return self.model
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.use_tfidf = False
            self.dimension = self.model.get_sentence_embedding_dimension()
            return self.model
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(max_features=384, stop_words="english")
            self.use_tfidf = True
            return None

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.use_tfidf:
            if self.tfidf_vectorizer is None:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.tfidf_vectorizer = TfidfVectorizer(max_features=384, stop_words="english")
            if not hasattr(self.tfidf_vectorizer, "vocabulary_"):
                matrix = self.tfidf_vectorizer.fit_transform(texts)
            else:
                matrix = self.tfidf_vectorizer.transform(texts)
            embeddings = matrix.toarray().astype(np.float32)
        else:
            model = self._get_model()
            embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            embeddings = np.array(embeddings, dtype=np.float32)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms

    def build_index(self, product_ids: list[str], documents: list[str]) -> None:
        self._get_model()
        embeddings = self._encode(documents)
        self.dimension = embeddings.shape[1]

        self.product_ids = product_ids
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss.index"))
        with open(path / "product_ids.json", "w") as f:
            json.dump(self.product_ids, f)
        meta = {"use_tfidf": self.use_tfidf, "dimension": self.dimension}
        with open(path / "semantic_meta.json", "w") as f:
            json.dump(meta, f)
        if self.use_tfidf and self.tfidf_vectorizer is not None:
            import joblib
            joblib.dump(self.tfidf_vectorizer, path / "tfidf_vectorizer.pkl")

    def load(self, path: Path) -> None:
        index_path = path / "faiss.index"
        ids_path = path / "product_ids.json"
        if not index_path.exists() or not ids_path.exists():
            return

        meta_path = path / "semantic_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.use_tfidf = meta.get("use_tfidf", False)
            self.dimension = meta.get("dimension", 384)

        self.index = faiss.read_index(str(index_path))
        with open(ids_path) as f:
            self.product_ids = json.load(f)

        if self.use_tfidf:
            tfidf_path = path / "tfidf_vectorizer.pkl"
            if tfidf_path.exists():
                import joblib
                self.tfidf_vectorizer = joblib.load(tfidf_path)
        else:
            self._get_model()

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        if self.index is None or not self.product_ids:
            return []

        query_embedding = self._encode([query])
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.product_ids)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.product_ids[idx], float(score)))
        return results
