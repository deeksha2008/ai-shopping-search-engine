import json
from pathlib import Path

import numpy as np

from backend.app.core.ranking.feature_builder import FeatureBuilder
from backend.app.models.schemas import ParsedQuery


class LTRRanker:
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.model = None
        self.feature_builder = FeatureBuilder()

    def load(self) -> None:
        model_path = self.models_dir / "ltr_ranker.txt"
        if not model_path.exists():
            return
        try:
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=str(model_path))
        except OSError:
            self.model = None

    def rank(
        self,
        parsed_query: ParsedQuery,
        candidates: list[tuple[str, float]],
        products: dict[str, dict],
        semantic_scores: dict[str, float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []

        if self.model is None:
            return candidates[:top_k]

        features = []
        product_ids = []
        for product_id, rrf_score in candidates:
            product = products.get(product_id)
            if not product:
                continue
            feature_vec = self.feature_builder.build_features(
                parsed_query,
                product,
                rrf_score,
                semantic_scores.get(product_id, 0.0),
            )
            features.append(feature_vec)
            product_ids.append(product_id)

        if not features:
            return []

        scores = self.model.predict(np.array(features))
        ranked = sorted(zip(product_ids, scores), key=lambda x: x[1], reverse=True)
        return [(pid, float(score)) for pid, score in ranked[:top_k]]

    def save_training_metadata(self, metadata: dict) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        with open(self.models_dir / "ltr_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
