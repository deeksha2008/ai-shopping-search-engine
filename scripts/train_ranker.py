"""Train LightGBM Learning-to-Rank model on synthetic interaction data."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.config import settings
from backend.app.core.ranking.feature_builder import FeatureBuilder
from backend.app.core.query_understanding.pipeline import QueryUnderstandingPipeline

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except OSError:
    LIGHTGBM_AVAILABLE = False


def create_training_data():
    data_dir = Path(settings.data_dir) / "generated"
    with open(data_dir / "products.json") as f:
        products_list = json.load(f)
    products = {p["id"]: p for p in products_list}

    with open(data_dir / "user_interactions.json") as f:
        interactions = json.load(f)

    pipeline = QueryUnderstandingPipeline(settings.data_dir)
    feature_builder = FeatureBuilder()

    queries = [
        "red nike shoes under 3000",
        "samsung mobile under 20000",
        "apple laptop",
        "sony headphones",
        "sports shoes",
        "smart tv",
        "nik shoes",
        "puma running shoes",
        "boat earbuds",
        "premium electronics",
        "black shoes nike",
        "lg tv under 50000",
    ]

    features = []
    labels = []
    groups = []

    for query in queries:
        parsed = pipeline.parse(query)
        group_features = []
        group_labels = []

        for product in products_list[:200]:
            relevance = 0
            if parsed.brand and product["brand"].lower() == parsed.brand.lower():
                relevance += 2
            if parsed.category and product["category"].lower() == parsed.category.lower():
                relevance += 2
            if parsed.budget and product["price"] <= parsed.budget:
                relevance += 1
            if parsed.color and product.get("color", "").lower() == parsed.color.lower():
                relevance += 1
            if query.lower() in product.get("search_text", ""):
                relevance += 1

            rrf_score = random_uniform_score(product)
            feat = feature_builder.build_features(parsed, product, rrf_score, rrf_score * 0.8)
            group_features.append(feat)
            group_labels.append(relevance)

        for interaction in interactions:
            if interaction.get("query") == query:
                pid = interaction["product_id"]
                if pid in products:
                    idx = next((i for i, p in enumerate(products_list[:200]) if p["id"] == pid), None)
                    if idx is not None:
                        itype = interaction["interaction_type"]
                        boost = {"click": 1, "add_to_cart": 2, "purchase": 3}.get(itype, 0)
                        group_labels[idx] = min(group_labels[idx] + boost, 5)

        if group_features:
            features.extend(group_features)
            labels.extend(group_labels)
            groups.append(len(group_features))

    for interaction in interactions[:5000]:
        pid = interaction["product_id"]
        if pid not in products:
            continue
        product = products[pid]
        query = interaction.get("query") or f"{product['brand']} {product['category']}"
        parsed = pipeline.parse(query)
        rrf_score = random_uniform_score(product)
        feat = feature_builder.build_features(parsed, product, rrf_score, rrf_score * 0.7)
        relevance = {"click": 1, "add_to_cart": 2, "purchase": 3, "search": 0}.get(
            interaction["interaction_type"], 0,
        )
        features.append(feat)
        labels.append(relevance)
        groups.append(1)

    return np.array(features), np.array(labels), groups


def random_uniform_score(product: dict) -> float:
    import random
    base = random.uniform(0.01, 0.5)
    base += product.get("click_count", 0) / 10000
    return min(base, 1.0)


def train():
    models_dir = Path(settings.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(settings.data_dir) / "generated"
    if not (data_dir / "products.json").exists():
        print("No data. Run generate_data.py first.")
        return

    if not LIGHTGBM_AVAILABLE:
        metadata = {
            "model_type": "RRF fallback (LightGBM unavailable — install libomp or use Docker)",
            "features": FeatureBuilder.FEATURE_NAMES,
        }
        with open(models_dir / "ltr_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print("LightGBM unavailable locally. Search will use RRF scores until trained in Docker.")
        return

    print("Creating training data...")
    X, y, groups = create_training_data()

    if len(X) == 0:
        print("No training data available.")
        return

    train_data = lgb.Dataset(X, label=y, group=groups)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 10,
        "verbose": -1,
    }

    print(f"Training LightGBM ranker on {len(X)} samples...")
    model = lgb.train(params, train_data, num_boost_round=100)

    model_path = models_dir / "ltr_ranker.txt"
    model.save_model(str(model_path))

    metadata = {
        "model_type": "LightGBM LambdaRank",
        "num_samples": len(X),
        "num_groups": len(groups),
        "features": FeatureBuilder.FEATURE_NAMES,
    }
    with open(models_dir / "ltr_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()
