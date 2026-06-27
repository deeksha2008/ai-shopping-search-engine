import math

import numpy as np

from backend.app.models.schemas import ParsedQuery


class FeatureBuilder:
    def build_features(
        self,
        parsed_query: ParsedQuery,
        product: dict,
        rrf_score: float,
        semantic_score: float = 0.0,
    ) -> np.ndarray:
        brand_match = 1.0 if parsed_query.brand and product["brand"].lower() == parsed_query.brand.lower() else 0.0
        category_match = (
            1.0 if parsed_query.category and product["category"].lower() == parsed_query.category.lower() else 0.0
        )
        color_match = 1.0 if parsed_query.color and product.get("color", "").lower() == parsed_query.color.lower() else 0.0

        price = product["price"]
        if parsed_query.budget:
            price_distance = max(0.0, (price - parsed_query.budget) / max(parsed_query.budget, 1))
        else:
            price_distance = 0.0

        budget_match = 1.0 if parsed_query.budget and price <= parsed_query.budget else 0.0

        rating = product.get("rating", 0.0) / 5.0
        discount = product.get("discount_pct", 0.0) / 100.0
        availability = 1.0 if product.get("in_stock", True) else 0.0

        clicks = math.log1p(product.get("click_count", 0))
        add_to_cart = math.log1p(product.get("add_to_cart_count", 0))
        purchases = math.log1p(product.get("purchase_count", 0))

        return np.array([
            rrf_score,
            semantic_score,
            brand_match,
            category_match,
            color_match,
            budget_match,
            price_distance,
            rating,
            discount,
            availability,
            clicks,
            add_to_cart,
            purchases,
        ], dtype=np.float32)

    FEATURE_NAMES = [
        "rrf_score", "semantic_score", "brand_match", "category_match",
        "color_match", "budget_match", "price_distance", "rating",
        "discount", "availability", "clicks", "add_to_cart", "purchases",
    ]
