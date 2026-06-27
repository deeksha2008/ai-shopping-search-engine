from backend.app.models.schemas import ParsedQuery, ProductResult, UserProfile


class PersonalizationEngine:
    RELEVANCE_WEIGHT = 0.6
    PREFERENCE_WEIGHT = 0.25
    BUSINESS_WEIGHT = 0.15

    def compute_preference_score(self, product: dict, profile: UserProfile | None) -> float:
        if profile is None:
            return 0.5

        score = 0.0
        weights = 0.0

        if profile.preferred_brands:
            weights += 0.4
            if product["brand"].lower() in [b.lower() for b in profile.preferred_brands]:
                score += 0.4

        if profile.preferred_categories:
            weights += 0.35
            if product["category"].lower() in [c.lower() for c in profile.preferred_categories]:
                score += 0.35

        if profile.budget_min is not None and profile.budget_max is not None:
            weights += 0.25
            if profile.budget_min <= product["price"] <= profile.budget_max:
                score += 0.25

        if profile.premium_preference:
            weights += 0.1
            if product["price"] > 10000 or product.get("rating", 0) >= 4.5:
                score += 0.1

        return score / weights if weights > 0 else 0.5

    def compute_business_score(self, product: dict) -> float:
        rating_score = product.get("rating", 0.0) / 5.0
        discount_score = min(product.get("discount_pct", 0.0) / 50.0, 1.0)
        availability_score = 1.0 if product.get("in_stock", True) else 0.0
        popularity = min(
            (product.get("click_count", 0) + product.get("purchase_count", 0) * 3) / 1000.0,
            1.0,
        )
        return 0.35 * rating_score + 0.25 * discount_score + 0.2 * availability_score + 0.2 * popularity

    def rerank(
        self,
        parsed_query: ParsedQuery,
        ranked: list[tuple[str, float]],
        products: dict[str, dict],
        profile: UserProfile | None,
    ) -> list[ProductResult]:
        if not ranked:
            return []

        max_relevance = max(score for _, score in ranked) or 1.0
        results: list[ProductResult] = []

        for product_id, relevance_score in ranked:
            product = products[product_id]
            norm_relevance = relevance_score / max_relevance
            preference_score = self.compute_preference_score(product, profile)
            business_score = self.compute_business_score(product)

            final_score = (
                self.RELEVANCE_WEIGHT * norm_relevance
                + self.PREFERENCE_WEIGHT * preference_score
                + self.BUSINESS_WEIGHT * business_score
            )

            results.append(
                ProductResult(
                    product_id=product_id,
                    title=product["title"],
                    brand=product["brand"],
                    category=product["category"],
                    price=product["price"],
                    rating=product.get("rating", 0.0),
                    discount_pct=product.get("discount_pct", 0.0),
                    in_stock=product.get("in_stock", True),
                    color=product.get("color"),
                    relevance_score=round(norm_relevance, 4),
                    personalization_score=round(preference_score, 4),
                    business_score=round(business_score, 4),
                    final_score=round(final_score, 4),
                )
            )

        results.sort(key=lambda r: r.final_score, reverse=True)
        for i, result in enumerate(results, start=1):
            result.rank = i
        return results
