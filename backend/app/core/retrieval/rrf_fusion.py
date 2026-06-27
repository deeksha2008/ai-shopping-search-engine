def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (product_id, _) in enumerate(ranked_list, start=1):
            scores[product_id] = scores.get(product_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused
