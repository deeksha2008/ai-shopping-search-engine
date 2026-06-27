"""Ingest generated JSON data into PostgreSQL and build search indices."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.config import settings
from backend.app.models.database import Product, UserPreference, engine, init_db, SessionLocal
from backend.app.core.retrieval.bm25_retriever import BM25Retriever
from backend.app.core.retrieval.semantic_retriever import SemanticRetriever


def ingest():
    data_dir = Path(settings.data_dir) / "generated"
    if not (data_dir / "products.json").exists():
        print("No data found. Running generate_data.py first...")
        from scripts.generate_data import main as generate
        generate()

    init_db()
    db = SessionLocal()

    with open(data_dir / "products.json") as f:
        products = json.load(f)

    db.query(Product).delete()
    for p in products:
        db.add(Product(
            id=p["id"],
            title=p["title"],
            description=p["description"],
            brand=p["brand"],
            category=p["category"],
            price=p["price"],
            rating=p.get("rating", 0),
            discount_pct=p.get("discount_pct", 0),
            in_stock=p.get("in_stock", True),
            color=p.get("color"),
            click_count=p.get("click_count", 0),
            add_to_cart_count=p.get("add_to_cart_count", 0),
            purchase_count=p.get("purchase_count", 0),
            search_text=p.get("search_text", p["title"]),
        ))

    profiles_path = data_dir / "user_profiles.json"
    if profiles_path.exists():
        db.query(UserPreference).delete()
        with open(profiles_path) as f:
            profiles = json.load(f)
        for profile in profiles:
            db.add(UserPreference(
                user_id=profile["user_id"],
                preferred_brands=json.dumps(profile["preferred_brands"]),
                preferred_categories=json.dumps(profile["preferred_categories"]),
                budget_min=profile.get("budget_min"),
                budget_max=profile.get("budget_max"),
                premium_preference=profile.get("premium_preference", False),
            ))

    db.commit()

    product_ids = [p["id"] for p in products]
    documents = [p.get("search_text", p["title"]) for p in products]

    index_path = Path(settings.data_dir) / "indices"
    index_path.mkdir(parents=True, exist_ok=True)

    print("Building FAISS semantic index...")
    semantic = SemanticRetriever(settings.models_dir)
    semantic.build_index(product_ids, documents)
    semantic.save(index_path)

    print(f"Ingested {len(products)} products into PostgreSQL")
    db.close()


if __name__ == "__main__":
    ingest()
