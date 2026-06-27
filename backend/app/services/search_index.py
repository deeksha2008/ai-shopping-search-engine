import json
from pathlib import Path

from backend.app.config import settings
from backend.app.core.personalization.personalization_engine import PersonalizationEngine
from backend.app.core.query_understanding.pipeline import QueryUnderstandingPipeline
from backend.app.core.ranking.ltr_ranker import LTRRanker
from backend.app.core.retrieval.bm25_retriever import BM25Retriever
from backend.app.core.retrieval.rrf_fusion import reciprocal_rank_fusion
from backend.app.core.retrieval.semantic_retriever import SemanticRetriever
from backend.app.core.zero_result.recovery import ZeroResultRecovery
from backend.app.models.database import Product, UserPreference
from backend.app.models.schemas import UserProfile


class SearchIndex:
    _instance: "SearchIndex | None" = None

    def __init__(self):
        self.products: dict[str, dict] = {}
        self.user_profiles: dict[str, UserProfile] = {}
        self.bm25 = BM25Retriever()
        self.semantic = SemanticRetriever(settings.models_dir)
        self.ltr = LTRRanker(settings.models_dir)
        self.query_pipeline = QueryUnderstandingPipeline(settings.data_dir)
        self.personalization = PersonalizationEngine()
        self.zero_result = ZeroResultRecovery(self.query_pipeline, self.bm25, self.semantic)
        self.loaded = False

    @classmethod
    def get_instance(cls) -> "SearchIndex":
        if cls._instance is None:
            cls._instance = SearchIndex()
        return cls._instance

    def load_from_db(self, db_session) -> None:
        products = db_session.query(Product).all()
        if not products:
            self._load_from_json()
            return

        self.products = {}
        product_ids = []
        documents = []

        for p in products:
            product_dict = {
                "product_id": p.id,
                "title": p.title,
                "description": p.description,
                "brand": p.brand,
                "category": p.category,
                "price": p.price,
                "rating": p.rating,
                "discount_pct": p.discount_pct,
                "in_stock": p.in_stock,
                "color": p.color,
                "click_count": p.click_count,
                "add_to_cart_count": p.add_to_cart_count,
                "purchase_count": p.purchase_count,
            }
            self.products[p.id] = product_dict
            product_ids.append(p.id)
            documents.append(p.search_text)

        self.bm25.build_index(product_ids, documents)

        index_path = Path(settings.data_dir) / "indices"
        self.semantic.load(index_path)
        if self.semantic.index is None:
            self.semantic.build_index(product_ids, documents)
            self.semantic.save(index_path)

        self.ltr.load()
        self._load_user_profiles_json()
        self.loaded = True

    def _load_user_profiles_json(self) -> None:
        profiles_path = Path(settings.data_dir) / "generated" / "user_profiles.json"
        if not profiles_path.exists():
            return
        with open(profiles_path) as f:
            profiles = json.load(f)
        for profile in profiles:
            self.user_profiles[profile["user_id"]] = UserProfile(
                user_id=profile["user_id"],
                preferred_brands=profile["preferred_brands"],
                preferred_categories=profile["preferred_categories"],
                budget_min=profile.get("budget_min"),
                budget_max=profile.get("budget_max"),
                premium_preference=profile.get("premium_preference", False),
            )

    def _load_from_json(self) -> None:
        products_path = Path(settings.data_dir) / "generated" / "products.json"
        if not products_path.exists():
            return

        with open(products_path) as f:
            raw_products = json.load(f)

        product_ids = []
        documents = []
        for p in raw_products:
            self.products[p["id"]] = p
            product_ids.append(p["id"])
            documents.append(p.get("search_text", f"{p['title']} {p['brand']} {p['category']}"))

        self.bm25.build_index(product_ids, documents)

        index_path = Path(settings.data_dir) / "indices"
        self.semantic.load(index_path)
        if self.semantic.index is None:
            self.semantic.build_index(product_ids, documents)
            self.semantic.save(index_path)

        self.ltr.load()
        self._load_user_profiles_json()
        self.loaded = True

    def get_user_profile(self, db_session, user_id: str | None) -> UserProfile | None:
        if not user_id:
            return None

        if user_id in self.user_profiles:
            return self.user_profiles[user_id]

        try:
            pref = db_session.query(UserPreference).filter(UserPreference.user_id == user_id).first()
            if not pref:
                return None

            return UserProfile(
                user_id=pref.user_id,
                preferred_brands=json.loads(pref.preferred_brands),
                preferred_categories=json.loads(pref.preferred_categories),
                budget_min=pref.budget_min,
                budget_max=pref.budget_max,
                premium_preference=pref.premium_preference,
            )
        except Exception:
            return self.user_profiles.get(user_id)
