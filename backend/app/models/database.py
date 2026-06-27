from collections.abc import Generator
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.app.config import settings


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    brand = Column(String, nullable=False, index=True)
    category = Column(String, nullable=False, index=True)
    price = Column(Float, nullable=False, index=True)
    rating = Column(Float, default=0.0)
    discount_pct = Column(Float, default=0.0)
    in_stock = Column(Boolean, default=True)
    color = Column(String, nullable=True)
    click_count = Column(Integer, default=0)
    add_to_cart_count = Column(Integer, default=0)
    purchase_count = Column(Integer, default=0)
    search_text = Column(Text, nullable=False)


class SearchEvent(Base):
    __tablename__ = "search_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(String, nullable=False)
    user_id = Column(String, nullable=True)
    results_count = Column(Integer, default=0)
    latency_ms = Column(Float, default=0.0)
    cache_hit = Column(Boolean, default=False)
    zero_result_recovery = Column(Boolean, default=False)
    clicked_product_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserPreference(Base):
    __tablename__ = "user_preferences"

    user_id = Column(String, primary_key=True)
    preferred_brands = Column(Text, default="[]")
    preferred_categories = Column(Text, default="[]")
    budget_min = Column(Float, nullable=True)
    budget_max = Column(Float, nullable=True)
    premium_preference = Column(Boolean, default=False)


engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


class _OfflineQuery:
    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return None

    def all(self):
        return []


class _OfflineSession:
    """No-op session when PostgreSQL is unavailable."""

    def query(self, *args, **kwargs):
        return _OfflineQuery()

    def add(self, *args, **kwargs):
        pass

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass

    def execute(self, *args, **kwargs):
        return None


def get_db() -> Generator[Any, None, None]:
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        yield db
    except Exception:
        offline = _OfflineSession()
        try:
            yield offline
        finally:
            offline.close()
    finally:
        db.close()
