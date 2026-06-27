import json
from typing import Any

import redis

from backend.app.config import settings


class CacheService:
    def __init__(self):
        self._client: redis.Redis | None = None
        self.hits = 0
        self.misses = 0

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(settings.redis_url, decode_responses=True)
        return self._client

    def _cache_key(self, query: str, user_id: str | None) -> str:
        user_part = user_id or "anonymous"
        return f"search:{user_part}:{query.lower().strip()}"

    def get(self, query: str, user_id: str | None) -> dict[str, Any] | None:
        try:
            cached = self.client.get(self._cache_key(query, user_id))
            if cached:
                self.hits += 1
                return json.loads(cached)
            self.misses += 1
            return None
        except redis.RedisError:
            self.misses += 1
            return None

    def set(self, query: str, user_id: str | None, data: dict[str, Any]) -> None:
        try:
            self.client.setex(
                self._cache_key(query, user_id),
                settings.cache_ttl_seconds,
                json.dumps(data),
            )
        except redis.RedisError:
            pass

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def ping(self) -> bool:
        try:
            return self.client.ping()
        except redis.RedisError:
            return False
