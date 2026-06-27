import json
import re
from pathlib import Path

from backend.app.models.schemas import ParsedQuery


class QueryUnderstandingPipeline:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.brands = self._load_json_list("brands.json", "name")
        self.categories = self._load_json_list("categories.json", "name")
        self.colors = [
            "red", "blue", "green", "black", "white", "yellow",
            "pink", "grey", "gray", "brown", "orange", "purple",
        ]
        self.synonyms = {
            "phone": ["mobile", "smartphone", "cellphone"],
            "laptop": ["notebook", "computer"],
            "shoes": ["footwear", "sneakers", "trainers"],
            "tv": ["television", "smart tv"],
            "headphones": ["earphones", "earbuds"],
            "watch": ["smartwatch", "wristwatch"],
        }
        self._typo_corrector = None

    def _load_json_list(self, filename: str, key: str) -> list[str]:
        path = self.data_dir / "generated" / filename
        if not path.exists():
            return []
        with open(path) as f:
            data = json.load(f)
        return [item[key].lower() for item in data]

    @property
    def typo_corrector(self):
        if self._typo_corrector is None:
            from backend.app.core.query_understanding.typo_corrector import TypoCorrector

            vocabulary = set(self.brands + self.categories + self.colors)
            for syns in self.synonyms.values():
                vocabulary.update(syns)
            self._typo_corrector = TypoCorrector(list(vocabulary))
        return self._typo_corrector

    def normalize(self, query: str) -> str:
        normalized = query.lower().strip()
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def extract_entities(self, query: str) -> dict:
        entities: dict = {}
        tokens = query.split()

        for brand in sorted(self.brands, key=len, reverse=True):
            if brand in query:
                entities["brand"] = brand.title()
                break

        for category in sorted(self.categories, key=len, reverse=True):
            if category in query:
                entities["category"] = category.title()
                break

        for color in self.colors:
            if color in query:
                entities["color"] = color.title()
                break

        budget_match = re.search(
            r"(?:under|below|less than|max|upto|up to)\s*(\d+)",
            query,
        )
        if budget_match:
            entities["budget"] = float(budget_match.group(1))

        return entities

    def expand_synonyms(self, query: str) -> str:
        tokens = query.split()
        expanded_tokens = list(tokens)
        for token in tokens:
            for canonical, syns in self.synonyms.items():
                if token == canonical or token in syns:
                    expanded_tokens.extend([canonical] + syns)
        return " ".join(dict.fromkeys(expanded_tokens))

    def parse(self, raw_query: str) -> ParsedQuery:
        normalized = self.normalize(raw_query)
        corrected = self.typo_corrector.correct_query(normalized)
        entities = self.extract_entities(corrected)
        expanded = self.expand_synonyms(corrected)

        return ParsedQuery(
            raw_query=raw_query,
            normalized_query=normalized,
            corrected_query=corrected,
            expanded_query=expanded,
            brand=entities.get("brand"),
            category=entities.get("category"),
            color=entities.get("color"),
            budget=entities.get("budget"),
            entities=entities,
        )
