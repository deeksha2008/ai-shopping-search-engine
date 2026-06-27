"""Generate synthetic e-commerce data for the search engine."""

import json
import random
import uuid
from pathlib import Path

BRANDS = [
    {"id": "b1", "name": "Nike", "category_focus": "Sports"},
    {"id": "b2", "name": "Samsung", "category_focus": "Electronics"},
    {"id": "b3", "name": "Apple", "category_focus": "Electronics"},
    {"id": "b4", "name": "Adidas", "category_focus": "Sports"},
    {"id": "b5", "name": "Sony", "category_focus": "Electronics"},
    {"id": "b6", "name": "Puma", "category_focus": "Sports"},
    {"id": "b7", "name": "LG", "category_focus": "Electronics"},
    {"id": "b8", "name": "Boat", "category_focus": "Electronics"},
    {"id": "b9", "name": "Levi's", "category_focus": "Fashion"},
    {"id": "b10", "name": "HP", "category_focus": "Electronics"},
]

CATEGORIES = [
    {"id": "c1", "name": "Shoes", "parent": "Sports"},
    {"id": "c2", "name": "Electronics", "parent": "Electronics"},
    {"id": "c3", "name": "Mobiles", "parent": "Electronics"},
    {"id": "c4", "name": "Laptops", "parent": "Electronics"},
    {"id": "c5", "name": "Headphones", "parent": "Electronics"},
    {"id": "c6", "name": "TV", "parent": "Electronics"},
    {"id": "c7", "name": "Clothing", "parent": "Fashion"},
    {"id": "c8", "name": "Watches", "parent": "Fashion"},
    {"id": "c9", "name": "Sports", "parent": "Sports"},
    {"id": "c10", "name": "Accessories", "parent": "Fashion"},
]

COLORS = ["Red", "Blue", "Green", "Black", "White", "Yellow", "Pink", "Grey", "Brown", "Orange"]
ADJECTIVES = ["Premium", "Pro", "Ultra", "Classic", "Lite", "Max", "Plus", "Elite", "Sport", "Smart"]

SYNTHETIC_USERS = [
    {
        "user_id": "user_a",
        "preferred_brands": ["Nike", "Adidas", "Puma"],
        "preferred_categories": ["Shoes", "Sports", "Clothing"],
        "budget_min": 2000,
        "budget_max": 5000,
        "premium_preference": False,
    },
    {
        "user_id": "user_b",
        "preferred_brands": ["Samsung", "Sony", "Apple"],
        "preferred_categories": ["Electronics", "Mobiles", "TV"],
        "budget_min": 15000,
        "budget_max": 80000,
        "premium_preference": True,
    },
    {
        "user_id": "user_c",
        "preferred_brands": ["Boat", "HP", "LG"],
        "preferred_categories": ["Headphones", "Laptops", "Accessories"],
        "budget_min": 1000,
        "budget_max": 25000,
        "premium_preference": False,
    },
]


def generate_products(count: int = 5000) -> list[dict]:
    products = []
    for _ in range(count):
        brand = random.choice(BRANDS)
        category = random.choice(CATEGORIES)
        color = random.choice(COLORS)
        adj = random.choice(ADJECTIVES)

        if category["name"] in ("Electronics", "Mobiles", "Laptops", "TV"):
            price = round(random.uniform(5000, 80000), 2)
        elif category["name"] == "Shoes":
            price = round(random.uniform(999, 8000), 2)
        else:
            price = round(random.uniform(499, 15000), 2)

        title = f"{adj} {brand['name']} {category['name']} {color}"
        description = (
            f"{title} - High quality {category['name'].lower()} from {brand['name']}. "
            f"Available in {color}. Great for everyday use."
        )
        search_text = f"{title} {description} {brand['name']} {category['name']} {color}".lower()

        products.append({
            "id": str(uuid.uuid4())[:8],
            "title": title,
            "description": description,
            "brand": brand["name"],
            "category": category["name"],
            "price": price,
            "rating": round(random.uniform(3.0, 5.0), 1),
            "discount_pct": round(random.choice([0, 5, 10, 15, 20, 30, 40]), 1),
            "in_stock": random.random() > 0.08,
            "color": color,
            "click_count": random.randint(0, 5000),
            "add_to_cart_count": random.randint(0, 800),
            "purchase_count": random.randint(0, 300),
            "search_text": search_text,
        })
    return products


def generate_interactions(products: list[dict], count: int = 50000) -> list[dict]:
    interactions = []
    queries = [
        "red nike shoes under 3000",
        "samsung mobile under 20000",
        "apple laptop",
        "sony headphones",
        "sports shoes",
        "smart tv 55 inch",
        "nik shoes",
        "puma running shoes",
        "boat earbuds",
        "premium electronics",
    ]

    for _ in range(count):
        product = random.choice(products)
        user = random.choice(SYNTHETIC_USERS)
        interaction_type = random.choices(
            ["click", "add_to_cart", "purchase", "search"],
            weights=[0.5, 0.25, 0.1, 0.15],
        )[0]

        interactions.append({
            "interaction_id": str(uuid.uuid4())[:12],
            "user_id": user["user_id"],
            "product_id": product["id"],
            "interaction_type": interaction_type,
            "query": random.choice(queries) if interaction_type == "search" else None,
            "timestamp": f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}T{random.randint(0,23):02d}:00:00",
        })
    return interactions


def main():
    output_dir = Path(__file__).resolve().parent.parent / "data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    products = generate_products(5000)
    interactions = generate_interactions(products)

    with open(output_dir / "brands.json", "w") as f:
        json.dump(BRANDS, f, indent=2)
    with open(output_dir / "categories.json", "w") as f:
        json.dump(CATEGORIES, f, indent=2)
    with open(output_dir / "products.json", "w") as f:
        json.dump(products, f, indent=2)
    with open(output_dir / "user_interactions.json", "w") as f:
        json.dump(interactions, f, indent=2)
    with open(output_dir / "user_profiles.json", "w") as f:
        json.dump(SYNTHETIC_USERS, f, indent=2)

    print(f"Generated {len(products)} products, {len(interactions)} interactions")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
