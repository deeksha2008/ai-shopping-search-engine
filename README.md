# AI-Powered Product Search & Shopping Intelligence Engine

Production-grade e-commerce search combining **BM25 + FAISS semantic retrieval**, **Reciprocal Rank Fusion**, **LightGBM Learning-to-Rank**, **personalization**, and **zero-result recovery** — built for Flipkart GRID-scale demos.

## Architecture

```
Streamlit Dashboard → FastAPI Gateway → Query Understanding Layer
  → Hybrid Retrieval (BM25 + FAISS) → RRF Fusion → LightGBM Ranker
  → Personalization Engine → Redis Cache → Results → Analytics
```

## Features

| Layer | Components |
|-------|-----------|
| Data Ingestion | Synthetic products, brands, categories, user interactions → PostgreSQL + FAISS |
| Query Understanding | Entity extraction, SymSpell typo correction, normalization, synonym expansion |
| Hybrid Retrieval | BM25 lexical + FAISS semantic + Reciprocal Rank Fusion |
| Learning-to-Rank | LightGBM LambdaRank with 13 query-product & business features |
| Personalization | User-profile re-ranking (0.6 relevance + 0.25 preference + 0.15 business) |
| Zero-Result Recovery | Typo correction → semantic fallback → alternative suggestions |
| Analytics | CTR, NDCG@10, MRR, latency, zero-result rate, cache hit ratio |
| Production | Docker, Redis caching, async FastAPI, structured logging, rate limiting, Prometheus |

## Quick Start (Docker)

```bash
cd ai-shopping-search-engine
cp .env.example .env
docker compose up --build
```

- **API**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501
- **Metrics**: http://localhost:8000/metrics

## Local Development (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate synthetic data + build FAISS index
python scripts/generate_data.py
python scripts/train_ranker.py

# Start API (works with JSON fallback if Postgres/Redis unavailable)
PYTHONPATH=. uvicorn backend.app.main:app --reload --port 8000

# Dashboard (separate terminal)
streamlit run dashboard/app.py
```

With Postgres + Redis running locally:

```bash
python scripts/ingest_data.py
```

## Example Queries

| Query | What it demonstrates |
|-------|---------------------|
| `red nike shoes under 3000` | Entity extraction + budget filter |
| `Nik shoes` | Typo correction + zero-result recovery |
| `samsung mobile under 20000` | Brand + category + budget |
| Same query as **user_a** vs **user_b** | Personalization re-ranking |

## API

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "red nike shoes under 3000", "user_id": "user_a", "limit": 10}'
```

## Project Structure

```
├── backend/app/
│   ├── core/query_understanding/   # Entity extraction, typo correction, synonyms
│   ├── core/retrieval/             # BM25, FAISS, RRF fusion
│   ├── core/ranking/               # LightGBM LTR features + ranker
│   ├── core/personalization/       # User-profile re-ranking
│   ├── core/zero_result/           # Recovery pipeline
│   ├── services/                   # Search, cache, analytics
│   └── api/routes/                 # FastAPI endpoints
├── dashboard/                      # Streamlit analytics UI
├── scripts/                        # Data generation, ingestion, training
├── data/generated/                 # Synthetic JSON datasets
└── models/                         # Trained LightGBM model
```

## Resume Bullets

- Built a production-grade AI Product Search Engine combining BM25, FAISS semantic retrieval, Reciprocal Rank Fusion and LightGBM Learning-to-Rank for large-scale e-commerce search.
- Engineered a query understanding pipeline with entity extraction, typo correction and synonym expansion to improve search relevance.
- Designed a personalization layer and Redis-based caching system to optimize latency (<100 ms) and user engagement metrics.
- Implemented analytics dashboards to track CTR, NDCG@10, MRR and zero-result recovery performance.

## Tech Stack

Python 3.11 · FastAPI · PostgreSQL · Redis · FAISS · SentenceTransformers · LightGBM · SymSpell · Streamlit · Docker
