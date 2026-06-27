import os

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Search Analytics Dashboard",
    page_icon="🔍",
    layout="wide",
)

st.title("AI Product Search — Analytics Dashboard")
st.caption("Track search latency, CTR, NDCG@10, MRR, zero-result rate, and cache performance")


@st.cache_data(ttl=30)
def fetch_metrics():
    try:
        resp = httpx.get(f"{API_BASE}/api/v1/analytics/metrics", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Could not fetch metrics: {e}")
        return None


def run_search(query: str, user_id: str | None = None):
    try:
        resp = httpx.post(
            f"{API_BASE}/api/v1/search",
            json={"query": query, "user_id": user_id, "limit": 10},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Search failed: {e}")
        return None


col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Search Demo")
    query = st.text_input("Search query", value="red nike shoes under 3000")
    user_id = st.selectbox(
        "User profile (personalization)",
        options=[None, "user_a", "user_b", "user_c"],
        format_func=lambda x: {
            None: "Anonymous",
            "user_a": "User A — Nike/Sports (₹2k–5k)",
            "user_b": "User B — Samsung/Electronics (Premium)",
            "user_c": "User C — Boat/HP/Accessories",
        }.get(x, str(x)),
    )

    if st.button("Search", type="primary"):
        with st.spinner("Searching..."):
            result = run_search(query, user_id)
        if result:
            st.success(
                f"Found {result['total_results']} results in {result['latency_ms']}ms "
                f"{'(cache hit)' if result['cache_hit'] else ''}"
            )

            if result.get("zero_result_recovery"):
                st.warning("Zero-result recovery activated")
                for suggestion in result.get("recovery_suggestions", []):
                    st.info(suggestion)

            parsed = result["parsed_query"]
            st.json({
                "brand": parsed.get("brand"),
                "category": parsed.get("category"),
                "color": parsed.get("color"),
                "budget": parsed.get("budget"),
                "corrected_query": parsed.get("corrected_query"),
            })

            if result["results"]:
                df = pd.DataFrame(result["results"])
                st.dataframe(
                    df[["rank", "title", "brand", "category", "price", "final_score",
                        "relevance_score", "personalization_score", "business_score"]],
                    use_container_width=True,
                )

with col2:
    st.subheader("Quick Demos")
    demo_queries = [
        ("Typo recovery", "Nik shoes"),
        ("Budget filter", "samsung mobile under 20000"),
        ("Brand + category", "apple laptop"),
        ("Zero-result edge", "xyzabc123"),
    ]
    for label, demo_query in demo_queries:
        if st.button(label, key=demo_query):
            result = run_search(demo_query, user_id)
            if result:
                st.session_state["last_demo"] = result

st.divider()
st.subheader("KPI Metrics")

metrics = fetch_metrics()
if metrics:
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Searches", metrics["total_searches"])
    m2.metric("P50 Latency", f"{metrics['search_latency_p50_ms']} ms")
    m3.metric("P95 Latency", f"{metrics['search_latency_p95_ms']} ms")
    m4.metric("CTR", f"{metrics['ctr']:.2%}")
    m5.metric("Zero-Result Rate", f"{metrics['zero_result_rate']:.2%}")
    m6.metric("Cache Hit Ratio", f"{metrics['cache_hit_ratio']:.2%}")

    kpi_data = pd.DataFrame([
        {"Metric": "NDCG@10", "Value": metrics["ndcg_at_10"]},
        {"Metric": "MRR", "Value": metrics["mrr"]},
        {"Metric": "CTR", "Value": metrics["ctr"]},
        {"Metric": "Cache Hit Ratio", "Value": metrics["cache_hit_ratio"]},
    ])
    fig = px.bar(kpi_data, x="Metric", y="Value", title="Search Quality KPIs", color="Metric")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Personalization Comparison")
col_a, col_b = st.columns(2)
compare_query = st.text_input("Query for A/B comparison", value="sports shoes")

if st.button("Compare User A vs User B"):
    result_a = run_search(compare_query, "user_a")
    result_b = run_search(compare_query, "user_b")

    if result_a and result_b:
        with col_a:
            st.markdown("**User A (Nike/Sports)**")
            for r in result_a["results"][:5]:
                st.write(f"{r['rank']}. {r['title']} — ₹{r['price']} (score: {r['final_score']})")
        with col_b:
            st.markdown("**User B (Samsung/Electronics)**")
            for r in result_b["results"][:5]:
                st.write(f"{r['rank']}. {r['title']} — ₹{r['price']} (score: {r['final_score']})")

st.divider()
st.subheader("Architecture")
st.code("""
React/Streamlit → FastAPI Gateway → Query Understanding
  → Hybrid Retrieval (BM25 + FAISS) → RRF Fusion
  → LightGBM Ranker → Personalization → Redis Cache → Results
""", language="text")
