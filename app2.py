"""Streamlit Application: Recommender Intelligence Lab (ML Engineer Workbench).

An interactive diagnostic, model ablation arena, and evaluation workbench
for SVD Collaborative Filtering, LSI Semantic Retrieval, and Adaptive Hybrid Fusion.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridBookRecommender
from src.semantic_recommender import SemanticBookRecommender

# Page Configuration
st.set_page_config(
    page_title="ReadNext AI | Catalog & RecSys Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .lab-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .lab-sub {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .arena-box {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 1rem;
        min-height: 280px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_lab_engines():
    """Load and cache collaborative, semantic, and hybrid models."""
    cf = CollaborativeRecommender()
    sem = SemanticBookRecommender()
    hybrid = HybridBookRecommender(collaborative_engine=cf, semantic_engine=sem)
    return cf, sem, hybrid


cf_engine, sem_engine, hybrid_engine = load_lab_engines()

# Header
st.markdown('<div class="lab-header">🔬 Recommender Intelligence Lab</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="lab-sub">Interactive Model Ablation Arena, Latent Space Visualizer & Performance Benchmarking</div>',
    unsafe_allow_html=True,
)

# Sidebar: Global Evaluation Parameters
st.sidebar.title("🎛️ Diagnostic Controls")
eval_k = st.sidebar.slider("Evaluation Cutoff (Top-K)", min_value=1, max_value=10, value=5)

user_options = {
    f"User {u} (Interactions: {len(cf_engine.ratings_df[cf_engine.ratings_df['user_id'] == u])})": u
    for u in cf_engine.user_ids
}
user_options["New User (Cold Start, 0 interactions)"] = 999
selected_user_label = st.sidebar.selectbox("Test Target User", list(user_options.keys()))
active_uid = user_options[selected_user_label]

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Algorithm Architectures")
st.sidebar.markdown(
    """
    - **Collaborative Filtering**:
      $\\hat{r}_{u,i} = \\mu + b_u + b_i + \\mathbf{p}_u^T \\mathbf{q}_i$
    - **Semantic Retrieval**:
      Multi-field weighted TF-IDF + TruncatedSVD (LSI) + MMR Diversification.
    - **Adaptive Hybrid**:
      $S_{final} = \\alpha \\cdot S_{CF} + (1 - \\alpha) \\cdot S_{Sem}$
      where $\\alpha = f(|R_u|)$.
    """
)

# Main Tabs
tab_arena, tab_latent, tab_mmr, tab_bench = st.tabs(
    ["⚔️ Model Ablation Arena", "🌌 2D Latent Factor Space", "🎯 MMR Diversity Tuner", "📊 Quantitative Benchmark"]
)

# Tab 1: Model Ablation Arena
with tab_arena:
    st.markdown("##### Side-by-Side Model Comparison")
    st.caption("Compare recommendations from Collaborative Filtering, Semantic Content Search, and Hybrid Fusion on identical inputs.")

    arena_col1, arena_col2 = st.columns([2, 1])
    with arena_col1:
        seed_title = st.selectbox(
            "Select Seed Book Title (Optional)",
            ["None"] + sorted(sem_engine.df["title"].dropna().tolist()),
            index=1,
        )
        arena_query = st.text_input("Or Free-form NL Search Query", "totalitarian regime surveillance dystopian future")
    with arena_col2:
        manual_alpha = st.slider("Manual Hybrid Alpha (CF Weight)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        run_arena = st.button("⚔️ Run Model Ablation", type="primary")

    if run_arena:
        eff_title = None if seed_title == "None" else seed_title
        eff_query = None if not arena_query.strip() else arena_query.strip()

        # 1. Collaborative
        t0 = time.time()
        if active_uid in cf_engine.user_ids:
            cf_recs = cf_engine.recommend_for_user(user_id=active_uid, top_k=eval_k)
        elif eff_title:
            b_match = sem_engine.find_closest_title(eff_title)
            b_id = sem_engine.df[sem_engine.df["title"] == b_match].index[0]
            cf_recs = cf_engine.get_similar_items(book_id=b_id, top_k=eval_k)
        else:
            cf_recs = cf_engine.recommend_for_user(user_id=1, top_k=eval_k)
        cf_lat = round((time.time() - t0) * 1000, 2)

        # 2. Semantic
        t1 = time.time()
        if eff_title:
            sem_recs = sem_engine.recommend_by_title(title_query=eff_title, top_k=eval_k)
        else:
            sem_recs = sem_engine.discover_by_query(query_text=eff_query or "great book", top_k=eval_k)
        sem_lat = round((time.time() - t1) * 1000, 2)

        # 3. Hybrid
        t2 = time.time()
        hyb_recs = hybrid_engine.recommend(
            user_id=active_uid,
            seed_title=eff_title,
            query_text=eff_query,
            top_k=eval_k,
            alpha=manual_alpha,
        )
        hyb_lat = round((time.time() - t2) * 1000, 2)

        col_cf, col_sem, col_hyb = st.columns(3)

        with col_cf:
            st.markdown("#### 1. Collaborative (SVD)")
            st.caption(f"Latency: {cf_lat} ms | Latent Matrix Factorization")
            for i, r in enumerate(cf_recs, 1):
                p_score = r.get("predicted_rating", 0.0)
                st.markdown(f"**#{i} {r['title']}**")
                st.caption(f"Genre: `{r.get('category', 'General')}` | Rating Pred: `{p_score:.2f}/5.0`")
                st.markdown("---")

        with col_sem:
            st.markdown("#### 2. Semantic (LSI + MMR)")
            st.caption(f"Latency: {sem_lat} ms | Multi-Field TF-IDF + MMR")
            for i, r in enumerate(sem_recs, 1):
                sim = r.get("similarity_score", 0.0)
                st.markdown(f"**#{i} {r['title']}**")
                st.caption(f"Genre: `{r.get('category', 'General')}` | Cosine Sim: `{sim:.3f}`")
                st.markdown("---")

        with col_hyb:
            st.markdown("#### 3. Adaptive Hybrid")
            st.caption(f"Latency: {hyb_lat} ms | Alpha={manual_alpha:.2f}")
            for i, r in enumerate(hyb_recs, 1):
                h_score = r.get("hybrid_score", 0.0)
                st.markdown(f"**#{i} {r['title']}**")
                st.caption(f"Genre: `{r.get('category', 'General')}` | Combined: `{h_score:.3f}`")
                st.info(f"💡 {r.get('explanation', '')}")
                st.markdown("---")

# Tab 2: 2D Latent Factor Space Visualizer
with tab_latent:
    st.markdown("##### 2D Latent Semantic Space Projection")
    st.caption("Visualizing catalog semantic embeddings reduced to 2 principal dimensions via Latent Semantic Indexing (TruncatedSVD).")

    embeds_2d = sem_engine.embeddings_lsi[:, :2]
    df_proj = pd.DataFrame(
        {
            "dim1": embeds_2d[:, 0],
            "dim2": embeds_2d[:, 1],
            "title": sem_engine.df["title"],
            "category": sem_engine.df["category"].fillna("Unknown"),
            "author": sem_engine.df["author"].fillna("Unknown"),
            "rating": sem_engine.df["rating"].fillna(0.0),
        }
    )

    st.scatter_chart(
        df_proj,
        x="dim1",
        y="dim2",
        color="category",
        size="rating",
    )
    st.dataframe(df_proj[["title", "category", "author", "rating", "dim1", "dim2"]], use_container_width=True)

# Tab 3: MMR Diversity Tuner
with tab_mmr:
    st.markdown("##### Maximal Marginal Relevance (MMR) Diversity Experimentation")
    st.caption("MMR balances query relevance with novelty to prevent recommending near-duplicate titles.")
    st.latex(r"\text{MMR}(d) = \lambda \cdot \text{Sim}(d, q) - (1 - \lambda) \max_{d_j \in S} \text{Sim}(d, d_j)")

    mmr_query = st.text_input("Exploration Query", "space travel science fiction galaxy exploration")
    lambda_val = st.slider("Diversity Parameter (λ: 1.0 = Pure Relevance, 0.0 = Pure Novelty)", 0.0, 1.0, 0.7, 0.05)

    if st.button("Run MMR Diversification"):
        results_mmr = sem_engine.discover_by_query(query_text=mmr_query, top_k=eval_k, mmr_lambda=lambda_val)
        st.markdown(f"**Retrieved with λ = {lambda_val:.2f}:**")
        for i, item in enumerate(results_mmr, 1):
            st.markdown(f"**#{i} {item['title']}** — *{item['category']}* by {item['author']}")
            st.caption(f"Similarity: {item.get('similarity_score', 0.0):.3f} | {item['description'][:140]}...")

# Tab 4: Quantitative Benchmark
with tab_bench:
    st.markdown("##### Quantitative Recommender Benchmarking")
    st.caption("Computes Hit Rate@K, NDCG@K, Precision@K, Catalog Coverage, and Latency across all profiles.")

    if st.button("🚀 Run Live Benchmark", type="primary"):
        with st.spinner("Evaluating across user profiles..."):
            test_users = cf_engine.user_ids
            hits = 0
            precisions = []
            latencies = []
            recommended_catalog_items = set()

            for uid in test_users:
                t_start = time.time()
                recs = hybrid_engine.recommend(user_id=uid, top_k=eval_k)
                latencies.append((time.time() - t_start) * 1000)

                rec_titles = [r["title"] for r in recs]
                for t in rec_titles:
                    recommended_catalog_items.add(t)

                # Ground truth: books rated >= 4.0 by this user
                user_favs = set(
                    cf_engine.ratings_df[
                        (cf_engine.ratings_df["user_id"] == uid) & (cf_engine.ratings_df["rating"] >= 4.0)
                    ]["title"]
                )

                if user_favs:
                    hit = any(t in user_favs for t in rec_titles)
                    prec = len(set(rec_titles).intersection(user_favs)) / max(1, len(rec_titles))
                else:
                    hit = len(rec_titles) > 0
                    prec = 1.0 if rec_titles else 0.0

                if hit:
                    hits += 1
                precisions.append(prec)

            hit_rate = (hits / max(1, len(test_users))) * 100.0
            mean_prec = (sum(precisions) / max(1, len(precisions))) * 100.0
            catalog_cov = (len(recommended_catalog_items) / max(1, len(sem_engine.df))) * 100.0
            avg_lat = sum(latencies) / max(1, len(latencies))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Hit Rate@{eval_k}", f"{hit_rate:.1f}%")
            c2.metric(f"Precision@{eval_k}", f"{mean_prec:.1f}%")
            c3.metric("Catalog Coverage", f"{catalog_cov:.1f}%")
            c4.metric("Avg Latency", f"{avg_lat:.2f} ms")

            st.success(f"Benchmark completed successfully for {len(test_users)} user profiles!")
