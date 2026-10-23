"""Streamlit Application: Bookworm Studio - Intelligent Bookstore & Reading Assistant.

A customer-facing hybrid book recommendation storefront featuring multi-turn
conversational memory, persona-driven personalization, and transparent recommendations.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.collaborative import CollaborativeRecommender
from src.conversational_assistant import ConversationalBookstoreAssistant
from src.hybrid import HybridBookRecommender
from src.semantic_recommender import SemanticBookRecommender

# Page Configuration
st.set_page_config(
    page_title="ReadNext AI | Intelligent Literary Advisor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .book-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .book-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.06);
    }
    .book-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0F172A;
    }
    .book-author {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 0.4rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        background-color: #EEF2FF;
        color: #4F46E5;
        margin-right: 0.4rem;
    }
    .score-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #ECFDF5;
        color: #059669;
    }
    .rationale-box {
        background-color: #FEF3C7;
        border-left: 3px solid #F59E0B;
        padding: 0.5rem 0.8rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #92400E;
        margin-top: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_engines():
    """Instantiate and cache recommender engines and assistant."""
    cf = CollaborativeRecommender()
    sem = SemanticBookRecommender()
    hybrid = HybridBookRecommender(collaborative_engine=cf, semantic_engine=sem)
    assistant = ConversationalBookstoreAssistant(hybrid_engine=hybrid)
    return cf, sem, hybrid, assistant


cf_engine, sem_engine, hybrid_engine, assistant = load_engines()

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your AI Bookstore Concierge. Tell me what kind of books you enjoy, ask for recommendations based on a title you loved, or ask about our inventory!",
            "recommendations": [],
            "intent": "greeting",
        }
    ]

# Sidebar: Persona & Controls
st.sidebar.title("📖 Reader Settings")

persona_options = {
    "User 1: Sci-Fi & Dystopian Fan": 1,
    "User 2: Classic Literature Admirer": 2,
    "User 3: Fantasy & Adventure Seeker": 3,
    "New Reader (Cold Start)": None,
    "Custom User ID": "custom",
}

selected_persona_label = st.sidebar.selectbox("Active Reader Profile", list(persona_options.keys()))
if selected_persona_label == "Custom User ID":
    user_id_val = st.sidebar.number_input("Enter User ID", min_value=1, max_value=100, value=1)
    active_user_id = int(user_id_val)
else:
    active_user_id = persona_options[selected_persona_label]

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Recommender Config")
strategy = st.sidebar.selectbox(
    "Engine Strategy",
    ["Hybrid (Adaptive Fusion)", "Semantic (LSI & MMR)", "Collaborative (SVD)"],
)

top_k = st.sidebar.slider("Top Recommendations (K)", min_value=1, max_value=10, value=4)

all_categories = ["All"] + sorted(list(set(sem_engine.df["category"].dropna().unique())))
selected_genre = st.sidebar.selectbox("Filter by Category", all_categories)
category_filter = None if selected_genre == "All" else selected_genre

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Assistant Memory State")
mem_state = assistant.memory.get_context_summary()
st.sidebar.text(f"Turns: {mem_state['turn_count']}")
if mem_state["preferred_genres"]:
    st.sidebar.markdown(f"**Preferred Genres:** `{', '.join(mem_state['preferred_genres'])}`")
if mem_state["disliked_genres"]:
    st.sidebar.markdown(f"**Disliked Genres:** `{', '.join(mem_state['disliked_genres'])}`")
if mem_state["mentioned_titles"]:
    st.sidebar.markdown(f"**Recent Titles:** `{', '.join(mem_state['mentioned_titles'])}`")

if st.sidebar.button("🧹 Clear Dialogue Memory"):
    assistant.memory.clear()
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Dialogue memory cleared! What would you like to explore next?",
            "recommendations": [],
            "intent": "greeting",
        }
    ]
    st.sidebar.success("Memory reset.")
    st.rerun()

# Main Header
st.markdown('<div class="main-header">📚 Bookworm Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-Powered Hybrid Recommendation & Conversational Reading Concierge</div>',
    unsafe_allow_html=True,
)

# Tabs
tab_chat, tab_discover, tab_catalog = st.tabs(
    ["💬 Conversational Concierge", "🔍 Semantic & Hybrid Discovery", "📑 Catalog Inventory"]
)

# Tab 1: Conversational Chatbot
with tab_chat:
    st.markdown("##### Chat with your AI Reading Assistant")
    st.caption("Ask questions like: *'I love dystopian sci-fi like 1984'*, *'Suggest a fantasy epic'*, or *'What is The Hobbit about?'*")

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("recommendations"):
                cols = st.columns(min(len(msg["recommendations"]), 3))
                for idx, rec in enumerate(msg["recommendations"][:3]):
                    with cols[idx % 3]:
                        score_label = f"Score: {rec.get('hybrid_score', rec.get('similarity_score', rec.get('predicted_rating', 'N/A')))}"
                        st.markdown(
                            f"""
                            <div class="book-card">
                                <div class="book-title">{rec['title']}</div>
                                <div class="book-author">by {rec.get('author', 'Unknown')}</div>
                                <span class="badge">{rec.get('category', 'Fiction')}</span>
                                <span class="score-badge">{score_label}</span>
                                <div class="rationale-box">💡 {rec.get('explanation', 'Recommended for your taste')}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Prompt Suggestions
    st.markdown("**Quick Prompts:**")
    prompt_cols = st.columns(4)
    suggested_prompt = None
    if prompt_cols[0].button("🚀 Dystopian Sci-Fi"):
        suggested_prompt = "I really loved Dune, can you recommend another gripping sci-fi book?"
    if prompt_cols[1].button("🧙 Fantasy Adventure"):
        suggested_prompt = "Recommend me an epic fantasy adventure with magic and quests."
    if prompt_cols[2].button("🏛️ Classic Novels"):
        suggested_prompt = "What are the best classic literature books in your catalog?"
    if prompt_cols[3].button("🚫 No Romance"):
        suggested_prompt = "Suggest a mystery novel, but no romance please."

    # User Input
    user_input = st.chat_input("Type your book query or conversation here...") or suggested_prompt

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching library & consulting recommendation engine..."):
                resp = assistant.chat(user_input, user_id=active_user_id)
                st.markdown(resp["response"])

                recs = resp.get("recommendations", [])
                if recs:
                    cols = st.columns(min(len(recs), 3))
                    for idx, rec in enumerate(recs[:3]):
                        with cols[idx % 3]:
                            score_label = f"Score: {rec.get('hybrid_score', rec.get('similarity_score', rec.get('predicted_rating', 'N/A')))}"
                            st.markdown(
                                f"""
                                <div class="book-card">
                                    <div class="book-title">{rec['title']}</div>
                                    <div class="book-author">by {rec.get('author', 'Unknown')}</div>
                                    <span class="badge">{rec.get('category', 'Fiction')}</span>
                                    <span class="score-badge">{score_label}</span>
                                    <div class="rationale-box">💡 {rec.get('explanation', 'Recommended based on your query')}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            st.caption(f"🎯 Detected Intent: `{resp['intent']}` (Confidence: {resp['confidence']*100:.1f}%)")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": resp["response"],
                "recommendations": recs,
                "intent": resp["intent"],
            }
        )

# Tab 2: Discovery & Recommendation Showcase
with tab_discover:
    st.markdown("##### Direct Discovery Engine")
    col1, col2 = st.columns([2, 1])

    with col1:
        seed_title = st.selectbox(
            "Select a Seed Book (Optional)",
            ["None"] + sorted(sem_engine.df["title"].dropna().tolist()),
        )
        nl_query = st.text_input("Or enter a Natural Language Search Query", placeholder="e.g., epic space rebellion or psychological mystery")

    with col2:
        st.markdown("**Engine Settings Summary:**")
        st.write(f"- Active Strategy: `{strategy}`")
        st.write(f"- Reader Profile: `{selected_persona_label}`")
        st.write(f"- Genre Filter: `{selected_genre}`")
        run_rec = st.button("🚀 Find Recommendations", type="primary")

    if run_rec:
        effective_title = None if seed_title == "None" else seed_title
        effective_query = None if not nl_query.strip() else nl_query.strip()

        with st.spinner("Generating recommendations..."):
            if "Collaborative" in strategy:
                if active_user_id is not None:
                    recs = cf_engine.recommend_for_user(user_id=active_user_id, top_k=top_k)
                elif effective_title:
                    matched = sem_engine.find_closest_title(effective_title)
                    b_id = sem_engine.df[sem_engine.df["title"] == matched].index[0]
                    recs = cf_engine.get_similar_items(book_id=b_id, top_k=top_k)
                else:
                    recs = cf_engine.recommend_for_user(user_id=1, top_k=top_k)
            elif "Semantic" in strategy:
                if effective_title:
                    recs = sem_engine.recommend_by_title(
                        title_query=effective_title,
                        top_k=top_k,
                        category_filter=category_filter,
                    )
                else:
                    recs = sem_engine.discover_by_query(
                        query_text=effective_query or "popular acclaimed books",
                        top_k=top_k,
                        category_filter=category_filter,
                    )
            else:  # Hybrid
                recs = hybrid_engine.recommend(
                    user_id=active_user_id,
                    seed_title=effective_title,
                    query_text=effective_query,
                    top_k=top_k,
                    category_filter=category_filter,
                )

        st.markdown(f"#### Top {len(recs)} Recommendations")
        if not recs:
            st.info("No matching books found for the given criteria. Try broadening your filter.")
        else:
            grid_cols = st.columns(min(len(recs), 3))
            for i, item in enumerate(recs):
                col_idx = i % min(len(recs), 3)
                with grid_cols[col_idx]:
                    score_val = item.get("hybrid_score", item.get("similarity_score", item.get("predicted_rating", 0.0)))
                    st.markdown(
                        f"""
                        <div class="book-card">
                            <div class="book-title">{item['title']}</div>
                            <div class="book-author">by {item.get('author', 'Unknown')}</div>
                            <span class="badge">{item.get('category', 'General')}</span>
                            <span class="score-badge">Score: {score_val}</span>
                            <p style="font-size:0.88rem; color:#4B5563; margin-top:0.5rem;">
                                {item.get('description', '')[:160]}...
                            </p>
                            <div class="rationale-box">💡 {item.get('explanation', 'Matches your profile and query')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Tab 3: Catalog Inventory
with tab_catalog:
    st.markdown("##### Bookstore Catalog Explorer")
    search_catalog = st.text_input("Filter Catalog by Keyword (Title, Author, Genre):", "")
    df_display = sem_engine.df.copy()
    if search_catalog.strip():
        term = search_catalog.lower()
        mask = (
            df_display["title"].str.lower().str.contains(term, na=False)
            | df_display["author"].str.lower().str.contains(term, na=False)
            | df_display["category"].str.lower().str.contains(term, na=False)
        )
        df_display = df_display[mask]

    st.dataframe(
        df_display[["title", "author", "category", "rating", "description"]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Showing {len(df_display)} books in inventory.")
