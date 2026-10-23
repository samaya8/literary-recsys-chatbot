"""Unit tests for the Hybrid Recommendation & Explainability Engine.

Tests coverage:
- Adaptive alpha balance determination
- Score fusion between collaborative and semantic engines
- Exclusion of seed books and user history
- Human-readable explanation generation
- Facet filtering and fallback handling
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridBookRecommender
from src.semantic_recommender import SemanticBookRecommender


@pytest.fixture
def hybrid_suite():
    """Build mock data and hybrid recommender instance."""
    books = pd.DataFrame([
        {"index": 0, "title": "The Hobbit", "authors": "J.R.R. Tolkien", "categories": "Fantasy", "description": "Hobbit journey with dwarves to the mountain."},
        {"index": 1, "title": "The Fellowship of the Ring", "authors": "J.R.R. Tolkien", "categories": "Fantasy", "description": "Frodo embarks to destroy the ring in Mordor."},
        {"index": 2, "title": "1984", "authors": "George Orwell", "categories": "Science Fiction", "description": "Dystopian totalitarian surveillance state."},
        {"index": 3, "title": "Brave New World", "authors": "Aldous Huxley", "categories": "Science Fiction", "description": "Futuristic controlled biological hierarchy."},
        {"index": 4, "title": "Clean Code", "authors": "Robert C. Martin", "categories": "Computer Science", "description": "Agile software craftsmanship guidelines."},
    ])

    ratings = pd.DataFrame([
        {"user_id": 1, "book_id": 0, "rating": 5.0},
        {"user_id": 1, "book_id": 1, "rating": 4.5},
        {"user_id": 1, "book_id": 2, "rating": 2.0},
        {"user_id": 2, "book_id": 2, "rating": 5.0},
        {"user_id": 2, "book_id": 3, "rating": 4.8},
        {"user_id": 3, "book_id": 4, "rating": 5.0},
    ])

    cf = CollaborativeRecommender(ratings_df=ratings, books_df=books, n_factors=2)
    sem = SemanticBookRecommender(df=books, n_components=2)
    hybrid = HybridBookRecommender(collaborative_engine=cf, semantic_engine=sem)

    return {"hybrid": hybrid, "books": books, "ratings": ratings}


def test_adaptive_alpha_computation(hybrid_suite):
    hybrid = hybrid_suite["hybrid"]

    # Cold start user (user_id=999) should receive 0.0 alpha (pure semantic)
    alpha_cold = hybrid._determine_adaptive_alpha(user_id=999)
    assert alpha_cold == 0.0

    # User 1 has 3 ratings -> emerging reader alpha 0.40
    alpha_u1 = hybrid._determine_adaptive_alpha(user_id=1)
    assert alpha_u1 == 0.40

    # Manual alpha override
    assert hybrid._determine_adaptive_alpha(user_id=1, manual_alpha=0.85) == 0.85


def test_hybrid_recommend_with_seed_title(hybrid_suite):
    hybrid = hybrid_suite["hybrid"]

    # Query with seed title "The Hobbit" for user 2 (Sci-Fi lover)
    recs = hybrid.recommend(user_id=2, seed_title="The Hobbit", top_k=2)

    assert len(recs) == 2
    rec_ids = [r["book_id"] for r in recs]
    # Seed title itself (book 0) must be excluded
    assert 0 not in rec_ids

    top_rec = recs[0]
    assert "hybrid_score" in top_rec
    assert "explanation" in top_rec
    assert "Recommended because" in top_rec["explanation"]


def test_hybrid_recommend_cold_start(hybrid_suite):
    hybrid = hybrid_suite["hybrid"]

    # Cold start user with free-form query
    recs = hybrid.recommend(user_id=None, query_text="dystopian totalitarian state", top_k=2)
    assert len(recs) == 2
    titles = [r["title"] for r in recs]
    assert "1984" in titles or "Brave New World" in titles


def test_hybrid_facet_filter(hybrid_suite):
    hybrid = hybrid_suite["hybrid"]

    recs = hybrid.recommend(user_id=1, seed_title="The Hobbit", top_k=5, category_filter="Science Fiction")
    for r in recs:
        assert r["category"] == "Science Fiction"
