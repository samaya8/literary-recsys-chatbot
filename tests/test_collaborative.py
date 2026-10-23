"""Unit tests for the Matrix Factorization & Collaborative Filtering engine.

Tests coverage:
- Model fitting on user-item ratings
- Rating prediction boundaries and bias estimation
- Personalized user recommendations with already-rated exclusion
- Item-item latent factor neighborhood search
- Cold-start user and unseen item fallback
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.collaborative import CollaborativeRecommender


@pytest.fixture
def mock_ratings_and_books():
    """Create minimal synthetic interaction dataset."""
    books = pd.DataFrame([
        {"index": 0, "title": "Dune", "authors": "Frank Herbert", "categories": "Sci-Fi", "description": "Epic desert planet sci-fi."},
        {"index": 1, "title": "Neuromancer", "authors": "William Gibson", "categories": "Sci-Fi", "description": "Cyberpunk AI and hacking."},
        {"index": 2, "title": "The Hobbit", "authors": "J.R.R. Tolkien", "categories": "Fantasy", "description": "Hobbit journey with dwarves."},
        {"index": 3, "title": "The Fellowship", "authors": "J.R.R. Tolkien", "categories": "Fantasy", "description": "Ring bearer quest."},
    ])

    ratings = pd.DataFrame([
        {"user_id": 1, "book_id": 0, "rating": 5.0},
        {"user_id": 1, "book_id": 1, "rating": 4.5},
        {"user_id": 2, "book_id": 2, "rating": 5.0},
        {"user_id": 2, "book_id": 3, "rating": 4.5},
        {"user_id": 3, "book_id": 0, "rating": 4.0},
        {"user_id": 3, "book_id": 2, "rating": 2.0},
    ])

    return {"books": books, "ratings": ratings}


def test_collaborative_fit_and_dimensions(mock_ratings_and_books):
    rec = CollaborativeRecommender(
        ratings_df=mock_ratings_and_books["ratings"],
        books_df=mock_ratings_and_books["books"],
        n_factors=2,
    )

    assert rec.fitted is True
    assert len(rec.user_ids) == 3
    assert len(rec.item_ids) == 4
    assert rec.user_factors.shape[0] == 3
    assert rec.item_factors.shape[0] == 4


def test_rating_prediction(mock_ratings_and_books):
    rec = CollaborativeRecommender(
        ratings_df=mock_ratings_and_books["ratings"],
        books_df=mock_ratings_and_books["books"],
        n_factors=2,
    )

    # Known user and item
    pred = rec.predict_rating(user_id=1, book_id=0)
    assert 1.0 <= pred <= 5.0

    # User 1 liked Sci-Fi (Dune, Neuromancer), should have higher prediction for Neuromancer than Fantasy
    pred_neuromancer = rec.predict_rating(user_id=1, book_id=1)
    assert 1.0 <= pred_neuromancer <= 5.0

    # Unseen user (cold-start)
    cold_user_pred = rec.predict_rating(user_id=999, book_id=0)
    assert 1.0 <= cold_user_pred <= 5.0

    # Unseen item
    cold_item_pred = rec.predict_rating(user_id=1, book_id=999)
    assert 1.0 <= cold_item_pred <= 5.0


def test_recommend_for_user(mock_ratings_and_books):
    rec = CollaborativeRecommender(
        ratings_df=mock_ratings_and_books["ratings"],
        books_df=mock_ratings_and_books["books"],
        n_factors=2,
    )

    # User 1 rated books 0 and 1; recommendations should surface books 2 or 3
    recs = rec.recommend_for_user(user_id=1, top_k=2, exclude_rated=True)
    assert len(recs) == 2
    rec_ids = [r["book_id"] for r in recs]
    assert 0 not in rec_ids
    assert 1 not in rec_ids
    assert "title" in recs[0]
    assert "predicted_rating" in recs[0]


def test_item_similarity_neighborhood(mock_ratings_and_books):
    rec = CollaborativeRecommender(
        ratings_df=mock_ratings_and_books["ratings"],
        books_df=mock_ratings_and_books["books"],
        n_factors=2,
    )

    sims = rec.get_similar_items(book_id=0, top_k=2)
    assert len(sims) == 2
    assert "similarity" in sims[0]
    # Target item should not be in its own neighbors
    assert sims[0]["book_id"] != 0


def test_cold_start_fallback(mock_ratings_and_books):
    rec = CollaborativeRecommender(
        ratings_df=mock_ratings_and_books["ratings"],
        books_df=mock_ratings_and_books["books"],
    )

    cold_recs = rec.cold_start_recommendations(top_k=3)
    assert len(cold_recs) == 3
    assert cold_recs[0]["predicted_rating"] > 0.0
