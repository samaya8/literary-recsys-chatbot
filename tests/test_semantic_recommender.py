"""Unit tests for the Semantic Vector Discovery & Multi-Field Content Engine.

Tests coverage:
- Semantic index construction with multi-field weighting
- Fuzzy title resolution
- Title-based recommendation with MMR diversification
- Free-form natural language query discovery
- Facet filtering and edge cases
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.semantic_recommender import SemanticBookRecommender


@pytest.fixture
def mock_catalog():
    """Create sample book catalog for semantic testing."""
    return pd.DataFrame([
        {
            "index": 0,
            "title": "The Hobbit",
            "authors": "J.R.R. Tolkien",
            "categories": "Fantasy",
            "description": "Bilbo Baggins embarks on a dangerous quest with Gandalf and dwarves to reclaim the Lonely Mountain.",
        },
        {
            "index": 1,
            "title": "The Fellowship of the Ring",
            "authors": "J.R.R. Tolkien",
            "categories": "Fantasy",
            "description": "Frodo Baggins must journey across Middle-earth to destroy the One Ring in the fires of Mount Doom.",
        },
        {
            "index": 2,
            "title": "1984",
            "authors": "George Orwell",
            "categories": "Science Fiction",
            "description": "Winston Smith lives in a dystopian totalitarian society ruled by Big Brother and the Party.",
        },
        {
            "index": 3,
            "title": "Brave New World",
            "authors": "Aldous Huxley",
            "categories": "Science Fiction",
            "description": "A futuristic utopian society controls citizens through conditioning and technology.",
        },
        {
            "index": 4,
            "title": "Clean Code",
            "authors": "Robert C. Martin",
            "categories": "Computer Science",
            "description": "A handbook of agile software craftsmanship and best practices for writing readable, maintainable software.",
        },
    ])


def test_semantic_recommender_initialization(mock_catalog):
    rec = SemanticBookRecommender(df=mock_catalog, n_components=3)
    assert rec.dense_embeddings.shape[0] == 5
    assert rec.dense_embeddings.shape[1] <= 3
    assert rec.similarity_matrix.shape == (5, 5)


def test_fuzzy_title_matching(mock_catalog):
    rec = SemanticBookRecommender(df=mock_catalog)
    match_1 = rec.find_closest_title("hobbit")
    assert match_1 == "The Hobbit"

    match_2 = rec.find_closest_title("fellowship ring")
    assert match_2 == "The Fellowship of the Ring"

    match_none = rec.find_closest_title("xyz non-existent book 999")
    assert match_none == ""


def test_recommend_by_title(mock_catalog):
    rec = SemanticBookRecommender(df=mock_catalog)
    recs = rec.recommend_by_title("The Hobbit", top_k=2)

    assert len(recs) == 2
    top = recs[0]
    # The Fellowship of the Ring shares author and fantasy themes with The Hobbit
    assert top["title"] == "The Fellowship of the Ring"
    assert top["similarity_score"] > 0.0


def test_discover_by_query(mock_catalog):
    rec = SemanticBookRecommender(df=mock_catalog)

    # Free-form search for software engineering
    code_recs = rec.discover_by_query("writing maintainable software craftmanship", top_k=1)
    assert len(code_recs) == 1
    assert code_recs[0]["title"] == "Clean Code"

    # Free-form search for dystopia
    dystopia_recs = rec.discover_by_query("totalitarian government surveillance", top_k=2)
    assert len(dystopia_recs) == 2
    titles = [r["title"] for r in dystopia_recs]
    assert "1984" in titles or "Brave New World" in titles


def test_category_filtering(mock_catalog):
    rec = SemanticBookRecommender(df=mock_catalog)
    recs = rec.discover_by_query("adventures and journey", top_k=5, category_filter="Science Fiction")
    for r in recs:
        assert r["category"] == "Science Fiction"


def test_empty_query_edge_case(mock_catalog):
    rec = SemanticBookRecommender(df=mock_catalog)
    assert rec.discover_by_query("") == []
    assert rec.recommend_by_title("Unknown Phantom Title") == []
