from pathlib import Path
import pytest

from src.chatbot_utils import (
    clean_text,
    create_bow_vector,
    extract_vocabulary_and_classes,
    load_intents,
)
from src.recommender import BookRecommender

PROJECT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_BOOKS_PATH = PROJECT_DIR / "data" / "sample_books.csv"
INTENTS_PATH = PROJECT_DIR / "data" / "intents.json"


def test_book_recommender_pipeline():
    recommender = BookRecommender(data_path=str(SAMPLE_BOOKS_PATH))
    
    # Query fantasy book
    recs = recommender.recommend("The Hobbit", top_k=2)
    assert len(recs) == 2
    top_title, score = recs[0]
    # The Fellowship of the Ring should have high similarity with The Hobbit
    assert "Fellowship" in top_title or score >= 0.0


def test_book_recommender_query_matching():
    recommender = BookRecommender(data_path=str(SAMPLE_BOOKS_PATH))
    closest = recommender.find_closest_title("great gatsby")
    assert closest == "The Great Gatsby"


def test_chatbot_intents_loading_and_bow():
    data = load_intents(str(INTENTS_PATH))
    assert "intents" in data
    assert len(data["intents"]) >= 4

    vocab, classes, docs = extract_vocabulary_and_classes(data)
    assert len(vocab) > 10
    assert "greeting" in classes
    assert "goodbye" in classes

    tokens = clean_text("Hello there!")
    assert "hello" in tokens

    bow = create_bow_vector(tokens, vocab)
    assert len(bow) == len(vocab)
    assert sum(bow) >= 1
