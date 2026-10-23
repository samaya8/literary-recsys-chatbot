"""Unit tests for the Conversational Bookstore Assistant with Multi-Turn Memory.

Tests coverage:
- DialogueMemory message tracking and profile updates
- Entity and negative constraint extraction
- Intent classification across standard bookstore intents
- Multi-turn conversation flow with context retention and recommendation dispatch
"""

from __future__ import annotations

import pytest

from src.conversational_assistant import (
    ConversationalBookstoreAssistant,
    DialogueMemory,
)


def test_dialogue_memory_operations():
    mem = DialogueMemory(max_turns=3)

    mem.add_user_message("Hello, I'm looking for a book.")
    mem.add_assistant_message("Hi! What genre do you enjoy?")

    assert len(mem.messages) == 2
    summary = mem.get_summary()
    assert summary["total_turns"] == 1

    # Update profile with preferred and disliked genres
    mem.update_profile({
        "genres": ["fantasy", "sci-fi"],
        "disliked_genres": ["horror"],
        "titles": ["The Hobbit"],
        "authors": ["Tolkien"],
    })

    assert "fantasy" in mem.favorite_genres
    assert "horror" in mem.disliked_genres
    assert "The Hobbit" in mem.mentioned_titles

    # If user later dislikes fantasy, it should be removed from favorites
    mem.update_profile({"disliked_genres": ["fantasy"]})
    assert "fantasy" not in mem.favorite_genres

    mem.clear()
    assert len(mem.messages) == 0
    assert len(mem.favorite_genres) == 0


def test_assistant_intent_classification():
    assistant = ConversationalBookstoreAssistant()

    # Greetings
    tag_greet, conf_greet = assistant.classify_intent("Hello, good day!")
    assert tag_greet == "greeting"
    assert conf_greet > 0.3

    # Goodbye
    tag_bye, conf_bye = assistant.classify_intent("See you later, goodbye!")
    assert tag_bye == "goodbye"

    # Hours
    tag_hours, conf_hours = assistant.classify_intent("What time do you close today?")
    assert tag_hours == "hours"

    # Recommendation
    tag_rec, conf_rec = assistant.classify_intent("Can you recommend a great book to read?")
    assert tag_rec == "recommendation"


def test_assistant_entity_extraction():
    assistant = ConversationalBookstoreAssistant()

    # Genre and title extraction
    entities = assistant.extract_entities("I loved 1984, are there any other great sci-fi books?")
    assert "science fiction" in entities["genres"]
    assert "1984" in entities["titles"]

    # Negative constraint extraction
    entities_neg = assistant.extract_entities("I want a classic novel, but no fantasy please.")
    assert "classics" in entities_neg["genres"]
    assert "fantasy" in entities_neg["disliked_genres"]


def test_assistant_multi_turn_chat_flow():
    assistant = ConversationalBookstoreAssistant()

    # Turn 1: User states interest in Fantasy
    out_1 = assistant.chat("Hi, I really love fantasy novels with magic.")
    assert len(out_1["recommendations"]) > 0
    # Recommendations should be Fantasy
    rec_cats = [r["category"] for r in out_1["recommendations"]]
    assert "Fantasy" in rec_cats

    # Turn 2: User follows up asking for another recommendation without restating genre
    out_2 = assistant.chat("Give me one more suggestion")
    assert len(out_2["recommendations"]) > 0
    # Should maintain Fantasy context from memory
    assert "fantasy" in assistant.memory.favorite_genres


def test_assistant_title_mention_flow():
    assistant = ConversationalBookstoreAssistant()

    out = assistant.chat("What do you recommend if I liked The Hobbit?")
    assert len(out["recommendations"]) > 0
    # Seed title "The Hobbit" should trigger similar books
    rec_titles = [r["title"] for r in out["recommendations"]]
    assert "The Hobbit" not in rec_titles  # Exclude seed title
    assert "The Fellowship of the Ring" in rec_titles or any("fantasy" in r["category"].lower() for r in out["recommendations"])
