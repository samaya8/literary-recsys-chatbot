"""Conversational Bookstore Assistant with Multi-Turn Memory & Hybrid Recommendation Dispatching.

Features:
1. Intent Classifier: Calibrated TF-IDF intent prediction with confidence thresholding.
2. Entity & Constraint Extractor: Parses book titles, authors, genres, and negative constraints.
3. Multi-Turn Dialogue Memory: Tracks ongoing session preferences and sliding conversation context.
4. Context-Aware Dispatcher: Fuses conversation intents with live hybrid book recommendations.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.chatbot_utils import clean_text, load_intents
from src.hybrid import HybridBookRecommender


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INTENTS_PATH = PROJECT_DIR / "data" / "intents.json"


class DialogueMemory:
    """Maintains multi-turn conversational state, preferences, and recommendation history."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages: List[Dict[str, Any]] = []
        self.favorite_genres: Set[str] = set()
        self.disliked_genres: Set[str] = set()
        self.mentioned_titles: Set[str] = set()
        self.mentioned_authors: Set[str] = set()
        self.last_recommendations: List[Dict[str, Any]] = []

    def add_user_message(self, text: str):
        """Append user utterance to conversation log."""
        self.messages.append({
            "role": "user",
            "content": text,
            "timestamp": time.time(),
        })
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-self.max_turns * 2 :]

    def add_assistant_message(self, text: str, recommendations: Optional[List[Dict[str, Any]]] = None):
        """Append assistant utterance and store active recommendation state."""
        self.messages.append({
            "role": "assistant",
            "content": text,
            "timestamp": time.time(),
            "recommendations": recommendations or [],
        })
        if recommendations:
            self.last_recommendations = recommendations

    def update_profile(self, extracted_entities: Dict[str, Any]):
        """Incorporate parsed entities and constraints into ongoing profile."""
        for g in extracted_entities.get("genres", []):
            self.favorite_genres.add(g)
        for g in extracted_entities.get("disliked_genres", []):
            self.disliked_genres.add(g)
            if g in self.favorite_genres:
                self.favorite_genres.remove(g)
        for t in extracted_entities.get("titles", []):
            self.mentioned_titles.add(t)
        for a in extracted_entities.get("authors", []):
            self.mentioned_authors.add(a)

    def clear(self):
        """Reset conversation session."""
        self.messages.clear()
        self.favorite_genres.clear()
        self.disliked_genres.clear()
        self.mentioned_titles.clear()
        self.mentioned_authors.clear()
        self.last_recommendations.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Return snapshot of current dialogue state."""
        return {
            "total_turns": len(self.messages) // 2,
            "favorite_genres": sorted(list(self.favorite_genres)),
            "disliked_genres": sorted(list(self.disliked_genres)),
            "mentioned_titles": sorted(list(self.mentioned_titles)),
            "last_recommendations_count": len(self.last_recommendations),
        }


class ConversationalBookstoreAssistant:
    """Conversational assistant with intent classification, slot extraction, and hybrid recommendations."""

    KNOWN_GENRES = {
        "fantasy": ["fantasy", "fantasies", "magic", "dragons", "wizards", "middle-earth"],
        "science fiction": ["sci-fi", "scifi", "science fiction", "science-fiction", "space", "dystopian", "cyberpunk", "futuristic", "robots"],
        "classics": ["classic", "classics", "classic literature", "historical", "vintage"],
        "computer science": ["computer science", "programming", "software", "coding", "machine learning", "deep learning", "algorithms"],
    }

    def __init__(
        self,
        intents_path: Optional[Union[str, Path]] = None,
        hybrid_recommender: Optional[HybridBookRecommender] = None,
    ):
        self.intents_path = Path(intents_path) if intents_path else DEFAULT_INTENTS_PATH
        self.recommender = hybrid_recommender or HybridBookRecommender()
        self.memory = DialogueMemory()

        self._train_intent_classifier()

    def _train_intent_classifier(self):
        """Train calibrated TF-IDF intent classifier on intents patterns."""
        if not self.intents_path.exists():
            # Minimal fallback schema
            self.classes_ = ["greeting", "goodbye", "recommendation"]
            self.vectorizer = TfidfVectorizer().fit(["hello", "bye", "recommend book"])
            self.clf = LogisticRegression().fit(self.vectorizer.transform(["hello", "bye", "recommend book"]), [0, 1, 2])
            self.responses_map = {
                "greeting": ["Hello! How can I help you find a great book today?"],
                "goodbye": ["Goodbye and happy reading!"],
                "recommendation": ["Here are some curated book recommendations for you:"],
            }
            return

        intents_data = load_intents(str(self.intents_path))
        texts = []
        labels = []
        self.responses_map = {}

        for intent in intents_data["intents"]:
            tag = intent["tag"]
            self.responses_map[tag] = intent.get("responses", ["How can I assist you?"])
            for pattern in intent["patterns"]:
                texts.append(pattern.lower())
                labels.append(tag)

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        X = self.vectorizer.fit_transform(texts)

        self.classes_ = sorted(list(set(labels)))
        y = np.array([self.classes_.index(lbl) for lbl in labels])

        self.clf = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        self.clf.fit(X, y)

    def classify_intent(self, text: str) -> Tuple[str, float]:
        """Classify user utterance into intent tag and confidence probability."""
        if not text.strip():
            return "unknown", 0.0

        vec = self.vectorizer.transform([text.lower()])
        probs = self.clf.predict_proba(vec)[0]
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])

        if confidence < 0.25:
            return "unknown", round(confidence, 3)

        return self.classes_[top_idx], round(confidence, 3)

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract genres, negative constraints, and matching catalog titles/authors."""
        tokens = clean_text(text)
        lower_text = text.lower()

        extracted_genres = []
        disliked_genres = []
        matched_titles = []
        matched_authors = []

        # 1. Identify negative constraints (e.g. "no fantasy", "don't like horror")
        negative_matches = re.findall(r"(?:no|not|dislike|don't like|hate)\s+([a-zA-Z_-]+)", lower_text)
        for neg in negative_matches:
            for genre_name, syns in self.KNOWN_GENRES.items():
                if neg in syns or neg == genre_name:
                    disliked_genres.append(genre_name)

        # 2. Identify preferred genres
        for genre_name, syns in self.KNOWN_GENRES.items():
            if genre_name not in disliked_genres:
                if any(re.search(rf"\b{re.escape(syn)}\b", lower_text) for syn in syns):
                    extracted_genres.append(genre_name)

        # 3. Fuzzy match catalog titles
        for title in self.recommender.books_df["title"].values:
            if str(title).lower() in lower_text:
                matched_titles.append(str(title))

        # 4. Match catalog authors
        for author in self.recommender.books_df["authors"].dropna().values:
            author_clean = str(author).lower()
            if len(author_clean) > 3 and author_clean in lower_text:
                matched_authors.append(str(author))

        return {
            "genres": extracted_genres,
            "disliked_genres": disliked_genres,
            "titles": matched_titles,
            "authors": matched_authors,
        }

    def chat(
        self,
        user_message: str,
        user_id: Optional[int] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Process user message in conversation context and formulate assistant response."""
        # Record user message in memory
        self.memory.add_user_message(user_message)

        # Entity extraction & memory profile update
        entities = self.extract_entities(user_message)
        self.memory.update_profile(entities)

        # Intent classification
        intent, confidence = self.classify_intent(user_message)

        recommendations = []
        response_text = ""

        lower_msg = user_message.lower()
        has_rec_keywords = bool(re.search(r"\b(suggest|suggestion|suggestions|recommend|recommendation|recommendations|another|more|what else|read next|give me)\b", lower_msg))

        # Check if user message explicitly requests recommendations or mentions titles/genres
        is_rec_trigger = (
            intent in ["recommendation", "search"]
            or bool(entities["genres"])
            or bool(entities["titles"])
            or has_rec_keywords
        )

        if is_rec_trigger:
            # Determine seed title or query
            seed_title = entities["titles"][0] if entities["titles"] else None
            genre_filter = entities["genres"][0].title() if entities["genres"] else None

            # Fallback to memory preferences if current utterance doesn't specify genre
            if not genre_filter and self.memory.favorite_genres:
                genre_filter = list(self.memory.favorite_genres)[0].title()

            recommendations = self.recommender.recommend(
                user_id=user_id,
                seed_title=seed_title,
                query_text=user_message if not seed_title else None,
                top_k=top_k,
                category_filter=genre_filter,
            )

            # Formulate response
            if seed_title and recommendations:
                response_text = f"Since you mentioned **'{seed_title}'**, here are {len(recommendations)} books with similar themes and reader reception:"
            elif genre_filter and recommendations:
                response_text = f"Here are top-rated **{genre_filter}** recommendations tailored for you:"
            elif recommendations:
                response_text = f"Based on your inquiry, here are {len(recommendations)} curated recommendations from our catalog:"
            else:
                response_text = "I couldn't find exact titles matching your criteria. Could you share another genre or favorite author?"
        elif intent in self.responses_map:
            # Deterministic selection from intent responses
            responses = self.responses_map[intent]
            response_text = responses[0]
        else:
            response_text = "I'm your bookstore assistant! You can ask for recommendations by genre (e.g. Fantasy, Sci-Fi), author, or specific book titles."

        # Record assistant reply in memory
        self.memory.add_assistant_message(response_text, recommendations=recommendations)

        return {
            "response": response_text,
            "intent": intent,
            "confidence": confidence,
            "recommendations": recommendations,
            "extracted_entities": entities,
            "memory_summary": self.memory.get_summary(),
        }
