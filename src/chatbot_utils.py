"""Intent parsing and preprocessing utilities for the bookstore chatbot."""

import json
import re
from typing import Dict, List, Tuple


def clean_text(text: str) -> List[str]:
    """Tokenize and normalize text without heavy external dependencies."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens


def load_intents(filepath: str) -> Dict:
    """Load and validate the intent schema JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "intents" not in data:
        raise ValueError("Invalid schema: 'intents' key missing.")
    return data


def extract_vocabulary_and_classes(intents_data: Dict) -> Tuple[List[str], List[str], List[Tuple[List[str], str]]]:
    """Extract vocabulary, intent classes, and document pairs from intents data."""
    words = set()
    classes = set()
    documents = []

    for intent in intents_data["intents"]:
        tag = intent["tag"]
        classes.add(tag)
        for pattern in intent["patterns"]:
            tokens = clean_text(pattern)
            words.update(tokens)
            documents.append((tokens, tag))

    return sorted(list(words)), sorted(list(classes)), documents


def create_bow_vector(tokens: List[str], vocabulary: List[str]) -> List[int]:
    """Convert tokenized words into a binary bag-of-words vector."""
    token_set = set(tokens)
    return [1 if word in token_set else 0 for word in vocabulary]
