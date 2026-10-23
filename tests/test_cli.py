"""Unit tests for the Book Recommendation & Chatbot CLI."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

# Ensure project root takes precedence over any site-packages
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Remove any previously cached third-party cli module if present
if "cli" in sys.modules and not getattr(sys.modules["cli"], "__file__", "").startswith(str(ROOT_DIR)):
    del sys.modules["cli"]

from cli import main, build_parser


def test_cli_help(capsys: pytest.CaptureFixture) -> None:
    """Test calling CLI with no arguments or --help."""
    ret = main([])
    assert ret == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower() or "book-recommender-cli" in captured.out


def test_cli_recommend_hybrid_title(capsys: pytest.CaptureFixture) -> None:
    """Test hybrid recommendation by seed title."""
    ret = main(["recommend", "--title", "Dune", "--top-k", "3", "--explain"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "BOOK RECOMMENDATIONS" in captured.out
    assert "Rationale:" in captured.out


def test_cli_recommend_hybrid_query_json(capsys: pytest.CaptureFixture) -> None:
    """Test hybrid discovery by query returning JSON."""
    ret = main(["recommend", "--query", "magic wizard school", "--top-k", "2", "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "recommendations" in data
    assert len(data["recommendations"]) == 2
    assert "HYBRID" in data["strategy"]


def test_cli_recommend_collaborative_user(capsys: pytest.CaptureFixture) -> None:
    """Test collaborative recommendation for a user."""
    ret = main(["recommend", "--strategy", "collaborative", "--user-id", "1", "--top-k", "3"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "COLLABORATIVE (SVD)" in captured.out


def test_cli_recommend_collaborative_title(capsys: pytest.CaptureFixture) -> None:
    """Test collaborative item-item recommendation by title."""
    ret = main(["recommend", "--strategy", "collaborative", "--title", "1984", "--top-k", "3"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "COLLABORATIVE (SVD)" in captured.out


def test_cli_recommend_collaborative_missing_args(capsys: pytest.CaptureFixture) -> None:
    """Test collaborative strategy error when neither user-id nor title is given."""
    ret = main(["recommend", "--strategy", "collaborative"])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Error: Collaborative strategy requires" in captured.err


def test_cli_recommend_semantic_query(capsys: pytest.CaptureFixture) -> None:
    """Test semantic recommendation by natural language query."""
    ret = main(["recommend", "--strategy", "semantic", "--query", "dystopian totalitarian surveillance", "--top-k", "2"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "SEMANTIC (LSI & MMR)" in captured.out


def test_cli_recommend_semantic_title(capsys: pytest.CaptureFixture) -> None:
    """Test semantic recommendation by title."""
    ret = main(["recommend", "--strategy", "semantic", "--title", "The Hobbit", "--top-k", "2"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "SEMANTIC (LSI & MMR)" in captured.out


def test_cli_recommend_semantic_missing_args(capsys: pytest.CaptureFixture) -> None:
    """Test semantic strategy error when neither title nor query is given."""
    ret = main(["recommend", "--strategy", "semantic"])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Error: Semantic strategy requires" in captured.err


def test_cli_chat_single_message(capsys: pytest.CaptureFixture) -> None:
    """Test single turn chat command."""
    ret = main(["chat", "--message", "Hi there, I need a recommendation", "--user-id", "1"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "BOOKSTORE ASSISTANT" in captured.out
    assert "Intent:" in captured.out


def test_cli_chat_single_message_json(capsys: pytest.CaptureFixture) -> None:
    """Test single turn chat returning JSON."""
    ret = main(["chat", "--message", "Tell me about science fiction books", "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "intent" in data
    assert "response" in data


def test_cli_chat_missing_message(capsys: pytest.CaptureFixture) -> None:
    """Test chat error when message is omitted."""
    ret = main(["chat"])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Error: Either --message or --interactive" in captured.err


def test_cli_chat_interactive(capsys: pytest.CaptureFixture) -> None:
    """Test interactive chat session flow."""
    inputs = ["clear", "Can you recommend a fantasy book?", "quit"]
    with patch("builtins.input", side_effect=inputs):
        ret = main(["chat", "--interactive"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "BOOKSTORE CONVERSATIONAL ASSISTANT" in captured.out
    assert "Memory cleared." in captured.out
    assert "Goodbye!" in captured.out


def test_cli_benchmark(capsys: pytest.CaptureFixture) -> None:
    """Test benchmark command."""
    ret = main(["benchmark", "--top-k", "3"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "RECOMMENDER ENGINE BENCHMARK" in captured.out
    assert "Hit Rate@3:" in captured.out


def test_cli_benchmark_json(capsys: pytest.CaptureFixture) -> None:
    """Test benchmark command with JSON output."""
    ret = main(["benchmark", "--top-k", "3", "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["benchmark"] == "Hybrid Retrieval Engine"
    assert "hit_rate_pct" in data
    assert "mean_latency_ms" in data
