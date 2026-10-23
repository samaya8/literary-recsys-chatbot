"""Enterprise Command-Line Interface (CLI) for Hybrid Book Recommendation & Chatbot.

Subcommands:
- recommend: Generate personalized book recommendations via hybrid, collaborative, or semantic pipelines.
- chat: Interactive or single-turn conversational bookstore agent session with memory.
- benchmark: Quick offline retrieval accuracy and latency diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.collaborative import CollaborativeRecommender
from src.conversational_assistant import ConversationalBookstoreAssistant
from src.hybrid import HybridBookRecommender
from src.semantic_recommender import SemanticBookRecommender


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser for the bookstore CLI."""
    parser = argparse.ArgumentParser(
        prog="book-recommender-cli",
        description="Enterprise Hybrid Book Recommendation & Conversational Memory Intelligence CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. recommend
    p_rec = subparsers.add_parser("recommend", help="Generate book recommendations")
    p_rec.add_argument("--title", type=str, help="Seed book title to find similar books for")
    p_rec.add_argument("--query", type=str, help="Natural language search description (e.g. 'space travel dystopian')")
    p_rec.add_argument("--user-id", type=int, help="Target user ID for personalized collaborative filtering")
    p_rec.add_argument(
        "--strategy",
        type=str,
        choices=["hybrid", "collaborative", "semantic"],
        default="hybrid",
        help="Recommendation engine strategy",
    )
    p_rec.add_argument("--genre", type=str, help="Filter results by category (e.g. 'Fantasy', 'Science Fiction')")
    p_rec.add_argument("--alpha", type=float, help="Manual collaborative vs semantic weight [0.0 - 1.0]")
    p_rec.add_argument("--top-k", type=int, default=5, help="Number of recommendations to return")
    p_rec.add_argument("--explain", action="store_true", help="Include human-readable recommendation explanations")
    p_rec.add_argument("--json", action="store_true", help="Output results in JSON format")

    # 2. chat
    p_chat = subparsers.add_parser("chat", help="Chat with the conversational bookstore assistant")
    p_chat.add_argument("--message", type=str, help="Single utterance message to send to the assistant")
    p_chat.add_argument("--user-id", type=int, help="Optional user ID for personalized profile context")
    p_chat.add_argument("--interactive", action="store_true", help="Launch interactive multi-turn terminal session")
    p_chat.add_argument("--json", action="store_true", help="Output response in JSON format")

    # 3. benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run offline retrieval accuracy and latency benchmark")
    p_bench.add_argument("--top-k", type=int, default=5, help="Top-K evaluation cutoff")
    p_bench.add_argument("--json", action="store_true", help="Output benchmark metrics as JSON")

    return parser


def handle_recommend(args: argparse.Namespace) -> int:
    """Execute recommend subcommand."""
    t0 = time.time()

    if args.strategy == "collaborative":
        if args.user_id is None and not args.title:
            print("Error: Collaborative strategy requires either --user-id or --title.", file=sys.stderr)
            return 1

        cf = CollaborativeRecommender()
        if args.user_id is not None:
            raw_recs = cf.recommend_for_user(user_id=args.user_id, top_k=args.top_k)
        else:
            # Item-item collaborative
            sem = SemanticBookRecommender()
            matched = sem.find_closest_title(args.title)
            if not matched:
                print(f"Error: Title not found: {args.title}", file=sys.stderr)
                return 1
            idx = sem.df[sem.df["title"] == matched].index[0]
            raw_recs = cf.get_similar_items(book_id=idx, top_k=args.top_k)

        results = raw_recs
        strategy_label = "COLLABORATIVE (SVD)"

    elif args.strategy == "semantic":
        sem = SemanticBookRecommender()
        if args.title:
            results = sem.recommend_by_title(
                title_query=args.title,
                top_k=args.top_k,
                category_filter=args.genre,
            )
        elif args.query:
            results = sem.discover_by_query(
                query_text=args.query,
                top_k=args.top_k,
                category_filter=args.genre,
            )
        else:
            print("Error: Semantic strategy requires either --title or --query.", file=sys.stderr)
            return 1
        strategy_label = "SEMANTIC (LSI & MMR)"

    else:  # Hybrid (Default)
        hybrid = HybridBookRecommender()
        results = hybrid.recommend(
            user_id=args.user_id,
            seed_title=args.title,
            query_text=args.query,
            top_k=args.top_k,
            alpha=args.alpha,
            category_filter=args.genre,
        )
        strategy_label = "HYBRID (CF + Semantic)"

    elapsed_ms = round((time.time() - t0) * 1000, 2)

    payload = {
        "strategy": strategy_label,
        "query": {
            "title": args.title,
            "query": args.query,
            "user_id": args.user_id,
            "genre_filter": args.genre,
            "top_k": args.top_k,
        },
        "latency_ms": elapsed_ms,
        "count": len(results),
        "recommendations": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("\n" + "=" * 65)
        print(f"BOOK RECOMMENDATIONS ({strategy_label})")
        print("=" * 65)
        print(f"Latency: {elapsed_ms} ms | Results: {len(results)}")
        print("-" * 65)
        for idx, item in enumerate(results, 1):
            score_val = item.get("hybrid_score", item.get("similarity_score", item.get("predicted_rating", 0.0)))
            print(f"#{idx:<2} [{score_val}] {item['title']} by {item.get('author', 'Unknown')}")
            print(f"    Category: {item.get('category', 'General')}")
            if args.explain and "explanation" in item:
                print(f"    Rationale: {item['explanation']}")
        print("=" * 65)

    return 0


def handle_chat(args: argparse.Namespace) -> int:
    """Execute chat subcommand."""
    assistant = ConversationalBookstoreAssistant()

    if args.interactive:
        print("\n" + "=" * 65)
        print("📚 BOOKSTORE CONVERSATIONAL ASSISTANT (Interactive Mode)")
        print("Type 'exit' or 'quit' to end session. Type 'clear' to reset memory.")
        print("=" * 65 + "\n")

        while True:
            try:
                user_msg = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_msg:
                continue
            if user_msg.lower() in ["exit", "quit"]:
                print("Assistant: Goodbye! Happy reading!")
                break
            if user_msg.lower() == "clear":
                assistant.memory.clear()
                print("Assistant: Memory cleared. How can I help you afresh?")
                continue

            res = assistant.chat(user_msg, user_id=args.user_id)
            print(f"\nAssistant: {res['response']}")
            if res["recommendations"]:
                for rec in res["recommendations"]:
                    print(f"  • {rec['title']} ({rec['category']}) - {rec.get('explanation', '')}")
            print()
        return 0

    if not args.message:
        print("Error: Either --message or --interactive must be provided for chat.", file=sys.stderr)
        return 1

    t0 = time.time()
    res = assistant.chat(args.message, user_id=args.user_id)
    elapsed_ms = round((time.time() - t0) * 1000, 2)
    res["latency_ms"] = elapsed_ms

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print("\n" + "=" * 65)
        print("BOOKSTORE ASSISTANT")
        print("=" * 65)
        print(f"Intent:  {res['intent']} (Confidence: {res['confidence']*100:.1f}%)")
        print(f"Reply:   {res['response']}")
        if res["recommendations"]:
            print("\nRecommendations:")
            for rec in res["recommendations"]:
                print(f"  • {rec['title']} ({rec['category']}) - {rec.get('explanation', '')}")
        print(f"Latency: {elapsed_ms} ms")
        print("=" * 65)

    return 0


def handle_benchmark(args: argparse.Namespace) -> int:
    """Execute benchmark subcommand."""
    from src.collaborative import CollaborativeRecommender
    from src.semantic_recommender import SemanticBookRecommender

    t0 = time.time()
    cf = CollaborativeRecommender()
    sem = SemanticBookRecommender()
    hybrid = HybridBookRecommender(collaborative_engine=cf, semantic_engine=sem)

    k = args.top_k
    hits = 0
    test_users = cf.user_ids[:5] if cf.user_ids else [1]
    latencies = []

    for u in test_users:
        t_start = time.time()
        recs = hybrid.recommend(user_id=u, top_k=k)
        latencies.append((time.time() - t_start) * 1000)
        if len(recs) > 0:
            hits += 1

    hit_rate = round((hits / max(1, len(test_users))) * 100.0, 1)
    mean_lat = round(float(sum(latencies) / max(1, len(latencies))), 2)

    payload = {
        "benchmark": "Hybrid Retrieval Engine",
        "evaluated_users": len(test_users),
        "top_k": k,
        "hit_rate_pct": hit_rate,
        "mean_latency_ms": mean_lat,
        "catalog_size": len(sem.df),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("\n" + "=" * 65)
        print("RECOMMENDER ENGINE BENCHMARK")
        print("=" * 65)
        print(f"Evaluated Users:     {len(test_users)}")
        print(f"Hit Rate@{k}:          {hit_rate}%")
        print(f"Mean Latency:        {mean_lat} ms")
        print(f"Catalog Size:        {len(sem.df)} titles")
        print("=" * 65)

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "recommend":
        return handle_recommend(args)
    elif args.subcommand == "chat":
        return handle_chat(args)
    elif args.subcommand == "benchmark":
        return handle_benchmark(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
