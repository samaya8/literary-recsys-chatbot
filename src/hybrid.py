"""Hybrid Recommendation Fusion & Explainability Engine.

Fuses:
1. Collaborative Filtering (Matrix Factorization SVD)
2. Semantic Content Discovery (Multi-Field LSI & TF-IDF)

Provides:
- Dynamic adaptive weight allocation based on user profile maturity.
- Human-readable transparent recommendation explanations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.collaborative import CollaborativeRecommender
from src.semantic_recommender import SemanticBookRecommender


class HybridBookRecommender:
    """Combines collaborative filtering and semantic content discovery with explanations."""

    def __init__(
        self,
        collaborative_engine: Optional[CollaborativeRecommender] = None,
        semantic_engine: Optional[SemanticBookRecommender] = None,
        books_df: Optional[pd.DataFrame] = None,
        ratings_df: Optional[pd.DataFrame] = None,
    ):
        self.collaborative = collaborative_engine or CollaborativeRecommender(
            books_df=books_df,
            ratings_df=ratings_df,
        )
        self.semantic = semantic_engine or SemanticBookRecommender(
            df=books_df or self.collaborative.books_df,
        )
        self.books_df = self.semantic.df

    @property
    def cf(self) -> CollaborativeRecommender:
        """Alias for collaborative engine."""
        return self.collaborative

    def _determine_adaptive_alpha(self, user_id: Optional[int], manual_alpha: Optional[float] = None) -> float:
        """Dynamically determine collaborative vs semantic balance based on user profile maturity."""
        if manual_alpha is not None:
            return float(np.clip(manual_alpha, 0.0, 1.0))

        if user_id is None or user_id not in self.collaborative.user_to_idx:
            # Cold-start user: 100% semantic
            return 0.0

        user_rating_count = len(self.collaborative.ratings_df[self.collaborative.ratings_df["user_id"] == user_id])
        if user_rating_count >= 5:
            return 0.65  # Experienced reader: heavy collaborative weight
        elif user_rating_count >= 2:
            return 0.40  # Emerging reader: balanced
        else:
            return 0.20

    def recommend(
        self,
        user_id: Optional[int] = None,
        seed_title: Optional[str] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
        alpha: Optional[float] = None,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Compute hybrid personalized recommendations with transparent explanations."""
        eff_alpha = self._determine_adaptive_alpha(user_id, alpha)

        # 1. Obtain Semantic/Content Scores
        content_scores: Dict[int, float] = {}
        matched_seed = ""

        if seed_title:
            matched_seed = self.semantic.find_closest_title(seed_title)
            if matched_seed:
                sem_recs = self.semantic.recommend_by_title(
                    matched_seed,
                    top_k=len(self.books_df),
                    lambda_mmr=1.0,
                )
                for r in sem_recs:
                    content_scores[r["book_id"]] = float(r["similarity_score"])
        elif query_text:
            sem_recs = self.semantic.discover_by_query(
                query_text,
                top_k=len(self.books_df),
                lambda_mmr=1.0,
            )
            for r in sem_recs:
                content_scores[r["book_id"]] = float(r["similarity_score"])
        else:
            # No content query: uniform neutral score
            for idx in self.books_df["index"].values:
                content_scores[int(idx)] = 0.5

        # 2. Exclude books user has already rated if user is known
        already_rated = set()
        if user_id is not None and not self.collaborative.ratings_df.empty:
            user_ratings = self.collaborative.ratings_df[self.collaborative.ratings_df["user_id"] == user_id]
            already_rated = set(user_ratings["book_id"].unique())

        # Exclude seed title itself from recommendations
        if matched_seed:
            seed_row = self.books_df[self.books_df["title"] == matched_seed]
            if not seed_row.empty:
                already_rated.add(int(seed_row.iloc[0]["index"]))

        # 3. Fuse Scores for Candidates
        candidates = []
        for _, row in self.books_df.iterrows():
            b_id = int(row.get("index", 0))
            if b_id in already_rated:
                continue

            if category_filter and str(row.get("categories", "")).lower() != category_filter.lower():
                continue

            # Collaborative component (normalized predicted rating in [0, 1])
            cf_pred = self.collaborative.predict_rating(user_id or -1, b_id)
            s_cf = cf_pred / 5.0

            # Content component
            s_content = content_scores.get(b_id, 0.2)

            # Composite hybrid score
            s_hybrid = eff_alpha * s_cf + (1.0 - eff_alpha) * s_content

            explanation = self._generate_explanation(
                title=str(row["title"]),
                category=str(row.get("categories", "General")),
                cf_pred=cf_pred,
                s_content=s_content,
                eff_alpha=eff_alpha,
                matched_seed=matched_seed,
                query_text=query_text,
            )

            candidates.append({
                "book_id": b_id,
                "title": str(row["title"]),
                "author": str(row.get("authors", "Unknown")),
                "category": str(row.get("categories", "General")),
                "description": str(row.get("description", "")),
                "hybrid_score": round(float(s_hybrid), 4),
                "cf_predicted_rating": cf_pred,
                "content_similarity": round(float(s_content), 4),
                "explanation": explanation,
            })

        # Rank candidates
        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top_recs = candidates[:top_k]

        for rank, item in enumerate(top_recs, 1):
            item["rank"] = rank

        return top_recs

    def _generate_explanation(
        self,
        title: str,
        category: str,
        cf_pred: float,
        s_content: float,
        eff_alpha: float,
        matched_seed: Optional[str] = None,
        query_text: Optional[str] = None,
    ) -> str:
        """Construct explainable rationale for recommendation."""
        reasons = []

        if eff_alpha >= 0.35 and cf_pred >= 4.2:
            reasons.append(f"readers with similar tastes rated this {cf_pred}★")

        if matched_seed and s_content >= 0.4:
            reasons.append(f"shares strong thematic elements with '{matched_seed}'")
        elif query_text and s_content >= 0.4:
            reasons.append(f"closely aligns with your search for '{query_text}'")

        if not reasons:
            reasons.append(f"stands out as a highly acclaimed {category} title")

        return f"Recommended because {', and '.join(reasons)}."
