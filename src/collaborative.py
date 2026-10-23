"""Collaborative Filtering and Matrix Factorization Recommender Engine.

Provides:
1. Matrix Factorization via Latent Factor SVD Decomposition with baseline biases:
   r_hat(u, i) = mu + b_u + b_i + p_u^T q_i
2. User-User and Item-Item Neighborhood Retrieval over latent embeddings.
3. Bayesian Weighted Rating Fallback for Cold-Start Users and Unseen Items.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BOOKS_PATH = PROJECT_DIR / "data" / "sample_books.csv"
DEFAULT_RATINGS_PATH = PROJECT_DIR / "data" / "sample_ratings.csv"


class CollaborativeRecommender:
    """Matrix Factorization & Collaborative Filtering recommender system."""

    def __init__(
        self,
        ratings_df: Optional[pd.DataFrame] = None,
        books_df: Optional[pd.DataFrame] = None,
        n_factors: int = 4,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.random_state = random_state

        # Load books catalog
        if books_df is not None:
            self.books_df = books_df.copy()
        elif DEFAULT_BOOKS_PATH.exists():
            self.books_df = pd.read_csv(DEFAULT_BOOKS_PATH)
        else:
            self.books_df = pd.DataFrame(columns=["index", "title", "authors", "categories", "description"])

        # Load ratings
        if ratings_df is not None:
            self.ratings_df = ratings_df.copy()
        elif DEFAULT_RATINGS_PATH.exists():
            self.ratings_df = pd.read_csv(DEFAULT_RATINGS_PATH)
        else:
            self.ratings_df = pd.DataFrame(columns=["user_id", "book_id", "rating"])

        # Ensure title column is available in ratings_df for evaluation & diagnostics
        if not self.ratings_df.empty and "title" not in self.ratings_df.columns and "title" in self.books_df.columns:
            idx_col = "index" if "index" in self.books_df.columns else self.books_df.index
            id_to_title = dict(zip(self.books_df[idx_col], self.books_df["title"]))
            self.ratings_df["title"] = self.ratings_df["book_id"].map(id_to_title)

        self.fitted = False
        self.global_mean: float = 3.5
        self.user_ids: List[int] = []
        self.item_ids: List[int] = []
        self.user_to_idx: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.user_biases: np.ndarray = np.array([])
        self.item_biases: np.ndarray = np.array([])
        self.user_factors: np.ndarray = np.array([])
        self.item_factors: np.ndarray = np.array([])
        self.item_similarity: np.ndarray = np.array([])
        self.user_similarity: np.ndarray = np.array([])

        if not self.ratings_df.empty:
            self.fit()

    def fit(self) -> CollaborativeRecommender:
        """Fit Matrix Factorization SVD model on user-item ratings."""
        if self.ratings_df.empty:
            self.fitted = True
            return self

        # Extract unique users and items
        self.user_ids = sorted(self.ratings_df["user_id"].unique().tolist())
        self.item_ids = sorted(self.ratings_df["book_id"].unique().tolist())

        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item_to_idx = {iid: idx for idx, iid in enumerate(self.item_ids)}

        num_users = len(self.user_ids)
        num_items = len(self.item_ids)

        # Global average rating
        self.global_mean = float(self.ratings_df["rating"].mean())

        # Construct interaction matrix
        R = np.full((num_users, num_items), np.nan, dtype=np.float32)
        for _, row in self.ratings_df.iterrows():
            u_idx = self.user_to_idx.get(row["user_id"])
            i_idx = self.item_to_idx.get(row["book_id"])
            if u_idx is not None and i_idx is not None:
                R[u_idx, i_idx] = float(row["rating"])

        # Compute user and item biases with mean imputation
        user_means = np.nanmean(R, axis=1)
        self.user_biases = np.where(np.isnan(user_means), 0.0, user_means - self.global_mean)

        item_means = np.nanmean(R, axis=0)
        self.item_biases = np.where(np.isnan(item_means), 0.0, item_means - self.global_mean)

        # Construct centered residual matrix for SVD
        R_imputed = np.copy(R)
        for u in range(num_users):
            for i in range(num_items):
                if np.isnan(R_imputed[u, i]):
                    R_imputed[u, i] = self.global_mean + self.user_biases[u] + self.item_biases[i]

        R_centered = R_imputed - self.global_mean

        # Latent Factor Decomposition
        actual_factors = min(self.n_factors, max(1, min(num_users, num_items) - 1))
        svd = TruncatedSVD(n_components=actual_factors, random_state=self.random_state)
        self.user_factors = svd.fit_transform(R_centered)
        self.item_factors = svd.components_.T

        # Normalize factors for cosine similarity
        u_norm = np.linalg.norm(self.user_factors, axis=1, keepdims=True)
        u_norm = np.where(u_norm == 0, 1.0, u_norm)
        u_normed = self.user_factors / u_norm

        i_norm = np.linalg.norm(self.item_factors, axis=1, keepdims=True)
        i_norm = np.where(i_norm == 0, 1.0, i_norm)
        i_normed = self.item_factors / i_norm

        self.user_similarity = cosine_similarity(u_normed)
        self.item_similarity = cosine_similarity(i_normed)
        self.fitted = True
        return self

    def predict_rating(self, user_id: int, book_id: int) -> float:
        """Predict expected rating for (user_id, book_id) in range [1.0, 5.0]."""
        if not self.fitted:
            self.fit()

        u_idx = self.user_to_idx.get(user_id)
        i_idx = self.item_to_idx.get(book_id)

        # Cold start handling
        if u_idx is None and i_idx is None:
            return round(self.global_mean, 2)
        elif u_idx is None:
            # New user: item average
            b_i = self.item_biases[i_idx]
            return round(float(np.clip(self.global_mean + b_i, 1.0, 5.0)), 2)
        elif i_idx is None:
            # Unseen item: user average
            b_u = self.user_biases[u_idx]
            return round(float(np.clip(self.global_mean + b_u, 1.0, 5.0)), 2)

        # Full SVD bias prediction
        pred = (
            self.global_mean
            + self.user_biases[u_idx]
            + self.item_biases[i_idx]
            + np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
        )
        return round(float(np.clip(pred, 1.0, 5.0)), 2)

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 5,
        exclude_rated: bool = True,
    ) -> List[Dict[str, Any]]:
        """Recommend Top-K items personalized for a user using Matrix Factorization."""
        if not self.fitted:
            self.fit()

        if user_id not in self.user_to_idx:
            return self.cold_start_recommendations(top_k=top_k)

        u_idx = self.user_to_idx[user_id]
        already_rated = set()
        if exclude_rated:
            user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
            already_rated = set(user_ratings["book_id"].unique())

        candidates = []
        for i_idx, b_id in enumerate(self.item_ids):
            if exclude_rated and b_id in already_rated:
                continue
            pred_score = self.predict_rating(user_id, b_id)
            title = self._get_book_title(b_id)
            category = self._get_book_category(b_id)
            author = self._get_book_author(b_id)

            candidates.append({
                "book_id": int(b_id),
                "title": title,
                "author": author,
                "category": category,
                "predicted_rating": pred_score,
                "score": round(pred_score / 5.0, 4),  # Normalized similarity score [0, 1]
            })

        candidates.sort(key=lambda x: x["predicted_rating"], reverse=True)
        return candidates[:top_k]

    def get_similar_items(self, book_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find Top-K collaborative item neighbors (readers who liked X also liked Y)."""
        if not self.fitted:
            self.fit()

        i_idx = self.item_to_idx.get(book_id)
        if i_idx is None:
            return []

        sim_scores = self.item_similarity[i_idx]
        sorted_indices = np.argsort(sim_scores)[::-1]

        results = []
        for rank_idx in sorted_indices:
            if rank_idx == i_idx:
                continue
            other_id = self.item_ids[rank_idx]
            sim = float(sim_scores[rank_idx])
            results.append({
                "book_id": int(other_id),
                "title": self._get_book_title(other_id),
                "category": self._get_book_category(other_id),
                "author": self._get_book_author(other_id),
                "similarity": round(sim, 4),
            })
            if len(results) >= top_k:
                break

        return results

    def cold_start_recommendations(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Bayesian weighted average popularity fallback for cold-start users."""
        if self.ratings_df.empty:
            # Fallback to book order in catalog
            return [
                {
                    "book_id": int(row.get("index", idx)),
                    "title": str(row.get("title", f"Book #{idx}")),
                    "author": str(row.get("authors", "Unknown")),
                    "category": str(row.get("categories", "General")),
                    "predicted_rating": 4.0,
                    "score": 0.8,
                }
                for idx, row in self.books_df.head(top_k).iterrows()
            ]

        # Compute Bayesian weighted rating per item
        stats = self.ratings_df.groupby("book_id")["rating"].agg(["count", "mean"]).reset_index()
        C = self.global_mean
        m = 2.0  # Minimum votes weight threshold

        stats["weighted_score"] = (stats["count"] / (stats["count"] + m)) * stats["mean"] + (m / (stats["count"] + m)) * C
        stats = stats.sort_values(by="weighted_score", ascending=False)

        recs = []
        for _, row in stats.head(top_k).iterrows():
            b_id = int(row["book_id"])
            score = float(row["weighted_score"])
            recs.append({
                "book_id": b_id,
                "title": self._get_book_title(b_id),
                "author": self._get_book_author(b_id),
                "category": self._get_book_category(b_id),
                "predicted_rating": round(score, 2),
                "score": round(score / 5.0, 4),
            })

        return recs

    def _get_book_title(self, book_id: int) -> str:
        """Resolve title from books catalog."""
        match = self.books_df[self.books_df["index"] == book_id]
        if not match.empty:
            return str(match.iloc[0]["title"])
        return f"Book #{book_id}"

    def _get_book_category(self, book_id: int) -> str:
        """Resolve category from books catalog."""
        match = self.books_df[self.books_df["index"] == book_id]
        if not match.empty:
            return str(match.iloc[0].get("categories", "General"))
        return "General"

    def _get_book_author(self, book_id: int) -> str:
        """Resolve author from books catalog."""
        match = self.books_df[self.books_df["index"] == book_id]
        if not match.empty:
            return str(match.iloc[0].get("authors", "Unknown"))
        return "Unknown"
