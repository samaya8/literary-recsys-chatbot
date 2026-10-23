"""Semantic Vector Discovery & Content-Based Recommendation Engine.

Provides:
1. Multi-Field Weighted TF-IDF with sublinear term frequency scaling.
2. Latent Semantic Indexing (LSI via TruncatedSVD) for dense topic representation.
3. Maximal Marginal Relevance (MMR) for diverse, non-redundant recommendation sets.
4. Natural Language query discovery (free-form semantic search without exact title knowledge).
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BOOKS_PATH = PROJECT_DIR / "data" / "sample_books.csv"


class SemanticBookRecommender:
    """Computes semantic book recommendations using multi-field LSI and MMR diversification."""

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        data_path: Optional[Union[str, Path]] = None,
        n_components: int = 8,
        random_state: int = 42,
    ):
        if df is not None:
            self.df = df.copy()
        elif data_path:
            self.df = pd.read_csv(data_path)
        elif DEFAULT_BOOKS_PATH.exists():
            self.df = pd.read_csv(DEFAULT_BOOKS_PATH)
        else:
            raise ValueError("No catalog data source provided.")

        self.n_components = n_components
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            sublinear_tf=True,
            ngram_range=(1, 2),
            max_features=2500,
        )

        self._build_semantic_index()

    def _build_semantic_index(self):
        """Construct weighted multi-field documents and compute dense semantic vectors."""
        weighted_docs = []
        for _, row in self.df.iterrows():
            title = str(row.get("title", ""))
            author = str(row.get("authors", ""))
            category = str(row.get("categories", ""))
            desc = str(row.get("description", ""))

            # Weighted field replication: title 3x, category 2x, author 2x, description 1x
            doc_parts = [
                title, title, title,
                category, category,
                author, author,
                desc,
            ]
            weighted_docs.append(" ".join(doc_parts))

        self.df["semantic_document"] = weighted_docs

        # 1. Sparse TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["semantic_document"])

        # 2. Dense Latent Semantic Indexing (LSI)
        num_docs = len(self.df)
        actual_k = min(self.n_components, max(1, num_docs - 1))
        self.svd = TruncatedSVD(n_components=actual_k, random_state=self.random_state)
        self.dense_embeddings = self.svd.fit_transform(self.tfidf_matrix)

        # Normalize dense vectors to unit sphere
        norms = np.linalg.norm(self.dense_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.dense_embeddings = self.dense_embeddings / norms

        # Item-item cosine similarity matrix
        self.similarity_matrix = cosine_similarity(self.dense_embeddings)

    def find_closest_title(self, query: str) -> str:
        """Fuzzy match query string to the closest catalog title."""
        all_titles = self.df["title"].astype(str).tolist()
        matches = difflib.get_close_matches(query, all_titles, n=1, cutoff=0.3)
        return matches[0] if matches else ""

    def _apply_mmr(
        self,
        query_sims: np.ndarray,
        candidate_indices: List[int],
        top_k: int = 5,
        lambda_param: float = 0.7,
    ) -> List[int]:
        """Maximal Marginal Relevance selection balancing similarity and diversity."""
        if len(candidate_indices) <= top_k:
            return candidate_indices

        selected: List[int] = []
        remaining = list(candidate_indices)

        while len(selected) < top_k and remaining:
            best_idx = -1
            best_score = -float("inf")

            for cand in remaining:
                relevance = query_sims[cand]
                if not selected:
                    redundancy = 0.0
                else:
                    redundancy = max(self.similarity_matrix[cand, s] for s in selected)

                mmr_score = lambda_param * relevance - (1.0 - lambda_param) * redundancy
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = cand

            if best_idx != -1:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break

        return selected

    def recommend_by_title(
        self,
        title_query: str,
        top_k: int = 5,
        lambda_mmr: float = 0.7,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Recommend Top-K books matching seed title with MMR diversification."""
        matched_title = self.find_closest_title(title_query)
        if not matched_title:
            return []

        query_idx = self.df[self.df["title"] == matched_title].index[0]
        query_sims = self.similarity_matrix[query_idx]

        candidates = [i for i in range(len(self.df)) if i != query_idx]

        if category_filter:
            candidates = [
                i for i in candidates
                if str(self.df.iloc[i].get("categories", "")).lower() == category_filter.lower()
            ]

        # Sort candidate indices by raw relevance
        candidates.sort(key=lambda i: query_sims[i], reverse=True)

        # Select via MMR
        selected_indices = self._apply_mmr(query_sims, candidates, top_k=top_k, lambda_param=lambda_mmr)

        results = []
        for rank, idx in enumerate(selected_indices, 1):
            row = self.df.iloc[idx]
            results.append({
                "rank": rank,
                "book_id": int(row.get("index", idx)),
                "title": str(row["title"]),
                "author": str(row.get("authors", "Unknown")),
                "category": str(row.get("categories", "General")),
                "similarity_score": round(float(query_sims[idx]), 4),
                "matched_seed": matched_title,
            })

        return results

    def discover_by_query(
        self,
        query_text: str,
        top_k: int = 5,
        lambda_mmr: float = 0.7,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Discover books from free-form natural language description."""
        if not query_text.strip():
            return []

        # Transform query into LSI dense vector
        q_tfidf = self.vectorizer.transform([query_text])
        q_dense = self.svd.transform(q_tfidf)
        norm = np.linalg.norm(q_dense)
        if norm > 0:
            q_dense = q_dense / norm

        query_sims = cosine_similarity(q_dense, self.dense_embeddings)[0]

        candidates = list(range(len(self.df)))
        if category_filter:
            candidates = [
                i for i in candidates
                if str(self.df.iloc[i].get("categories", "")).lower() == category_filter.lower()
            ]

        candidates.sort(key=lambda i: query_sims[i], reverse=True)
        selected_indices = self._apply_mmr(query_sims, candidates, top_k=top_k, lambda_param=lambda_mmr)

        results = []
        for rank, idx in enumerate(selected_indices, 1):
            row = self.df.iloc[idx]
            results.append({
                "rank": rank,
                "book_id": int(row.get("index", idx)),
                "title": str(row["title"]),
                "author": str(row.get("authors", "Unknown")),
                "category": str(row.get("categories", "General")),
                "description": str(row.get("description", "")),
                "similarity_score": round(float(query_sims[idx]), 4),
            })

        return results
