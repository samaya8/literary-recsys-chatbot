"""Content-based book recommendation engine using TF-IDF and Cosine Similarity."""

from typing import List, Tuple
import difflib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BookRecommender:
    """Computes book recommendations from content descriptions and metadata."""

    def __init__(self, data_path: str = None, df: pd.DataFrame = None):
        if df is not None:
            self.df = df.copy()
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._prepare_engine()

    def _prepare_engine(self):
        # Combine relevant metadata features into a unified text representation
        features = []
        for _, row in self.df.iterrows():
            parts = [
                str(row.get("title", "")),
                str(row.get("authors", "")),
                str(row.get("categories", "")),
                str(row.get("description", "")),
            ]
            features.append(" ".join(parts))
        self.df["combined_features"] = features
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combined_features"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def find_closest_title(self, query: str) -> str:
        """Find the closest matching title in catalog using string similarity."""
        all_titles = self.df["title"].astype(str).tolist()
        matches = difflib.get_close_matches(query, all_titles, n=1, cutoff=0.3)
        return matches[0] if matches else ""

    def recommend(self, title_query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return Top-K recommended titles and their similarity scores.
        
        Args:
            title_query: Query book title.
            top_k: Number of recommendations to return.
            
        Returns:
            List of tuples (recommended_title, similarity_score).
        """
        matched_title = self.find_closest_title(title_query)
        if not matched_title:
            return []

        idx = self.df[self.df["title"] == matched_title].index[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Exclude query book itself
        recommendations = []
        for i, score in sim_scores:
            if i != idx and len(recommendations) < top_k:
                recommendations.append((self.df.iloc[i]["title"], float(score)))

        return recommendations
