"""Unit tests for the automated benchmark evaluation harness."""

from __future__ import annotations

import math
import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evals.runner import (
    compute_dcg,
    prepare_leave_one_out_split,
    run_full_benchmark,
    export_markdown_report,
)


def test_compute_dcg() -> None:
    """Test DCG calculation for ranks."""
    # Rank 1: 1 / log2(2) = 1.0
    assert pytest.approx(compute_dcg(1), 1e-4) == 1.0
    # Rank 2: 1 / log2(3) ≈ 0.6309
    assert pytest.approx(compute_dcg(2), 1e-4) == 1.0 / math.log2(3)


def test_prepare_leave_one_out_split() -> None:
    """Test leave-one-out data splitting."""
    dummy_ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "book_id": [0, 1, 2, 3, 4],
            "rating": [5.0, 4.5, 3.0, 4.0, 5.0],
            "timestamp": [1, 2, 3, 4, 5],
        }
    )
    train_df, test_records = prepare_leave_one_out_split(dummy_ratings)

    assert len(test_records) == 2  # Both users have at least 2 ratings >= 4.0
    assert len(train_df) == 3  # 5 - 2 held out = 3 train rows
    assert test_records[0]["user_id"] == 1
    assert test_records[0]["book_id"] == 1  # Last rating >= 4.0 for user 1


def test_run_full_benchmark() -> None:
    """Test running full benchmark on sample data."""
    results = run_full_benchmark(k=3)
    assert "results" in results
    assert "collaborative" in results["results"]
    assert "semantic" in results["results"]
    assert "hybrid" in results["results"]

    hyb_res = results["results"]["hybrid"]
    assert hyb_res["k"] == 3
    assert "hit_rate" in hyb_res
    assert "ndcg" in hyb_res
    assert "precision" in hyb_res
    assert "recall" in hyb_res
    assert "catalog_coverage_pct" in hyb_res
    assert hyb_res["hit_rate"] >= 0.0


def test_export_markdown_report(tmp_path: Path) -> None:
    """Test exporting markdown report to file."""
    data = run_full_benchmark(k=3)
    out_file = tmp_path / "test_report.md"
    report_text = export_markdown_report(data, out_file)

    assert out_file.exists()
    assert "# Recommender Engine Evaluation & Benchmark Report" in report_text
    assert "| Metric | Collaborative (SVD) | Semantic (LSI+MMR) | Hybrid Fusion |" in report_text
