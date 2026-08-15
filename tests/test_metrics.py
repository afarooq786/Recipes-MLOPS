"""
Unit tests for evaluation/metrics.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import (
    grouped_precision_at_k,
    precision_at_k,
    safe_roc_auc,
)


class TestPrecisionAtK:
    def test_perfect_ranking(self):
        y_true = np.array([1, 1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        assert precision_at_k(y_true, y_score, k=3) == 1.0

    def test_worst_ranking(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_score = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        assert precision_at_k(y_true, y_score, k=3) == 0.0

    def test_k_larger_than_list_uses_full_list(self):
        y_true = np.array([1, 0])
        y_score = np.array([0.9, 0.1])
        # k=5 but only 2 items exist -> effective k is 2
        assert precision_at_k(y_true, y_score, k=5) == 0.5

    def test_empty_input_returns_none(self):
        assert precision_at_k(np.array([]), np.array([]), k=5) is None


class TestGroupedPrecisionAtK:
    def test_returns_macro_average_and_per_group(self):
        y_true = np.array([1, 0, 1, 0])
        y_score = np.array([0.9, 0.1, 0.8, 0.2])
        groups = np.array(["a", "a", "b", "b"])
        result = grouped_precision_at_k(y_true, y_score, groups, k=1)
        assert "macro_avg_precision_at_k" in result
        assert result["macro_avg_precision_at_k"] == pytest.approx(1.0)
        assert result["num_groups"] == 2

    def test_handles_single_group(self):
        y_true = np.array([1, 1, 0])
        y_score = np.array([0.9, 0.8, 0.1])
        groups = np.array(["a", "a", "a"])
        result = grouped_precision_at_k(y_true, y_score, groups, k=2)
        assert result["macro_avg_precision_at_k"] == pytest.approx(1.0)


class TestSafeRocAuc:
    def test_normal_case(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])
        auc = safe_roc_auc(y_true, y_score)
        assert auc is not None
        assert 0.0 <= auc <= 1.0

    def test_single_class_returns_none(self):
        # ROC-AUC is undefined with only one class present -- must not raise.
        y_true = np.array([1, 1, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])
        assert safe_roc_auc(y_true, y_score) is None

    def test_empty_returns_none(self):
        assert safe_roc_auc(np.array([]), np.array([])) is None
