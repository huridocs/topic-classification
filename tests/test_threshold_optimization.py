import pytest

import numpy as np
import numpy.testing as npt
import pandas as pd

from app.threshold_optimization import ComputeThresholds
from app.threshold_optimization import compute_scores
from app.threshold_optimization import build_score_matrix
from app.threshold_optimization import optimize_threshold


class Test_thresholdOptimization:

    def test_compute_scores(self) -> None:
        train_probs = [0.7, 0.8, 0.9, 0.99]
        false_probs = [0.2, 0.4]
        f1, precision, recall = compute_scores(train_probs, false_probs, 0.5)
        assert f1 == 1
        assert precision == 1
        assert recall == 1

        f1, precision, recall = compute_scores(train_probs, false_probs, 0.85)
        assert f1 == pytest.approx(2 / 3.0)
        assert precision == 1.0
        assert recall == 0.5

        f1, precision, recall = compute_scores(train_probs, false_probs, 0.3)
        assert f1 == pytest.approx(8 / 9.0)
        assert precision == 0.8
        assert recall == 1.0

    def test_compute_scores_division_by_zero(self) -> None:
        """ ensure that division by zero returns 0.0 instead of NaN"""
        f1, precision, recall = compute_scores([], [0.4], 0.5)
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_compute_thresholds_default(self) -> None:
        """Return default threshold if not enough training samples are given"""
        default_ti = ComputeThresholds('test', [0.4, 0.5, 0.7], [0.1, 0.1])
        assert default_ti.suggested_threshold == 0.5

    def test_compute_thresholds_optimized(self) -> None:
        ti = ComputeThresholds('optimized threshold', [0.8] * 20, [0.1] * 20)
        assert ti.suggested_threshold == 0.5

    def test_build_score_matrix(self) -> None:
        train_probs = [0.7, 0.8, 0.9, 0.99]
        false_probs = [0.2, 0.4, 0.7]
        scores = build_score_matrix(train_probs, false_probs)
        thresholds = [round(elem, 2) for elem in np.arange(0.05, 1, 0.05)]

        assert np.equal(scores.threshold.values, thresholds).all()
        assert (scores <= 1.0).all().all() and (scores >= 0.0).all().all()
        assert (scores[scores['threshold'] >= 0.75].precision == 1.0).all()

    def test_optimize_threshold(self) -> None:
        scores = pd.DataFrame(
            [(0.2, 0.3, 0.2, 0.4), (0.4, 0.7, 0.8, 0.6), (0.7, 0.8, 0.7, 0.9)],
            columns=['threshold', 'f1', 'precision', 'recall'])
        assert optimize_threshold(scores) == 0.7
        assert optimize_threshold(scores, min_prec=0.8) == 0.4
        # if minimum precision cannot be achieved return default threshold
        assert optimize_threshold(scores, min_prec=1.1) == 0.5

    def test_optimize_threshold_multimax(self) -> None:
        scores = pd.DataFrame([(0.3, 0.7, 0.3, 0.9),
                               (0.4, 0.7, 0.6, 0.7),
                               (0.5, 0.7, 0.4, 0.8),
                               (0.6, 0.7, 0.7, 0.7),
                               (0.7, 0.7, 0.9, 0.5)],
                              columns=['threshold', 'f1',
                                       'precision', 'recall'])
        assert optimize_threshold(scores) == 0.5
        assert optimize_threshold(scores, 0.6) == 0.6
        assert optimize_threshold(scores, 0.7) == 0.7
        assert optimize_threshold(scores, 0.9) == 0.7
