import numpy as np
import pandas as pd
import pytest

from app.thresholds import build_score_matrix, compute, compute_scores, optimize


class TestThresholds:

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

    def test_compute_default(self) -> None:
        """Return default threshold if not enough training samples are given"""
        default_ti = compute('test', [0.4, 0.5, 0.7], [0.1, 0.1])
        assert default_ti.suggested_threshold == 0.5

    def test_compute_optimized(self) -> None:
        ti = compute('optimized threshold', [0.8] * 20, [0.1] * 20)
        assert ti.suggested_threshold == 0.5

    def test_build_score_matrix(self) -> None:
        train_probs = [0.7, 0.8, 0.9, 0.99]
        false_probs = [0.2, 0.4, 0.7]
        scores = build_score_matrix(train_probs, false_probs)
        thresholds = [round(elem, 2) for elem in np.arange(0.05, 1, 0.05)]

        assert np.equal(scores.index.values, thresholds).all()
        assert (scores <= 1.0).all().all() and (scores >= 0.0).all().all()
        assert (scores.loc[scores.index.values >= 0.75].precision == 1.0).all()

    def test_optimize(self) -> None:
        scores = pd.DataFrame([(0.3, 0.2, 0.4), (0.7, 0.8, 0.6),
                               (0.8, 0.7, 0.9)],
                              columns=['f1', 'precision', 'recall'],
                              index=[0.2, 0.4, 0.7])
        assert optimize(scores) == 0.7
        assert optimize(scores, min_prec=0.8) == 0.4
        assert optimize(scores, min_prec=0.8, min_rec=0.5) == 0.4
        # if minimum precision cannot be achieved return default threshold
        assert optimize(scores, min_prec=1.1) == 0.5

    def test_optimize__multimax(self) -> None:
        scores = pd.DataFrame([(0.7, 0.3, 0.9), (0.7, 0.6, 0.85),
                               (0.7, 0.4, 0.87), (0.7, 0.7, 0.85),
                               (0.7, 0.9, 0.85)],
                              columns=['f1', 'precision', 'recall'],
                              index=[0.3, 0.4, 0.5, 0.6, 0.7])
        assert optimize(scores) == 0.5
        assert optimize(scores, 0.6) == 0.6
        assert optimize(scores, 0.7) == 0.7
        assert optimize(scores, 0.9) == 0.7
