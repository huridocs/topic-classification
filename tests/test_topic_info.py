import pandas as pd

from app.topic_info import TopicInfo

SCORES = pd.DataFrame([(0.3, 0.2, 0.4), (0.7, 0.8, 0.6), (0.8, 0.7, 0.9)],
                      index=[0.2, 0.4, 0.7],
                      columns=['f1', 'precision', 'recall'])


class TestTopicInfo:

    def setup(self) -> None:
        self.topic_info = TopicInfo('test')
        self.topic_info.scores = SCORES

    def test_get_quality(self) -> None:
        self.topic_info.suggested_threshold = 0.7
        assert self.topic_info.get_quality() == 0.8
        self.topic_info.suggested_threshold = 0.2
        assert self.topic_info.get_quality() == 0.3

    def test_closest_threshold(self) -> None:
        assert self.topic_info.closest_threshold(0.25) == 0.2
        assert self.topic_info.closest_threshold(0.35) == 0.4
        assert self.topic_info.closest_threshold(0.3) == 0.2

    def test_get_confidence_at_probability_empty(self) -> None:
        self.topic_info.scores = pd.DataFrame([])
        assert self.topic_info.get_confidence_at_probability(0.1) == 0.3
        assert self.topic_info.get_confidence_at_probability(0.5) == 0.3
        assert self.topic_info.get_confidence_at_probability(0.9) == 0.3

    def test_get_confidence_at_probability(self) -> None:
        assert self.topic_info.get_confidence_at_probability(0.4) == 0.8
        assert self.topic_info.get_confidence_at_probability(0.35) == 0.8
        assert self.topic_info.get_confidence_at_probability(0.3) == 0.2
