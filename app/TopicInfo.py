import json
from typing import Any, Dict

import pandas as pd


class TopicInfo:
    """Collect thresholding and quality information about one topic."""

    def __init__(self, topic: str):
        self.topic = topic
        self.num_samples = 0
        self.thresholds: Dict[int, float] = {}
        self.recalls: Dict[int, float] = {}
        self.suggested_threshold = 0.5
        self.f1_quality_at_suggested = 0.0
        self.precision_at_suggested = 0.0
        self.scores = pd.DataFrame(columns=['threshold', 'f1',
                                            'precision', 'recall'])

    def to_json_dict(self) -> Dict[str, Any]:
        obj_dict = self.__dict__
        obj_dict['scores'] = self.scores.to_json()
        return obj_dict

    def load_json_dict(self, v: Dict[str, Any]) -> None:
        self.__dict__ = v
        self.scores = pd.read_json(self.scores)
        self.thresholds = {int(k): v for k, v in self.thresholds.items()}
        self.recalls = {int(k): v for k, v in self.recalls.items()}

    def get_quality(self, prob: float) -> float:
        quality = 0.0
        for precision_100, threshold in self.thresholds.items():
            if prob >= threshold:
                quality = precision_100 / 100.0
        return quality

    def __str__(self) -> str:
        res = [
            '%s has %d train, suggested quality %.02f@t=%.02f' %
            (self.topic, self.num_samples, self.f1_quality_at_suggested,
             self.suggested_threshold)
        ]
        for thres in self.thresholds.keys():
            res.append(
                '  t=%.02f -> %.02f@p%.01f' %
                (self.thresholds[thres], self.recalls[thres], thres / 100.0))
        return '\n'.join(res)


def save_thresholds(topic_infos: Dict[str, TopicInfo], path: str) -> None:
    with open(path, 'w') as f:
        f.write(
            json.dumps({t: v.to_json_dict()
                        for t, v in topic_infos.items()},
                       indent=4,
                       sort_keys=True))
