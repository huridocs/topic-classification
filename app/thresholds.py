import json
import math
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from app.topic_info import TopicInfo


def compute_precision(true_pos: float, false_pos: float) -> float:
    if true_pos + false_pos > 0:
        return true_pos / (true_pos + false_pos)
    return 0.0


def compute_recall(true_pos: float, train_probs: List[float]) -> float:
    if len(train_probs) > 0:
        return true_pos / len(train_probs)
    return 0.0


def compute_f1(precision: float, recall: float) -> float:
    if precision + recall > 0:
        return 2 * (precision * recall / (precision + recall))
    return 0.0


def compute_scores(train_probs: List[float], false_probs: List[float],
                   thres: float) -> Tuple[float, float, float]:
    true_pos = sum([1.0 for p in train_probs if p >= thres])
    false_pos = sum([1.0 for p in false_probs if p >= thres])

    precision = compute_precision(true_pos, false_pos)
    recall = compute_recall(true_pos, train_probs)
    f1 = compute_f1(precision, recall)

    return (f1, precision, recall)


def build_score_matrix(train_probs: List[float],
                       false_probs: List[float]) -> pd.DataFrame:
    scores = []
    for thres in np.arange(0.05, 1, 0.05):
        f1, precision, recall = compute_scores(train_probs, false_probs, thres)
        scores.append((thres, f1, precision, recall))

    matrix = pd.DataFrame(scores,
                          columns=['threshold', 'f1', 'precision', 'recall'])
    return matrix.round(2).set_index('threshold')


def optimize(scores: pd.DataFrame, min_prec: float = 0.3,
             min_rec: float = 0.85) -> Any:
    org_scores = scores.copy(deep=True)
    scores = scores[(scores.precision >= min_prec) & (scores.recall >= min_rec)]
    # if minimum precision is not achived return default threshold
    if len(scores) == 0:
        scores = org_scores[org_scores.precision >= min_prec]

        if len(scores) == 0:
            print('No scores with minimum precision are found'
                  ': default threshold is used')
            return 0.5

        print('No scores with minimum recall and precision are found'
              ': optimize recall with minimum precision')
        max_rec = scores.recall.max()
        return scores[scores.recall == max_rec].index[-1]

    max_f1 = scores.f1.max()
    thresholds = scores[scores.f1 == max_f1].index
    # if at multiple thresholds the max f1 is reached select central index
    if len(thresholds) > 1:
        central_ind = math.floor(len(thresholds) / 2)
        return thresholds[central_ind]
    return thresholds[0]


def compute(topic: str, train_probs: List[float],
            false_probs: List[float]) -> TopicInfo:

    ti = TopicInfo(topic)
    ti.num_samples = len(train_probs)

    print(topic)
    # use default threshold if too less samples are provided
    if ti.num_samples < 10:
        print('Not enough samples -> default threshold is used')
        f1, precision, recall = compute_scores(train_probs, false_probs,
                                               ti.suggested_threshold)
        ti.scores = pd.DataFrame([(f1, precision, recall)],
                                 columns=['f1', 'precision', 'recall'],
                                 index=[ti.suggested_threshold]).round(2)
        return ti

    # else optimize threshold based on scores
    scores = build_score_matrix(train_probs, false_probs)
    threshold = optimize(scores)
    ti.scores = scores
    ti.suggested_threshold = threshold
    return ti


def evaluate(topic_infos: Dict[str, TopicInfo]) -> pd.DataFrame:
    measures: Dict[str, List[Any]] = {}
    for topic, info in topic_infos.items():
        metrics = info.scores.loc[info.suggested_threshold].tolist()
        metrics.append(info.suggested_threshold)
        metrics.append(info.num_samples)
        measures.update({topic: metrics})
    evaluation: pd.DataFrame = pd.DataFrame.from_dict(
        measures,
        orient='index',
        columns=['f1', 'precision', 'recall', 'threshold', 'num_samples'])
    evaluation.drop('nan', inplace=True)
    evaluation.loc['avg'] = evaluation.mean().round(2)
    return evaluation


def topics_above_threshold(topic_infos: Dict[str, TopicInfo],
                           probabilities: Dict[str, float]) -> Set[str]:
    predicted_topics = []
    for topic, info in topic_infos.items():
        if probabilities.get(topic, 0.0) >= info.suggested_threshold:
            predicted_topics.append(topic)
    return set(predicted_topics)


def save(topic_infos: Dict[str, TopicInfo], path: str) -> None:
    with open(path, 'w') as f:
        f.write(
            json.dumps({t: v.to_json_dict()
                        for t, v in topic_infos.items()},
                       indent=4,
                       sort_keys=True))


def quality(topic_infos: Dict[str, TopicInfo],
            sample_probs: List[Dict[str, float]],
            train_labels: List[Set[str]]) -> Dict[str, Any]:
    num_complete = 0.0
    num_with_prediction = 0.0
    sum_extra = 0.0
    missing_topics: Counter = Counter()
    extra_topics: Counter = Counter()
    for i, probs in enumerate(sample_probs):
        pred_topics = topics_above_threshold(topic_infos, probs)
        train_topics = train_labels[i]

        correct_predictions = train_topics.intersection(pred_topics)

        if len(pred_topics) > 0:
            num_with_prediction += 1
        if len(correct_predictions) >= len(train_topics):
            num_complete += 1

        sum_extra += len(pred_topics) - len(correct_predictions)

        missed_topics = train_topics.difference(pred_topics)
        for topic in missed_topics:
            missing_topics[topic] += 1

        bad_topics = pred_topics.difference(train_topics)
        for topic in bad_topics:
            extra_topics[topic] += 1

    completeness = num_complete / len(train_labels) * 100
    prediction_ratio = num_with_prediction / len(train_labels) * 100
    completeness_among_prediction = num_complete / num_with_prediction * 100
    extra = sum_extra / len(train_labels)
    top_missing_topics = {
        k: (float(v) / len(train_labels) * 100)
        for (k, v) in missing_topics.most_common(10)
    }
    top_extra_topics = {
        k: (float(v) / len(train_labels) * 100)
        for (k, v) in extra_topics.most_common(10)
    }
    quality = {
        'completeness': completeness,
        'prediction_ratio': prediction_ratio,
        'completeness_among_prediction': completeness_among_prediction,
        'extra': extra,
        'missing': top_missing_topics,
        'bad': top_extra_topics
    }
    return quality
