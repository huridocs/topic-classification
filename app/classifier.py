import collections
import csv
import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np
import tensorflow as tf
from flask import Blueprint
from flask import current_app as app
from flask import jsonify, request

from app import tasks
from app.embedder import MAX_SEQ_LENGTH, Embedder
from app.model_config import InstanceConfig
from app.models import ClassificationSample, hasher, session, sessionLock

classify_bp = Blueprint('classify_bp', __name__)


class TopicInfo:
    """Collect thresholding and quality information about one topic."""

    def __init__(self, topic: str):
        self.topic = topic
        self.num_samples = 0
        self.thresholds: Dict[int, float] = {}
        self.recalls: Dict[int, float] = {}
        self.suggested_threshold = 1.1
        self.f1_quality_at_suggested = 0.0
        self.precision_at_suggested = 0.0

    def to_json_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def load_json_dict(self, v: Dict[str, Any]) -> None:
        self.__dict__ = v
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


def ComputeThresholds(topic: str, train_probs: List[float],
                      false_probs: List[float]) -> TopicInfo:
    all_probs = train_probs + false_probs
    all_probs.sort()
    ti = TopicInfo(topic)
    ti.num_samples = len(train_probs)

    # No point in learning from too few samples.
    if ti.num_samples < 10:
        return ti

    for thres in all_probs:
        true_pos = sum([1.0 for p in train_probs if p >= thres])
        false_pos = sum([1.0 for p in false_probs if p >= thres])
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / len(train_probs)
        f1 = 2 * precision * recall / (precision + recall) if (
            precision + recall) > 0 else 0

        # Only increase suggested_threshold until precision hits 50%
        if (precision >= 0.3 and f1 > ti.f1_quality_at_suggested and
                ti.precision_at_suggested <= 0.3):
            ti.precision_at_suggested = precision
            ti.f1_quality_at_suggested = f1
            ti.suggested_threshold = thres

        for target in [20, 30, 40, 50, 60, 70, 80, 90]:
            if (target not in ti.thresholds and precision >= target / 100.0):
                ti.thresholds[target] = thres
                ti.recalls[target] = recall
    return ti


class Classifier:
    """ Classifier may classify a string sequence's topic probability vector.

        Parameters:
            base_classifier_dir (str): the local path to the dir
                    containing all saved classifier models and their instances.
            model_name (str): the name of the training model
                              (e.g. UPR_2percent_ps0).
    """

    def __init__(self, base_classifier_dir: str, model_name: str):
        self.logger = logging.getLogger()
        self.lock = threading.Lock()
        if not os.path.isdir(base_classifier_dir):
            raise Exception('Invalid base_classifier_dir: %s' %
                            base_classifier_dir)
        self.model_name = model_name
        self.model_config_path = os.path.join(base_classifier_dir, model_name)
        self.topic_infos: Dict[str, TopicInfo] = {}
        self._load_instance_config()

    def _load_instance_config(self) -> None:
        if not os.path.isdir(self.model_config_path):
            raise Exception('Invalid model path: %s' % self.model_config_path)
        instances = os.listdir(self.model_config_path)
        # pick the latest released instance
        for i in sorted(instances, reverse=True):
            with open(os.path.join(self.model_config_path, i,
                                   'config.json')) as f:
                d = json.loads(f.read())
                if d['is_released']:
                    self.instance = i
                    self.instance_config = InstanceConfig(d)
                    self.instance_dir = os.path.join(self.model_config_path,
                                                     self.instance)
                    self._init_vocab()
                    self._init_thresholds()
                    self._init_embedding()
                    self._init_predictor()
                    return
        raise Exception(
            'No valid instance of model found in %s, instances were %s' %
            (self.model_config_path, instances))

    def _init_vocab(self) -> None:
        path_to_vocab = os.path.join(self.instance_dir,
                                     self.instance_config.vocab)
        try:
            with open(path_to_vocab, 'r') as f:
                self.vocab: List[str] = [
                    line.rstrip() for line in f.readlines() if line.rstrip()
                ]
        except Exception as e:
            raise Exception(
                'Failure to load vocab file from %s with exception: %s' %
                (path_to_vocab, e))

    def _init_embedding(self) -> None:
        try:
            self.embedder = Embedder(self.instance_config.bert)
        except Exception:
            self.logger.error(
                'Failure to load embedding created using BERT model %s:' %
                self.instance_config.bert)
            raise

    def _init_predictor(self) -> None:
        try:
            self.predictor = tf.contrib.predictor.from_saved_model(
                self.instance_dir)
            self.predictor.graph.finalize()

        except Exception:
            self.logger.error(
                'Failure to create predictor based on classifer at path %s' %
                self.instance_dir)
            raise

    def _init_thresholds(self) -> None:
        path_to_thresholds = os.path.join(self.instance_dir, 'thresholds.json')
        try:
            if not os.path.exists(path_to_thresholds):
                self.logger.warning(
                    ('The model at %s does not have thresholds, ' +
                     'consider ./run local --mode thresholds --model %s') %
                    (self.instance_dir, self.model_name))
            else:
                with open(path_to_thresholds, 'r') as f:
                    for k, v in json.load(f).items():
                        ti = TopicInfo(k)
                        ti.load_json_dict(v)
                        self.topic_infos[k] = ti
            for topic in self.vocab:
                if topic not in self.topic_infos:
                    self.topic_infos[topic] = TopicInfo(topic)
        except Exception as e:
            raise Exception(
                'Failure to load thresholds file from %s with exception: %s' %
                (path_to_thresholds, e))

    def _classify_probs(self, seqs: List[str]) -> List[Dict[str, float]]:
        if len(seqs) == 0:
            return []
        # Split very large requests into chunks since
        # (intermediate) bert data is huge.
        if len(seqs) > 1000:
            return (self._classify_probs(seqs[:1000]) +
                    self._classify_probs(seqs[1000:]))

        embeddings = self.embedder.get_embedding(seqs)
        embedding_shape = embeddings[0].shape
        all_embeddings = np.zeros(
            [len(embeddings), MAX_SEQ_LENGTH, embedding_shape[1]])
        all_input_mask = np.zeros([len(embeddings), MAX_SEQ_LENGTH])

        for i, matrix in enumerate(embeddings):
            all_embeddings[i][:len(matrix)] = matrix
            all_input_mask[i][:len(matrix)] = 1

        prediction = self.predictor(
            dict(embeddings=all_embeddings, input_mask=all_input_mask))
        probabilities = prediction['probabilities']
        # TODO(bdittes): Use prediction['attention']
        self.logger.debug(probabilities)
        topic_probs: List[Dict[str, float]] = [{}] * len(seqs)
        for i in range(len(seqs)):
            # map results back to topic strings,
            # according to classifier metadata
            # e.g. {'education': 0.8, 'rights of the child': 0.9}
            topic_probs[i] = {
                t: p
                # p.item() is used to convert from numpy float to python float.
                for t, p in zip(self.vocab,
                                [p.item() for p in probabilities[i]])
                if p > 0
            }
        return topic_probs

    def _props_to_quality(self, topic_probs: List[Dict[str, float]]
                          ) -> List[Dict[str, float]]:
        topic_quality: List[Dict[str, float]] = [{}] * len(topic_probs)
        for i in range(len(topic_probs)):
            topic_quality[i] = {
                t: self.topic_infos[t].get_quality(p)
                for t, p in topic_probs[i].items()
                if p >= self.topic_infos[t].suggested_threshold
            }
        return topic_quality

    def classify(self, seqs: List[str]) -> List[Dict[str, float]]:
        """ classify and returns all the sequences' topic to quality dicts.

        Parameters:
            seqs ([str]): The text sequences to be classified with this model.

        Returns:
            [{str: float}]: For each sequence (in order),
                            the mapping from topic to quality.
        """
        return self._props_to_quality(self._classify_probs(seqs))

    @staticmethod
    def _build_info(train_labels: List[Set[str]],
                    sample_probs: List[Dict[str, float]],
                    topic: str) -> TopicInfo:
        train_probs = []
        false_probs = []
        for i, sample_trains in enumerate(train_labels):
            sample_prob = sample_probs[i].get(topic, 0.0)
            if topic in sample_trains:
                train_probs.append(sample_prob)
            else:
                false_probs.append(sample_prob)
        return ComputeThresholds(topic, train_probs, false_probs)

    def refresh_thresholds(self,
                           limit: int = 2000,
                           subset_file: Optional[str] = None) -> None:
        subset_seqs: List[str] = []
        if subset_file:
            with open(subset_file, 'r') as subset_handle:
                subset_seqs = [
                    row[0]
                    for row in csv.reader(subset_handle, delimiter=',')
                    if row
                ]
        print(subset_seqs[:10])
        with sessionLock:
            samples: List[ClassificationSample] = list(
                ClassificationSample.query.find(
                    dict(model=self.model_name,
                         use_for_training=True)).sort('-seqHash').limit(limit))
            if subset_seqs:
                samples = [
                    s for s in samples if any(x in s.seq for x in subset_seqs)
                ]
            seqs = [s.seq for s in samples]
            train_labels: List[Set[str]] = [
                set([l.topic for l in s.training_labels]) for s in samples
            ]

        sample_probs = self._classify_probs(seqs)

        # TODO(bdittes): Enable multiprocessing.
        for ti in [
                Classifier._build_info(train_labels, sample_probs, topic)
                for topic in self.vocab
        ]:
            self.logger.info(str(ti))
            self.topic_infos[ti.topic] = ti

        sample_quality = self._props_to_quality(sample_probs)
        for precision in [30, 40, 50, 60, 70, 80, 90]:
            num_complete = 0.0
            sum_extra = 0.0
            missing_topics: collections.Counter = collections.Counter()
            for i, sample_trains in enumerate(train_labels):
                num_found = 0
                for train_topic in sample_trains:
                    sample_qual = sample_quality[i].get(train_topic, 0.0)
                    if sample_qual >= precision / 100.0:
                        num_found += 1
                    else:
                        missing_topics[train_topic] += 1
                if num_found >= len(sample_trains):
                    num_complete += 1
                sum_extra += len([
                    q for q in sample_quality[i].values()
                    if q >= precision / 100.0
                ]) - num_found
            print(('%02.0f%% precision -> %02.0f%% complete, ' +
                   '+%01.1f extra wrong labels') %
                  (precision, num_complete / len(train_labels) * 100,
                   sum_extra / len(train_labels)))
            # precision,
            # dict(perc_complete=num_complete / len(train_labels),
            #      avg_extra=sum_extra / len(train_labels)))
            print(' ', [(k, '%2.0f%%' % (float(v) / len(train_labels) * 100))
                        for (k, v) in missing_topics.most_common(10)])
            print()

        path_to_thresholds = os.path.join(self.instance_dir, 'thresholds.json')
        with open(path_to_thresholds, 'w') as f:
            f.write(
                json.dumps(
                    {t: v.to_json_dict()
                     for t, v in self.topic_infos.items()},
                    indent=4,
                    sort_keys=True))

    def refresh_predictions(self, limit: int = 2000) -> None:
        with sessionLock:
            samples: List[ClassificationSample] = list(
                ClassificationSample.query.find(
                    dict(model=self.model_name)).sort('-seqHash').limit(limit))
            seqs = [s.seq for s in samples]

        sample_probs = self.classify(seqs)

        with sessionLock:
            for i, sample in enumerate(samples):
                sample.predicted_labels = sorted([
                    dict(topic=t, quality=q)
                    for t, q in sample_probs[i].items()
                ],
                                                 key=lambda o: -o['quality'])
            session.flush()


class ClassifierCache:

    class Entry:

        def __init__(self, c: Classifier):
            self.c = c
            self.creation = datetime.now()

    _lock = threading.Lock()
    _entries: Dict[str, Entry] = {}

    @classmethod
    def get(cls, base_classifier_dir: str, model: str) -> Classifier:
        key = os.path.join(base_classifier_dir, model)
        with cls._lock:
            if key not in cls._entries or (
                    datetime.now() - cls._entries[key].creation).seconds > 300:
                c = Classifier(base_classifier_dir, model)
                cls._entries[key] = ClassifierCache.Entry(c)
            return cls._entries[key].c

    @classmethod
    def clear(cls, base_classifier_dir: str, model: str) -> None:
        key = os.path.join(base_classifier_dir, model)
        with cls._lock:
            if key in cls._entries:
                del cls._entries[key]

    @classmethod
    def clear_all(cls) -> None:
        with cls._lock:
            cls._entries = {}


class _RefreshThresholdsTask(tasks.TaskProvider):

    def __init__(self, json: Any):
        super().__init__(json)
        self.base_classifier_dir = json['base_classifier_dir']
        self.model = json['model']
        self.limit = json['limit'] if 'limit' in json else 2000

    def Run(self, status_holder: tasks.StatusHolder) -> None:
        status_holder.status = 'Loading classifier ' + self.model
        # Don't use the cache for long-running operations
        c = Classifier(self.base_classifier_dir, self.model)
        status_holder.status = 'Refreshing thresholds for ' + self.model
        c.refresh_thresholds(self.limit)
        ClassifierCache.clear(self.base_classifier_dir, self.model)
        status_holder.status = ''


tasks.providers['RefreshThresholds'] = _RefreshThresholdsTask


class _RefreshPredictionsTask(tasks.TaskProvider):

    def __init__(self, json: Any):
        super().__init__(json)
        self.base_classifier_dir = json['base_classifier_dir']
        self.model = json['model']
        self.limit = json['limit'] if 'limit' in json else 2000

    def Run(self, status_holder: tasks.StatusHolder) -> None:
        status_holder.status = 'Loading classifier ' + self.model
        # Don't use the cache for long-running operations
        c = Classifier(self.base_classifier_dir, self.model)
        status_holder.status = 'Refreshing predictions ' + self.model
        c.refresh_predictions(self.limit)
        status_holder.status = ''


tasks.providers['RefreshPredictions'] = _RefreshPredictionsTask


@classify_bp.route('/classify', methods=['POST'])
def classify() -> Any:
    # request.args: &model=upr-info_issues[&probs]
    # request.get_json: {'seq'='hello world', 'probs': True/False}
    data = request.get_json()
    args = request.args

    c = ClassifierCache.get(app.config['BASE_CLASSIFIER_DIR'], args['model'])
    with c.lock:
        # Allow 'probs' to be set in args or data as an option to return
        # raw probabilities.
        if 'probs' in args or ('probs' in data and data['probs']):
            results = c._classify_probs(data['seqs'])
        else:
            results = c.classify(data['seqs'])
    return jsonify(results)


@classify_bp.route('/classification_sample', methods=['POST'])
def add_samples() -> Any:
    # request.args: &model=upr-info_issues
    # request.get_json: {'samples': [{'seq'="hello world',
    #                                 'training_labels'=[
    #                                     {'topic':"Murder},
    #                                     {'topic': 'Justice'}]},
    #                                ...] }
    data = request.get_json()
    args = request.args

    if not args['model']:
        raise Exception('You need to pass &model=...')

    processed: Set[str] = set()

    with sessionLock:
        for sample in data['samples']:
            sample_labels = (sample['training_labels']
                             if 'training_labels' in sample else [])
            seqHash = hasher(sample['seq'])
            if seqHash in processed:
                continue
            processed.add(seqHash)
            existing: ClassificationSample = ClassificationSample.query.get(
                model=args['model'], seqHash=seqHash)
            if existing:
                existing.training_labels = sample_labels
                existing.use_for_training = len(sample_labels) > 0
            else:
                ClassificationSample(model=args['model'],
                                     seq=sample['seq'],
                                     seqHash=seqHash,
                                     training_labels=sample_labels,
                                     use_for_training=len(sample_labels) > 0)
        session.flush()
        return jsonify({})


@classify_bp.route('/classification_sample', methods=['GET'])
def get_samples() -> Any:
    # request.args: &model=upr-info_issues&seq=*[&limit=123]
    args = request.args

    if not args['model']:
        raise Exception('You need to pass &model=...')
    if not args['seq']:
        raise Exception('You need to pass &seq=...')
    limit = int(args['limit']) if 'limit' in args else 1000

    if args['seq'] == '*':
        filter = {'model': args['model']}
    else:
        filter = {'model': args['model'], 'seqHash': hasher(args['seq'])}
    with sessionLock:
        res = list(ClassificationSample.query.find(filter).limit(limit))
        return jsonify([r.to_json_dict() for r in res])


@classify_bp.route('/classification_sample', methods=['DELETE'])
def delete_samples() -> Any:
    # request.args: &model=upr-info_issues&seq=*
    args = request.args

    if not args['model']:
        raise Exception('You need to pass &model=...')
    if not args['seq']:
        raise Exception('You need to pass &seq=...')
    with sessionLock:
        if args['seq'] == '*':
            ClassificationSample.query.remove({'model': args['model']})
        else:
            ClassificationSample.query.remove({
                'model': args['model'],
                'seqHash': hasher(args['seq'])
            })
        session.flush()
    return jsonify({})
