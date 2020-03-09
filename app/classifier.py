import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Set

import numpy as np
import tensorflow as tf
from flask import Blueprint
from flask import current_app as app
from flask import jsonify, request

import app.thresholds as thresholds
from app.embedder import MAX_SEQ_LENGTH, Embedder
from app.model_config import InstanceConfig
from app.models import ClassificationSample, hasher, session, sessionLock
from app.topic_info import TopicInfo

classify_bp = Blueprint('classify_bp', __name__)


class Classifier:
    """ Classifier may classify a string sequence's topic probability vector.

        Parameters:
            base_classifier_dir (str): the local path to the dir
                    containing all saved classifier models and their instances.
            model_name (str): the name of the training model
                              (e.g. UPR_2percent_ps0).
    """

    def __init__(self,
                 base_classifier_dir: str,
                 model_name: str,
                 forced_instance: str = ''):
        self.logger = logging.getLogger()
        self.lock = threading.Lock()
        if not os.path.isdir(base_classifier_dir):
            raise Exception('Invalid base_classifier_dir: %s' %
                            base_classifier_dir)
        self.model_name = model_name
        self.model_config_path = os.path.join(base_classifier_dir, model_name)
        self.forced_instance = forced_instance
        self.topic_infos: Dict[str, TopicInfo] = {}
        self.quality_info: Dict[str, Any] = {}
        self._load_instance_config()

    def _load_instance_config(self) -> None:
        if not os.path.isdir(self.model_config_path):
            raise Exception('Invalid model path: %s' % self.model_config_path)
        instances = os.listdir(self.model_config_path)
        # pick the latest released instance
        for i in sorted(instances, reverse=True):
            config_path = os.path.join(self.model_config_path, i, 'config.json')
            if not os.path.exists(config_path):
                continue
            if self.forced_instance and self.forced_instance != i:
                continue
            with open(config_path) as f:
                d = json.loads(f.read())
                if d['is_released'] or self.forced_instance == i:
                    self.instance = i
                    self.instance_config = InstanceConfig(d)
                    self.instance_dir = os.path.join(self.model_config_path,
                                                     self.instance)
                    self._init_labels()
                    self._init_thresholds()
                    self._init_quality()
                    self._init_embedding()
                    self._init_predictor()
                    return
        raise Exception(
            'No valid instance of model found in %s, instances were %s' %
            (self.model_config_path, instances))

    def _init_labels(self) -> None:
        path_to_labels = os.path.join(self.instance_dir,
                                      self.instance_config.labels)
        try:
            with open(path_to_labels, 'r') as f:
                self.labels: List[str] = [
                    line.rstrip() for line in f.readlines() if line.rstrip()
                ]
        except Exception as e:
            raise Exception(
                'Failure to load labels file from %s with exception: %s' %
                (path_to_labels, e))

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
            for topic in self.labels:
                if topic not in self.topic_infos:
                    self.topic_infos[topic] = TopicInfo(topic)
        except Exception as e:
            raise Exception(
                'Failure to load thresholds file from %s with exception: %s' %
                (path_to_thresholds, e))

    def _init_quality(self) -> None:
        path_to_quality = os.path.join(self.instance_dir, 'quality.json')
        try:
            if not os.path.exists(path_to_quality):
                self.logger.warning(
                    ('The model at %s does not have quality, ' +
                     'consider ./run local --mode thresholds --model %s') %
                    (self.instance_dir, self.model_name))
            else:
                with open(path_to_quality, 'r') as f:
                    for k, v in json.load(f).items():
                        self.quality_info[k] = v
        except Exception as e:
            raise Exception(
                'Failure to load quality file from %s with exception %s' %
                (path_to_quality, e))

    def _classify_probs(self, seqs: List[str],
                        batch_size: int = 1000) -> List[Dict[str, float]]:
        if len(seqs) == 0:
            return []
        # Split very large requests into chunks since
        # (intermediate) bert data is huge.
        if len(seqs) > batch_size:
            result: List[Dict[str, float]] = []
            for i in range(0, len(seqs), batch_size):
                result += self._classify_probs(seqs[i:i + batch_size])
            return result

        embeddings = self.embedder.get_embedding(seqs)
        embedding_shape = embeddings[0].shape
        all_embeddings = np.zeros(
            [len(embeddings), MAX_SEQ_LENGTH, embedding_shape[1]])
        all_input_mask = np.zeros([len(embeddings), MAX_SEQ_LENGTH])

        for i, matrix in enumerate(embeddings):
            if matrix is None:
                raise Exception('duplicate sequences are not supported.')
            all_embeddings[i][:len(matrix)] = matrix
            all_input_mask[i][:len(matrix)] = 1

        with self.lock:
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
                for t, p in zip(self.labels,
                                [p.item() for p in probabilities[i]])
                if p > 0
            }
        return topic_probs

    def _props_to_quality(self, topic_probs: List[Dict[str, float]]
                          ) -> List[Dict[str, float]]:
        topic_quality: List[Dict[str, float]] = [{}] * len(topic_probs)
        for i in range(len(topic_probs)):
            topic_quality[i] = {
                t: self.topic_infos[t].get_confidence_at_probability(p)
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
        return thresholds.compute(topic, train_probs, false_probs)

    def refresh_thresholds(self, seqs: List[str],
                           training_labels: List[Set[str]]) -> None:
        assert len(seqs) == len(training_labels)
        if len(seqs) < 10:
            raise RuntimeError('Cannot refresh thresholds since there '
                               'are not enough training samples!')
        sample_probs = self._classify_probs(seqs)

        # TODO(bdittes): Enable multiprocessing.
        for ti in [
                Classifier._build_info(training_labels, sample_probs, topic)
                for topic in self.labels
        ]:
            self.logger.info(str(ti))
            self.topic_infos[ti.topic] = ti

        quality = thresholds.quality(self.topic_infos, sample_probs,
                                     training_labels)

        path_to_quality = os.path.join(self.instance_dir, 'quality.json')
        with open(path_to_quality, 'w') as f:
            f.write(json.dumps(quality, indent=4, sort_keys=True))

        path_to_evaluation = os.path.join(self.instance_dir, 'evaluation.csv')
        evaluation = thresholds.evaluate(self.topic_infos)
        evaluation.to_csv(path_to_evaluation)

        path_to_thresholds = os.path.join(self.instance_dir, 'thresholds.json')
        thresholds.save(self.topic_infos, path_to_thresholds)

    @staticmethod
    def quality_to_predicted_labels(sample_probs: Dict[str, float]
                                    ) -> List[Any]:
        return sorted(
            [dict(topic=t, quality=q) for t, q in sample_probs.items()],
            key=lambda o: -o['quality'])


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


@classify_bp.route('/clear_cache', methods=['PUT'])
def clear_cache() -> Any:
    ClassifierCache.clear_all()


@classify_bp.route('/classify', methods=['POST'])
def classify() -> Any:
    # request.args: &model=upr-info_issues[&probs]
    # request.get_json: {'samples': [{'seq': 'hello world',
    #                                 'sharedId': 'asda12'},
    #                                ...]}
    # returns {'samples': [{'seq': '',
    #                       'sharedId': 'asda12',
    #                       'model_version': '1234',
    #                       'predicted_labels': [...]}]}
    data = request.get_json()
    args = request.args

    if not args['model']:
        raise Exception('You need to pass &model=...')

    c = ClassifierCache.get(app.config['BASE_CLASSIFIER_DIR'], args['model'])

    seq_hash_to_seq_index: Dict[str, int] = {}
    seqs_to_classify: List[str] = []

    for i, sample in enumerate(data['samples']):
        if not sample['seq']:
            continue
        seqHash = hasher(sample['seq'])
        if seqHash in seq_hash_to_seq_index:
            continue
        seqs_to_classify.append(sample['seq'])
        seq_hash_to_seq_index[seqHash] = len(seqs_to_classify) - 1

    classified_seqs: List[Dict[str, float]]
    if seqs_to_classify:
        classified_seqs = c.classify(seqs_to_classify)

    response = []
    for i, sample in enumerate(data['samples']):
        if not sample['seq']:
            continue
        sharedId = (sample['sharedId'] if 'sharedId' in sample else '')

        predicted_labels = (Classifier.quality_to_predicted_labels(
            classified_seqs[seq_hash_to_seq_index[seqHash]]))
        response.append(
            dict(seq='' if sharedId else sample['seq'],
                 sharedId=sharedId,
                 predicted_labels=predicted_labels,
                 model_version=c.instance))
    with sessionLock:
        session.clear()
    return jsonify(dict(samples=response))


@classify_bp.route('/classification_sample', methods=['PUT'])
def add_samples() -> Any:
    # request.args: &model=upr-info_issues
    # request.get_json: {'samples': [{'seq': 'hello world',
    #                                 'sharedId': 'asda12',
    #                                 'training_labels'?: [
    #                                     {'topic': 'Murder'},
    #                                     {'topic': 'Justice'}]},
    #                                ...],
    #                    'refresh_predictions': true }
    # returns {'samples': [{'seq': '',
    #                       'sharedId': 'asda12',
    #                       'predicted_labels': [...]}]}
    data = request.get_json()
    args = request.args

    if not args['model']:
        raise Exception('You need to pass &model=...')

    processed: Set[str] = set()
    response = []

    c = ClassifierCache.get(app.config['BASE_CLASSIFIER_DIR'], args['model'])

    refresh_predictions = (data['refresh_predictions']
                           if 'refresh_predictions' in data else False)

    seq_hash_to_seq_index: Dict[str, int] = {}
    seqs_to_classify: List[str] = []

    for i, sample in enumerate(data['samples']):
        if not sample['seq']:
            continue
        seqHash = hasher(sample['seq'])
        if seqHash in seq_hash_to_seq_index:
            continue
        with sessionLock:
            existing1: ClassificationSample = ClassificationSample.query.get(
                model=args['model'], seqHash=seqHash)
            if (refresh_predictions or not existing1 or
                    not existing1.predicted_labels):
                seqs_to_classify.append(sample['seq'])
                seq_hash_to_seq_index[seqHash] = len(seqs_to_classify) - 1

    classified_seqs: List[Dict[str, float]]
    if seqs_to_classify:
        classified_seqs = c.classify(seqs_to_classify)

    for i, sample in enumerate(data['samples']):
        if not sample['seq']:
            continue
        seqHash = hasher(sample['seq'])
        sharedId = (sample['sharedId'] if 'sharedId' in sample else '')
        sample_labels = (sample['training_labels']
                         if 'training_labels' in sample else [])

        with sessionLock:
            existing: ClassificationSample = ClassificationSample.query.get(
                model=args['model'], seqHash=seqHash)
            if existing:
                response_sample = existing
                if 'training_labels' in sample:
                    existing.training_labels = sample_labels
                    existing.use_for_training = len(sample_labels) > 0
                if 'sharedId' in sample:
                    existing.sharedId = sharedId
            elif seqHash not in processed:
                response_sample = ClassificationSample(
                    model=args['model'],
                    seq=sample['seq'],
                    seqHash=seqHash,
                    training_labels=sample_labels,
                    sharedId=sharedId,
                    use_for_training=len(sample_labels) > 0)
            session.flush()
        processed.add(seqHash)
        if response_sample:
            if not response_sample.predicted_labels or refresh_predictions:
                predicted_labels = (Classifier.quality_to_predicted_labels(
                    classified_seqs[seq_hash_to_seq_index[seqHash]]))
                with sessionLock:
                    response_sample.predicted_labels = predicted_labels
                    session.flush()

            response.append(
                dict(seq='' if sharedId else sample['seq'],
                     sharedId=sharedId,
                     predicted_labels=response_sample.predicted_labels))
    with sessionLock:
        session.clear()
    return jsonify(dict(samples=response))


@classify_bp.route('/classification_sample', methods=['GET'])
def get_samples() -> Any:
    # request.args: &model=upr-info_issues&seq=*[&limit=123]
    args = request.args

    if 'model' not in args:
        raise Exception('You need to pass &model=...')
    if 'seq' not in args:
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
