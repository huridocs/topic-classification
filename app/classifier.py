import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Set

import numpy as np
import tensorflow as tf
from flask import Blueprint
from flask import current_app as app
from flask import jsonify, request
from ming import schema
from ming.odm import FieldProperty, MappedClass, Mapper

from app.db import hasher, session
from app.embedder import MAX_SEQ_LENGTH, Embedder
from app.model_config import InstanceConfig

classify_bp = Blueprint('classify_bp', __name__)


class ClassificationSample(MappedClass):
    """Python representation of a classification sample (training and/or predicted) in MongoDB."""

    class __mongometa__:
        session = session
        name = 'classification_sample'
        indexes = [('model',)]
        unique_indexes = [('model', 'seqHash')]

    _id = FieldProperty(schema.ObjectId)
    model = FieldProperty(schema.String, required=True)
    seq = FieldProperty(schema.String, required=True)
    seqHash = FieldProperty(schema.String, required=True)
    training_labels = FieldProperty(schema.Array(schema.Object(fields={'topic': schema.String})))
    predicted_labels = FieldProperty(schema.Array(schema.Object(
        fields={'topic': schema.String, 'probability': schema.Float})))
    update_timestamp = FieldProperty(datetime, if_missing=datetime.utcnow)


Mapper.compile_all()
Mapper.ensure_all_indexes()


class TopicInfo:
    def __init__(self, topic: str):
        self.topic = topic
        self.num_samples = 0
        self.thresholds: Dict[int, float] = {}
        self.recalls: Dict[int, float] = {}
        self.suggested_threshold = 1.1
        self.f1_quality = 0.0

    def load_json(self, v: Any) -> None:
        self.__dict__ = v
        self.thresholds = {int(k): v for k, v in self.thresholds.items()}
        self.recalls = {int(k): v for k, v in self.recalls.items()}

    def get_quality(self, prob: float) -> float:
        if len(self.thresholds) == 0:
            return prob
        quality = 0.0
        for precision_100, threshold in self.thresholds.items():
            if prob > threshold:
                quality = precision_100 / 100.0
        return quality

    def compute_threshols(self, train_probs: List[float], false_probs: List[float]) -> None:
        all_probs = train_probs + false_probs
        all_probs.sort()
        self.num_samples = len(train_probs)

        # No point in learning from too few samples.
        if self.num_samples < 10:
            return

        for thres in all_probs:
            true_pos = sum([1.0 for p in train_probs if p >= thres])
            false_pos = sum([1.0 for p in false_probs if p >= thres])
            precision = true_pos / (true_pos + false_pos)
            for target in [20, 30, 40, 50, 60, 70, 80, 90]:
                if target not in self.thresholds and precision >= target / 100.0:
                    self.thresholds[target] = thres
                    self.recalls[target] = true_pos / len(train_probs)

        if len(self.thresholds) == 0:
            return

        def f1(precision_100: int) -> float:
            precision = precision_100 / 100.0
            recall = self.recalls[precision_100]
            return (2 * precision * recall / (precision + recall))
        best_precision = max(self.thresholds.keys(), key=f1)
        self.suggested_threshold = self.thresholds[best_precision]
        self.f1_quality = f1(best_precision)

    def __str__(self) -> str:
        res = ['%s has %d train, best quality %.02f@t=%.02f' %
               (self.topic, self.num_samples, self.f1_quality, self.suggested_threshold)]
        for thres in self.thresholds.keys():
            res.append('  t=%.02f -> %.02f@p%.01f' %
                       (self.thresholds[thres], self.recalls[thres], thres/100.0))
        return '\n'.join(res)


class Classifier:
    """ Classifier may classify a string sequence's topic probability vector.

        Parameters:
            base_classifier_dir (str): the local path to the dir
                    containing all saved classifier models and their instances.
            model_name (str): the name of the training model (e.g. UPR_2percent_ps0).
    """

    def __init__(self,
                 base_classifier_dir: str,
                 model_name: str):
        self.logger = logging.getLogger()
        if not os.path.isdir(base_classifier_dir):
            raise Exception(
                'Invalid base_classifier_dir: %s' % base_classifier_dir)
        self.model_name = model_name
        self.model_config_path = os.path.join(base_classifier_dir, model_name)
        self.topic_infos: Dict[str, TopicInfo] = {}
        self._load_instance_config()
        self._init_embedding()
        self._init_predictor()

    def _load_instance_config(self) -> None:
        if not os.path.isdir(self.model_config_path):
            raise Exception(
                'Invalid model path: %s' % self.model_config_path)
        instances = os.listdir(self.model_config_path)
        # pick the latest released instance
        for i in sorted(instances, reverse=True):
            with open(os.path.join(
                    self.model_config_path, i, 'config.json')) as f:
                d = json.loads(f.read())
                if d['is_released']:
                    self.instance = i
                    self.instance_config = InstanceConfig(d)
                    self.instance_dir = os.path.join(self.model_config_path, self.instance)
                    self._init_vocab()
                    self._init_thresholds()
                    return
        raise Exception(
            'No valid instance of model found in %s, instances were %s' % (
                self.model_config_path, instances))

    def _init_vocab(self) -> None:
        path_to_vocab = os.path.join(self.instance_dir, self.instance_config.vocab)
        try:
            with open(path_to_vocab, 'r') as f:
                self.vocab = [line.rstrip() for line in f.readlines() if line.rstrip()]
        except Exception as e:
            raise Exception(
                'Failure to load vocab file from %s with exception: %s' % (path_to_vocab, e))

    def _init_embedding(self) -> None:
        try:
            self.embedder = Embedder(self.instance_config.bert)
        except Exception:
            self.logger.error(
                'Failure to load embedding created using BERT model %s:' % self.instance_config.bert
            )
            raise

    def _init_predictor(self) -> None:
        try:
            self.predictor = tf.contrib.predictor.from_saved_model(self.instance_dir)
        except Exception:
            self.logger.error(
                'Failure to create predictor based on classifer at path %s' % self.instance_dir
            )
            raise

    def _init_thresholds(self) -> None:
        path_to_thresholds = os.path.join(self.instance_dir, 'thresholds.json')
        try:
            if not os.path.exists(path_to_thresholds):
                self.logger.warn(('The model at %s does not have thresholds, ' +
                                  'consider ./run local --mode thresholds --model %s') %
                                 (self.instance_dir, self.model_name))
            else:
                with open(path_to_thresholds, 'r') as f:
                    for k, v in json.load(f).items():
                        ti = TopicInfo(k)
                        ti.load_json(v)
                        self.topic_infos[k] = ti
            for topic in self.vocab:
                if topic not in self.topic_infos:
                    self.topic_infos[topic] = TopicInfo(topic)
        except Exception as e:
            raise Exception(
                'Failure to load thresholds file from %s with exception: %s' %
                (path_to_thresholds, e))

    def classify(self, seqs: List[str], use_thresholds: bool = True) \
            -> List[Dict[str, float]]:
        """ classify calculates and returns all the sequences' topic probability vectors.

        Parameters:
            seqs ([str]): The text sequences to be classified with this model.
            use_thresholds: if true, use per-topic thresholds and output quality per topic.

        Returns:
            {str: [(str, float)]}: Per sequence, the topic probabilty vector in descending
                    order, with topics below the minimum threshold (default=topic-dependent)
                    discarded.
        """
        if len(seqs) == 0:
            return []
        # Split very large requests into chunks since (intermediate) bert data is huge.
        if len(seqs) > 1000:
            return (self.classify(seqs[:1000], use_thresholds) +
                    self.classify(seqs[1000:], use_thresholds))

        embeddings = self.embedder.get_embedding(seqs)
        embedding_shape = embeddings[0].shape
        all_embeddings = 0.5 * np.ones([len(embeddings), MAX_SEQ_LENGTH, embedding_shape[1]])
        all_input_mask = np.zeros([len(embeddings), MAX_SEQ_LENGTH])

        for i, matrix in enumerate(embeddings):
            all_embeddings[i][:len(matrix)] = matrix
            all_input_mask[i][:len(matrix)] = 1

        features = {
            'embeddings': all_embeddings,
            'input_mask': all_input_mask,
        }

        predictions = self.predictor(features)
        probabilities = predictions['probabilities']
        self.logger.debug(probabilities)

        result: List[Dict[str, float]] = [{}] * len(seqs)
        for i, seq in enumerate(seqs):
            # map results back to topic strings, according to classifier metadata
            # e.g. {'education': 0.8, 'rights of the child':, 0.9}
            topic_probs: Dict[str, float] = dict(zip(self.vocab,
                                                     [p.item() for p in probabilities[i]]))
            if use_thresholds:
                # discard all topic probability tuples who are too unlikely
                topic_probs = {t: self.topic_infos[t].get_quality(p)
                               for t, p in topic_probs.items()
                               if p >= self.topic_infos[t].suggested_threshold}
            result[i] = topic_probs
        return result

    def refresh_thresholds(self, limit: int = 2000) -> None:
        samples = list(ClassificationSample.query.find(
                       dict(model=self.model_name,
                            training_labels={'$exists': True, '$not': {'$size': 0}}))
                       .sort('-seqHash').limit(limit).all())
        seqs: List[str] = [s.seq for s in samples]
        sample_probs = self.classify(seqs, False)

        for topic in self.vocab:
            train_probs = []
            false_probs = []
            for i, sample in enumerate(samples):
                sample_prob = sample_probs[i].get(topic, 0.0)
                if sample_prob > 0.01:
                    if topic in [tl.topic for tl in sample.training_labels]:
                        train_probs.append(sample_prob)
                    else:
                        false_probs.append(sample_prob)
            ti = TopicInfo(topic)
            ti.compute_threshols(train_probs, false_probs)
            self.logger.info(str(ti))
            self.topic_infos[topic] = ti

        path_to_thresholds = os.path.join(self.instance_dir, 'thresholds.json')
        with open(path_to_thresholds, 'w') as f:
            f.write(json.dumps({t: v.__dict__ for t, v in self.topic_infos.items()},
                               indent=4, sort_keys=True))


@classify_bp.route('/classify', methods=['POST'])
def classify() -> Any:
    # request.args: &model=upr-info_issues
    # request.get_json: {'seq'="hello world'}
    data = request.get_json()
    args = request.args

    c = Classifier(app.config['BASE_CLASSIFIER_DIR'], args['model'])

    results = c.classify(data['seqs'])
    return jsonify(results)


@classify_bp.route('/classification_sample', methods=['POST'])
def add_sample() -> Any:
    # request.args: &model=upr-info_issues
    # request.get_json: {'samples': [{'seq'="hello world',
    #                                 'training_labels'=[{'topic':"Murder},{'topic': 'Justice'}]},
    #                                ...] }
    data = request.get_json()
    args = request.args

    if not args['model']:
        raise Exception('You need to pass &model=...')

    processed: Set[str] = set()

    for sample in data['samples']:
        seqHash = hasher(sample['seq'])
        if seqHash in processed:
            continue
        processed.add(seqHash)
        existing: ClassificationSample = ClassificationSample.query.get(
            model=args['model'], seqHash=seqHash)
        if existing:
            existing.training_labels = sample['training_labels']
        else:
            ClassificationSample(
                model=args['model'], seq=sample['seq'], seqHash=seqHash,
                training_labels=sample['training_labels'])
    session.flush()
    return jsonify({})


@classify_bp.route('/classification_sample', methods=['DELETE'])
def delete_sample() -> Any:
    # request.args: &model=upr-info_issues&seq=*
    args = request.args

    if not args['model']:
        raise Exception('You need to pass &model=...')
    if not args['seq']:
        raise Exception('You need to pass &seq=...')
    if args['seq'] == '*':
        ClassificationSample.query.remove({'model': args['model']})
    else:
        ClassificationSample.query.remove({'model': args['model'], 'seqHash': hasher(args['seq'])})
    session.flush()
    return jsonify({})
