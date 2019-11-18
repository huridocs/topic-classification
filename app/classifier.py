import json
import logging
import os
import string
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from flask import Blueprint
from flask import current_app as app
from flask import jsonify, request
from ming import schema
from ming.odm import FieldProperty, MappedClass, Mapper

from app import model_fetcher
from app.db import session
from app.embedder import MAX_SEQ_LENGTH, Embedder
from app.model_config import InstanceConfig

classify_bp = Blueprint('classify_bp', __name__)


class ClassificationSample(MappedClass):
    """Python representation of a classification sample (training and/or predicted) in MongoDB."""

    class __mongometa__:
        session = session
        name = 'classification_sample'
        unique_indexes = [('model', 'seq')]

    _id = FieldProperty(schema.ObjectId)
    model = FieldProperty(schema.String, required=True)
    seq = FieldProperty(schema.String, required=True)
    training_labels = FieldProperty(schema.Array(schema.Object(fields={"topic": schema.String})))
    predicted_labels = FieldProperty(schema.Array(schema.Object(
        fields={"topic": schema.String, "probability": schema.Float})))
    update_timestamp = FieldProperty(datetime, if_missing=datetime.utcnow)


Mapper.compile_all()
Mapper.ensure_all_indexes()


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
        self.model_config_path = os.path.join(base_classifier_dir, model_name)
        self._load_instance()
        self.embedder = Embedder(self.instance_config.bert)
        self.predictor = tf.contrib.predictor.from_saved_model(self.instance_dir)

    def _load_instance(self):
        instances = os.listdir(self.model_config_path)
        # pick the latest released instance
        for i in sorted(instances, reverse=True):
            with open(os.path.join(
                    self.model_config_path, i, "config.json")) as f:
                d = json.loads(f.read())
                if d["is_released"]:
                    self.instance = i
                    self.instance_config = InstanceConfig(d)
                    self.instance_dir = os.path.join(self.model_config_path, self.instance)
                    self._load_vocab(os.path.join(
                        self.instance_dir,
                        self.instance_config.vocab))
                    return
        raise Exception(
            "No valid instance of model found in %s, instances were %s" % (
                self.model_config_path, instances))

    def _load_vocab(self, path_to_vocab: str):
        with open(path_to_vocab, 'r') as f:
            self.vocab = [line.rstrip() for line in f.readlines() if line.rstrip()]

    def classify(self, seqs: List[str], fixed_threshold: Optional[float] = None) \
            -> Dict[str, List[Tuple[str, float]]]:
        """ classify calculates and returns all the sequences' topic probability vectors.

        Parameters:
            seqs ([str]): The text sequences to be classified with this model.
            fixed_threshold (float or None): if set, apply the same threshold to all topics.

        Returns:
            {str: [(str, float)]}: Per sequence, the topic probabilty vector in descending
                    order, with topics below the minimum threshold (default=topic-dependent)
                    discarded.
        """
        if len(seqs) == 0:
            return {}

        embeddings = self.embedder.get_embedding(seqs)
        embedding_shape = embeddings[seqs[0]].shape
        all_embeddings = np.zeros([len(embeddings), MAX_SEQ_LENGTH, embedding_shape[1]])
        all_input_mask = np.zeros([len(embeddings), MAX_SEQ_LENGTH])

        for i, (_, matrix) in enumerate(embeddings.items()):
            all_embeddings[i][:len(matrix)] = matrix
            all_input_mask[i][:len(matrix)] = 1

        features = {
            "embeddings": all_embeddings,
            "input_mask": all_input_mask,
        }

        predictions = self.predictor(features)
        probabilities = predictions["probabilities"]
        self.logger.debug(probabilities)

        result: Dict[str, List[Tuple[str, float]]] = {}
        for i, (seq, _) in enumerate(embeddings.items()):
            # map results back to topic strings, according to classifier metadata
            # e.g. [('education', 0.8), ('rights of the child', 0.9)]
            topic_probs: List[Tuple[str, float]] = list(zip(self.vocab,
                                                            probabilities[i]))
            # sort the results in descending order of topic probability
            topic_probs.sort(key=lambda tup: tup[1], reverse=True)
            if fixed_threshold:
                # discard all topic probability tuples who are too unlikely
                topic_probs = [o for o in topic_probs if o[1] > fixed_threshold]
            result[seq] = topic_probs
        return result


@classify_bp.route('/classify', methods=['POST'])
def classify():
    # request.args: &model=upr-info_issues
    # request.get_json: {"seq"="hello world"}
    error = None
    data = request.get_json()
    args = request.args

    c = Classifier(app.config["BASE_CLASSIFIER_DIR"], args['model'])

    results = c.classify(data['seqs'])
    return jsonify(str(results))


@classify_bp.route('/classification_sample', methods=['POST'])
def add_sample():
    # request.args: &model=upr-info_issues
    # request.get_json: {"samples": [{"seq"="hello world",
    #                                 "training_labels"=[{"topic":"Murder},{"topic": "Justice"}]},
    #                                ...] }
    error = None
    data = request.get_json()
    args = request.args

    if not args['model']:
        raise Exception('You need to pass &model=...')

    for sample in data['samples']:
        existing: ClassificationSample = ClassificationSample.query.get(
            model=args['model'], seq=sample['seq'])
        if existing:
            existing.training_labels = sample['training_labels']
        else:
            n = ClassificationSample(
                model=args['model'], seq=sample['seq'], training_labels=sample['training_labels'])
    session.flush()
    return ""


@classify_bp.route('/classification_sample', methods=['DELETE'])
def delete_sample():
    # request.args: &model=upr-info_issues&seq=*
    error = None
    args = request.args

    if not args['model']:
        raise Exception('You need to pass &model=...')
    if not args['seq']:
        raise Exception('You need to pass &seq=...')
    if args['seq'] == '*':
        ClassificationSample.query.remove({'model': args['model']})
    else:
        ClassificationSample.query.remove({'model': args['model'], 'seq': args['seq']})
    session.flush()
    return ""
