import json
import logging
import os
import string
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import tensorflow as tf
from flask import Blueprint
from flask import current_app as app
from flask import jsonify, request

from app.embedder import Embedder, MAX_SEQ_LENGTH
from app.model_config import InstanceConfig
from app import model_fetcher

classify_bp = Blueprint('classify_bp', __name__)


class Classifier:
    """ Classifier may classify a string sequence's topic probability vector.

        Parameters:
            base_classifier_dir (str): the local path to the dir
                    containing all saved classifier models and their instances.
    """

    def __init__(self,
                 base_classifier_dir: str):

        self.logger = logging.getLogger()
        self.base_classifier_dir = base_classifier_dir

    def _load_instance(self, path_to_model: str):
        instances = os.listdir(path_to_model)
        # pick the latest released instance
        for i in sorted(instances, reverse=True):
            with open(os.path.join(
                    path_to_model, i, "config.json")) as f:
                d = json.loads(f.read())
                if d["is_released"]:
                    self.instance = i
                    self.instance_config = InstanceConfig(d)
                    return
        raise Exception(
            "No valid instance of model found in %s, instances were %s" % (
                path_to_model, instances))

    def _load_vocab(self, path_to_vocab: str):
        with open(path_to_vocab, 'r') as f:
            self.vocab = [line.rstrip() for line in f.readlines() if line.rstrip() != '']

    def classify(self, seqs: List[str], model_name: str, fixed_threshold: Optional[float] = None) \
            -> Dict[str, List[Tuple[str, float]]]:
        """ classify calculates and returns all the sequences' topic probability vectors.

        Parameters:
            seqs ([str]): The text sequences to be classified with this model.
            model_name (str): the name of the training model (e.g. UPR_2percent_ps0).
            fixed_threshold (float or None): if set, apply the same threshold to all topics.

        Returns:
            {str: [(str, float)]}: Per sequence, the topic probabilty vector in descending
                    order, with topics below the minimum threshold (default=topic-dependent)
                    discarded.
        """
        if len(seqs) == 0:
            return {}

        model_config_path = os.path.join(self.base_classifier_dir, model_name)
        self._load_instance(model_config_path)

        instance_dir = os.path.join(model_config_path, self.instance)
        embedder = Embedder(self.instance_config.bert)
        self._load_vocab(os.path.join(
            instance_dir,
            self.instance_config.vocab))

        predictor = tf.contrib.predictor.from_saved_model(instance_dir)
        embeddings = embedder.GetEmbedding(seqs)
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

        predictions = predictor(features)
        probabilities = predictions["probabilities"]
        logging.getLogger().debug(probabilities)

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
    # request.get_json: {"seq"="hello world", "model"="upr_info_issues"}
    error = None
    data = request.get_json()
    args = request.args

    c = Classifier(app.config["BASE_CLASSIFIER_DIR"])

    results = c.classify(data['seqs'] if 'seqs' in data else [data['seq']], args['model'])
    return jsonify(str(results))
