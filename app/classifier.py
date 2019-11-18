import json
import logging
import string
import time
import os

from flask import jsonify
from flask import Blueprint
from flask import current_app as app
from flask import request
import numpy as np
import tensorflow as tf

from app.embedder import Embedder
from app.model_config import InstanceConfig
from app import model_fetcher

# The magic ML threshold of values we're confident are relevant enough
RETAIN_THRESHOLD = 0.4

classify_bp = Blueprint('classify_bp', __name__)


class Classifier:
    """ Classifier may classify a string sequence's topic probability vector.

        Parameters:
            base_classifier_dir (str): the local path to the dir
                    containing all saved classifier models and their instances.
    """
    def __init__(self,
                 path_to_bert: str,
                 path_to_classifier: str,
                 path_to_vocab: str):
        self.logger = logging.getLogger('app.logger')
        self._init_vocab(path_to_vocab)
        self._init_embedding(path_to_bert)
        self._init_predictor(path_to_classifier)

    def _init_vocab(self, path_to_vocab: str):
        try:
            with open(path_to_vocab, 'r') as f:
                self.vocab = f.readlines()
        except Exception as e:
            self.logger.error(
                "Failure to load vocab file from %s:" % path_to_vocab)
            raise

    def _init_embedding(self, path_to_bert: str):
        try:
            self.embedder = embedder.Embedder(path_to_bert)
        except Exception as e:
            self.logger.error(
                "Failure to load embedding created using BERT model %s:" % path_to_bert
            )
            raise

    def _init_predictor(self, path_to_classifier: str):
        try:
            self.predictor = tf.contrib.predictor.from_saved_model(
                path_to_classifier)
        except Exception as e:
            self.logger.error(
                "Failure to create predictor based on classifer at path %s" %
                path_to_classifier
            )
            raise

    def classify(self, seq: str, model_name: str) -> [(str, float)]:
        """ classify calculates and returns a particular sequence's topic probability vector.

        Parameters:
            seq (str): The text sequence to be classified with this model.
            model_name (str): the name of the training model (e.g.
                    UPR_2percent_ps0)

        Returns:
            [(str, float)]: The topic probabilty vector in descending order,
                    with topics below the minimum threshold (default=0.4)
                    discarded.
        """

        model_config_path = os.path.join(self.base_classifier_dir, model_name)
        self._load_instance(model_config_path)

        instance_dir = os.path.join(model_config_path, self.instance)
        embedder = Embedder(self.instance_config.bert)
        self._load_vocab(os.path.join(
                instance_dir,
                self.instance_config.vocab))

        predictor = tf.contrib.predictor.from_saved_model(instance_dir)
        embedding = embedder.GetEmbedding(seq)
        input_mask = [1] * embedding.shape[0]

        # classify seq, with its embedding matrix, using a specific model
        features = {
            "embeddings": np.expand_dims(embedding, axis=0),
            "input_mask": np.expand_dims(input_mask, axis=0),
        }

        predictions = predictor(features)
        probabilities = predictions["probabilities"][0]
        logging.getLogger().debug(probabilities)

        # map results back to topic strings, according to classifier metadata
        # e.g. [('education', 0.8), ('rights of the child', 0.9)]
        topic_probs = list(zip(
            (topic.rstrip() for topic in self.vocab),
            probabilities))
        # sort the results in descending order of topic probability
        topic_probs.sort(key=lambda tup: tup[1], reverse=True)

        # discard all topic probability tuples who are too unlikely
        topic_probs = [o for o in topic_probs if o[1] > RETAIN_THRESHOLD]
        self.logger.info(
            "Filtered %d results that were at or below the precision "
            "threshold." % (len(self.vocab) - len(topic_probs)))
        return topic_probs


@classify_bp.route('/classify', methods=['POST'])
def classify():
    # request.get_json: {"seq"="hello world", "model"="upr_info_issues"}
    error = None
    data = request.get_json()
    args = request.args

    c = Classifier(app.config["BASE_CLASSIFIER_DIR"])

    results = c.classify(data['seq'], args["model"])
    return jsonify(str(results))
