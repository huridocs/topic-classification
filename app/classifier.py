from operator import itemgetter
import logging
import string
import time
import os
import numpy as np
import tensorflow as tf

from app import embedder


def fetchVocab(vocab: str):
    with open(vocab, 'r') as f:
        return f.readlines()


class Classifier:

    def __init__(self, bert, classifier, vocab: str):
        self.logger = logging.getLogger('app.logger')
        self.vocab = fetchVocab(classifier + "/" + vocab)
        self.classifier = classifier
        self.embedder = embedder.Embedder(bert)
        self.predictor = tf.contrib.predictor.from_saved_model(self.classifier)

    def classify(self, seq: str):
        embedding = self.embedder.GetEmbedding(seq)
        input_mask = [1] * embedding.shape[0]

        # classify seq, with its embedding matrix, using a specific model
        features = {
            "embeddings": np.expand_dims(embedding, axis=0),
            "input_mask": np.expand_dims(input_mask, axis=0),
        }

        predictions = self.predictor(features)
        results = predictions["probabilities"][0]
        self.logger.debug(results)

        # map results back to topic strings, according to classifier metadata
        out = list(zip((t.rstrip() for t in self.vocab), results))
        out.sort(key=itemgetter(1), reverse=True)

        # TODO: filter results to those above a particular threshold
        out = list(o for o in out if o[1] > 0)
        return out
