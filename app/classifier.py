import logging
import time
import os
import numpy as np
import tensorflow as tf

from app import embedder

MAX_SEQ_LENGTH = 256


class Classifier:

    def __init__(self, bert_classifier: (str, str)):
        self.bert = bert_classifier[0]
        self.classifier = bert_classifier[1]
        self.embedder = embedder.Embedder(self.bert)
        self.predictor = tf.contrib.predictor.from_saved_model(self.classifier)
        self.logger = logging.getLogger('app.logger')

    def classify(self, seq: str):
        # calculate UID of seqs
        # fetch embedding matrix
        # TODO: fix padding curfuffle by exporting as CPU model
        embedding, seq_length = self.embedder.GetEmbedding(seq)
        input_mask = [1] * seq_length
        # classify seq, with its embedding matrix, using a specific model

        # We don't need this on CPU.
        while len(input_mask) < MAX_SEQ_LENGTH:
            input_mask.append(0)

        features = {
            "embeddings": embedding.reshape(1, MAX_SEQ_LENGTH, -1),
            "input_mask": np.array(input_mask).reshape(1, -1),
        }

        predictions = self.predictor(features)

        # filter results
        # map results back to topic strings, according to classifier metadata
        # return topic confidence array
        out = predictions["probabilities"][0]
        self.logger.debug(out)

        return out
