from operator import itemgetter
import logging
import string
import time
import os
import numpy as np
import tensorflow as tf

from app import embedder

MAX_SEQ_LENGTH = 256


class Classifier:

    def __init__(self, bert_classifier: (str, str), vocab: str):
        self.logger = logging.getLogger('app.logger')
        self.bert = bert_classifier[0]
        self.classifier = bert_classifier[1]
        self.vocab = self.fetchVocab(vocab)
        self.embedder = embedder.Embedder(self.bert)
        self.predictor = tf.contrib.predictor.from_saved_model(self.classifier)

    def fetchVocab(self, vocab: str):
        with open(vocab, 'r') as f:
            return f.readlines()

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
        probs = predictions["probabilities"][0]
        self.logger.debug(probs)

        # map results back to topic strings, according to classifier metadata
        out = list(zip((t.rstrip() for t in self.vocab), probs))
        self.logger.info("============UNSORTED==============")
        for topic in out:
            self.logger.info(topic)

        out.sort(key=itemgetter(1), reverse=True)
        self.logger.info("==============SORTED==============")
        for topic in out:
            self.logger.info(topic)

        # return topic confidence array
        return out
