from datetime import datetime
import logging

from bert import tokenization as token
from flask import jsonify
from flask import Blueprint
from flask import current_app as app
from flask import request
from ming import Field
from ming import schema
from ming import Session
from ming import create_datastore
from ming.declarative import Document
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Used for making sure all sentences end up
# padded to equivalent vector lengths.
MAX_SEQ_LENGTH = 256

# TODO: Parameterize the MongoDB session for testing.
bind = create_datastore('classifier_dev')
session = Session(bind)

embed_bp = Blueprint('embed_bp', __name__)


class Embedding(Document):
    """Python representation of embedding cache schema in MongoDB."""

    class __mongometa__:
        session = session
        name = 'embedding_cache'

    _id = Field(schema.ObjectId)
    bert = Field(schema.String)
    seq = Field(schema.String)
    embedding = Field(schema.Array(schema.Array(schema.Float)))
    update_timestamp = Field(datetime, if_missing=datetime.utcnow)


class Embedder:
    def __init__(self, bert: str):
        self.bert = bert
        self.tf_hub = hub.Module(bert, trainable=True)
        self.logger = logging.getLogger("app.logger")

        t_info = self.tf_hub(signature="tokenization_info", as_dict=True)
        with tf.compat.v1.Session() as sess:
            # vocab_file is a BERT-global, stable mapping {tokens: ids}
            vocab_file, do_lower_case = sess.run(
                [t_info["vocab_file"],
                 t_info["do_lower_case"]])
            self.tokenizer = token.FullTokenizer(
                vocab_file=vocab_file,
                do_lower_case=do_lower_case)

    def GetEmbedding(self, seq: str) -> np.array:
        # fetch from cache
        obj = Embedding.m.get(bert=self.bert, seq=seq)
        if obj:
            self.logger.info("Using embedding matrix fetched from MongoDB...")
            # convert list back from list(floats) to np.array
            matrix = np.array(obj.embedding)
            self.logger.debug(matrix)
            return matrix

        self.logger.info("Embedding matrix for bert=%s, seq=%s "
                         "not found in cache. Generating..." %
                         (self.bert, seq))
        matrix = self._buildEmbedding(seq)

        # convert npArray to list for storage in MongoDB
        l_matrix = matrix.tolist()
        e = Embedding.make(dict(bert=self.bert, seq=seq, embedding=l_matrix))
        e.m.save()
        self.logger.info("Embedding matrix stored in MongoDB.")
        return matrix

    def _buildEmbedding(self, seq: str) -> np.array:
        tokens = self.tokenizer.tokenize(seq)
        if len(tokens) > MAX_SEQ_LENGTH - 2:
            tokens = tokens[:(MAX_SEQ_LENGTH - 2)]

        # CLS and SEP is a relic of multi-sentence pre-training
        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        # segment_ids refers again to multi-sentence pre-training
        # TODO: maybe scrap
        segment_ids = [0] * len(tokens)

        # per word, get IDs from vocab file
        # e.g. [980234, 8792450, 132, 5002]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # e.g. [1, 1, 1, 1]
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (MAX_SEQ_LENGTH - len(input_ids))  # [0, 0, 0, ...]
        input_ids += padding  # [980234, 8792450, 132, 5002, 0, 0, 0, ...]
        input_mask += padding  # [1, 1, 1, 1, 0, ...]
        segment_ids += padding  # [0, 0, 0, ...]

        assert len(input_ids) == MAX_SEQ_LENGTH
        assert len(input_mask) == MAX_SEQ_LENGTH
        assert len(segment_ids) == MAX_SEQ_LENGTH

        bert_inputs = dict(input_ids=tf.expand_dims(input_ids, 0),
                           input_mask=tf.expand_dims(input_mask, 0),
                           segment_ids=tf.expand_dims(segment_ids, 0))
        bert_outputs = self.tf_hub(
            inputs=bert_inputs, signature="tokens", as_dict=True)

        seq_output = bert_outputs["sequence_output"]

        with tf.compat.v1.Session() as sess:
            sess.run([
                tf.compat.v1.global_variables_initializer(),
                tf.compat.v1.tables_initializer()])
            out = sess.run(seq_output)[0][:len(tokens), :]

        return out


@embed_bp.route('/embed', methods=['POST'])
def embed():
    # request.get_json: {
    #     "seq"="hello world",
    #     "bert": "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    # }
    error = None
    data = request.get_json()

    e = Embedder(data['bert'])
    matrix = e.GetEmbedding(data['seq'])

    return jsonify(str(len(matrix)))
