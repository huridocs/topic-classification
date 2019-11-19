import logging
import pickle
from datetime import datetime
from typing import Any, Dict, List, Set

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from bert import tokenization as token
from flask import Blueprint
from flask import current_app as app
from flask import jsonify, request
from ming import schema
from ming.odm import FieldProperty, MappedClass, Mapper

from app.db import hasher, session

# Used for making sure all sentences end up
# padded to equivalent vector lengths.
MAX_SEQ_LENGTH = 256

embed_bp = Blueprint('embed_bp', __name__)


class Embedding(MappedClass):
    """Python representation of embedding cache schema in MongoDB."""

    class __mongometa__:
        session = session
        name = 'embedding_cache'
        unique_indexes = [('bert', 'seqHash')]

    _id = FieldProperty(schema.ObjectId)
    bert = FieldProperty(schema.String)
    seq = FieldProperty(schema.String)
    seqHash = FieldProperty(schema.String)
    embedding = FieldProperty(schema.Binary)
    update_timestamp = FieldProperty(datetime, if_missing=datetime.utcnow)


Mapper.compile_all()
Mapper.ensure_all_indexes()


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

    def get_embedding(self, seqs: List[str]) -> List[np.array]:
        if len(seqs) == 0:
            return []
        if len(seqs) > 5000:
            raise Exception('You should never handle more than 5000 berts at the same time!')

        hashed_seq_to_index: Dict[str, int] = {}
        for i, seq in enumerate(seqs):
            hashed_seq_to_index[hasher(seq)] = i

        result: List[np.array] = [None] * len(seqs)
        # fetch from cache
        self.logger.info('Looking up cache for %d seqs' % (len(seqs)))
        for entry in Embedding.query.find(
                dict(bert=self.bert,
                     seqHash={'$in': list(hashed_seq_to_index.keys())}),
                projection=('seqHash', 'embedding')):
            result[hashed_seq_to_index[entry.seqHash]] = pickle.loads(entry.embedding)

        undone_seqs: List[str] = []
        for seq in seqs:
            if result[hashed_seq_to_index[hasher(seq)]] is None:
                undone_seqs.append(seq)

        self.logger.info("Using %d of %d embedding matrices fetched from MongoDB." %
                         (len(seqs) - len(undone_seqs), len(seqs)))
        if len(undone_seqs) == 0:
            return result

        self.logger.info("Building %d embedding matrics with TensorFlow..." %
                         (len(undone_seqs)))
        done_seqs = self._build_embedding(undone_seqs)

        for seq, matrix in zip(undone_seqs, done_seqs):
            result[hashed_seq_to_index[hasher(seq)]] = matrix
            # convert npArray to list for storage in MongoDB
            e = Embedding(bert=self.bert, seq=seq, seqHash=hasher(
                seq), embedding=pickle.dumps(matrix))
        session.flush()
        self.logger.info("Stored %d embedding matrices in MongoDB." % len(done_seqs))
        return result

    def _build_embedding(self, seqs: List[str]) -> List[np.array]:
        num_seqs = len(seqs)
        all_input_ids = np.zeros([num_seqs, MAX_SEQ_LENGTH])
        all_input_masks = np.zeros([num_seqs, MAX_SEQ_LENGTH])
        all_segment_ids = np.zeros([num_seqs, MAX_SEQ_LENGTH])

        for i, seq in enumerate(seqs):
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
            all_input_ids[i] = input_ids
            all_input_masks[i] = input_mask
            all_segment_ids[i] = segment_ids

        bert_inputs = dict(input_ids=all_input_ids,
                           input_mask=all_input_masks,
                           segment_ids=all_segment_ids)
        bert_outputs = self.tf_hub(
            inputs=bert_inputs, signature="tokens", as_dict=True)

        seq_output = bert_outputs["sequence_output"]

        with tf.compat.v1.Session() as sess:
            sess.run([
                tf.compat.v1.global_variables_initializer(),
                tf.compat.v1.tables_initializer()])
            all_embeddings = sess.run(seq_output)
            out: List[np.ndarray] = [None] * num_seqs
            for i, seq in enumerate(seqs):
                out[i] = all_embeddings[i][:int(sum(all_input_masks[i]))]
            return out


@embed_bp.route('/embed', methods=['POST'])
def embed() -> Any:
    # request.get_json: {
    #     "seq"="hello world",
    #     "bert": "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    # }
    error = None
    data = request.get_json()

    e = Embedder(data['bert'])
    ms = e.get_embedding(data['seqs'])
    result = [str(len(m)) for m in ms]
    return jsonify(result)
