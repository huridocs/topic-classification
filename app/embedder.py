import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from bert import tokenization as token
from flask import Blueprint, jsonify, request

from app.models import Embedding, hasher, session, sessionLock

# Used for making sure all sentences end up
# padded to equivalent vector lengths.
MAX_SEQ_LENGTH = 256

embed_bp = Blueprint('embed_bp', __name__)


class _Embedder:

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer: token.FullTokenizer

    def embed(self, all_input_ids: Any, all_input_masks: Any,
              all_segment_ids: Any) -> Any:
        pass


class _HubEmbedder(_Embedder):

    def __init__(self, hub_path: str):
        super().__init__()
        g = tf.Graph()
        with g.as_default():
            tf_hub = hub.Module(hub_path, trainable=True)
            t_info = tf_hub(signature='tokenization_info', as_dict=True)

            self.bert_in_ids = tf.compat.v1.placeholder(dtype=tf.int32,
                                                        shape=None)
            self.bert_in_mask = tf.compat.v1.placeholder(dtype=tf.int32,
                                                         shape=None)
            self.bert_in_segment = tf.compat.v1.placeholder(dtype=tf.int32,
                                                            shape=None)
            self.bert_out = tf_hub(inputs=dict(
                input_ids=self.bert_in_ids,
                input_mask=self.bert_in_mask,
                segment_ids=self.bert_in_segment),
                                   signature='tokens',
                                   as_dict=True)['sequence_output']

            init_op = tf.group([
                tf.compat.v1.global_variables_initializer(),
                tf.compat.v1.tables_initializer()
            ])
        g.finalize()

        self.session = tf.compat.v1.Session(graph=g)
        self.session.run(init_op)
        vocab_file, do_lower_case = self.session.run(
            [t_info['vocab_file'], t_info['do_lower_case']])
        print('###', vocab_file, do_lower_case)
        self.tokenizer = token.FullTokenizer(vocab_file=vocab_file,
                                             do_lower_case=do_lower_case)

    def embed(self, all_input_ids: Any, all_input_masks: Any,
              all_segment_ids: Any) -> Any:
        bert_inputs = {
            self.bert_in_ids: all_input_ids,
            self.bert_in_mask: all_input_masks,
            self.bert_in_segment: all_segment_ids
        }
        return self.session.run(self.bert_out, bert_inputs)


class _DirEmbedder(_Embedder):

    def __init__(self, dirname: str):
        super().__init__()
        self.predictor = tf.contrib.predictor.from_saved_model(dirname)
        vocab_file = os.path.join(dirname, 'input.vocab')
        self.tokenizer = token.FullTokenizer(vocab_file=vocab_file,
                                             do_lower_case=True)

    def embed(self, all_input_ids: Any, all_input_masks: Any,
              all_segment_ids: Any) -> Any:
        bert_inputs = {
            'input_ids': all_input_ids,
            'input_mask': all_input_masks,
            'segment_ids': all_segment_ids
        }
        return self.predictor(bert_inputs)['sequence_output']


class Embedder:

    def __init__(self, bert: str):
        if not bert:
            raise Exception('You must provide a bert in order to '
                            'interact with embeddings.')
        self.bert = bert
        self.logger = logging.getLogger('app.logger')
        self.embedder: Optional[_Embedder] = None

    def _init_session(self) -> None:
        """Lazy init of TF session."""
        if self.embedder:
            return
        if self.bert.startswith('http'):
            self.embedder = _HubEmbedder(self.bert)
        else:
            self.embedder = _DirEmbedder(self.bert)

    def get_embedding(self, seqs: List[str]) -> List[np.array]:
        if len(seqs) == 0:
            return []
        if len(seqs) > 5000:
            raise Exception(
                'You should never handle more than 5000 berts at the same time!'
            )

        hashed_seq_to_index: Dict[str, int] = {}
        for i, seq in enumerate(seqs):
            hashed_seq_to_index[hasher(seq)] = i

        result: List[np.array] = [None] * len(seqs)
        # fetch from cache
        with sessionLock:
            for entry in Embedding.query.find(
                    dict(bert=self.bert,
                         seqHash={'$in': list(hashed_seq_to_index.keys())}),
                    projection=('seqHash', 'embedding')):
                result[hashed_seq_to_index[entry.seqHash]] = pickle.loads(
                    entry.embedding)

        undone_seqs: List[str] = []
        for seq in seqs:
            if result[hashed_seq_to_index[hasher(seq)]] is None:
                undone_seqs.append(seq)

        self.logger.info(
            'Using %d of %d embedding matrices fetched from MongoDB.' %
            (len(seqs) - len(undone_seqs), len(seqs)))
        if len(undone_seqs) == 0:
            return result

        self.logger.info('Building %d embedding matrices with TensorFlow...' %
                         (len(undone_seqs)))
        done_seqs = self._build_embedding(undone_seqs)

        with sessionLock:
            for seq, matrix in zip(undone_seqs, done_seqs):
                result[hashed_seq_to_index[hasher(seq)]] = matrix
                # convert npArray to list for storage in MongoDB
                Embedding(bert=self.bert,
                          seq=seq,
                          seqHash=hasher(seq),
                          embedding=pickle.dumps(matrix))
            session.flush()
        self.logger.info('Stored %d embedding matrices in MongoDB.' %
                         len(done_seqs))
        return result

    def _build_embedding(self, seqs: List[str]) -> List[np.array]:
        if len(seqs) == 0:
            return []
        self._init_session()
        if not self.embedder:
            raise RuntimeError('Failed to initialize embedder!')

        num_seqs = len(seqs)
        all_input_ids = np.zeros([num_seqs, MAX_SEQ_LENGTH])
        all_input_masks = np.zeros([num_seqs, MAX_SEQ_LENGTH])
        all_segment_ids = np.zeros([num_seqs, MAX_SEQ_LENGTH])

        for i, seq in enumerate(seqs):
            tokens = self.embedder.tokenizer.tokenize(seq)
            if len(tokens) > MAX_SEQ_LENGTH - 2:
                tokens = tokens[:(MAX_SEQ_LENGTH - 2)]

            # CLS and SEP is a relic of multi-sentence pre-training
            tokens = ['[CLS]'] + tokens + ['[SEP]']

            # segment_ids refers again to multi-sentence pre-training
            # TODO(sam): maybe scrap
            segment_ids = [0] * len(tokens)

            # per word, get IDs from vocab file
            # e.g. [980234, 8792450, 132, 5002]
            input_ids = self.embedder.tokenizer.convert_tokens_to_ids(tokens)

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

        all_embeddings = self.embedder.embed(all_input_ids, all_input_masks,
                                             all_segment_ids)
        out: List[np.ndarray] = [None] * num_seqs
        for i in range(len(seqs)):
            out[i] = all_embeddings[i][:int(sum(all_input_masks[i]))]
        return out


@embed_bp.route('/embed', methods=['POST'])
def embed() -> Any:
    # request.get_json: {
    #     'seq'="hello world',
    #     'bert': 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    # }
    data = request.get_json()

    e = Embedder(data['bert'])
    ms = e.get_embedding(data['seqs'])
    result = [len(m.tostring()) for m in ms]
    return jsonify(result)
