from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bert import tokenization
import tensorflow as tf
import tensorflow_hub as hub

MAX_SEQ_LENGTH = 256


class Embedder:
    def __init__(self, bert: str):
        self.bert = hub.Module(bert, trainable=True)

        tokenization_info = self.bert(
            signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)

    def embed(self, seq: str):
        tokens = self.tokenizer.tokenize(seq)
        if len(tokens) > MAX_SEQ_LENGTH - 2:
            tokens = tokens[:(MAX_SEQ_LENGTH - 2)]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (MAX_SEQ_LENGTH - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == MAX_SEQ_LENGTH
        assert len(input_mask) == MAX_SEQ_LENGTH
        assert len(segment_ids) == MAX_SEQ_LENGTH

        bert_inputs = dict(input_ids=tf.expand_dims(input_ids, 0),
                           input_mask=tf.expand_dims(input_mask, 0),
                           segment_ids=tf.expand_dims(segment_ids, 0))
        bert_outputs = self.bert(inputs=bert_inputs, signature="tokens",
                                 as_dict=True)

        seq_output = bert_outputs["sequence_output"]

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            return sess.run(seq_output)[0][0:len(tokens)]
