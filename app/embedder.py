import logging
from bert import tokenization as token
from datetime import datetime
from ming import Field
from ming import schema
from ming import Session
from ming import create_datastore
from ming.declarative import Document
import numpy
import tensorflow as tf
import tensorflow_hub as hub

# Used for making sure all sentences end up
# padded to equivalent vector lengths.
MAX_SEQ_LENGTH = 256

bind = create_datastore('tutorial')
session = Session(bind)


class Embedding(Document):
    """Python representation of embedding cache schema in MongoDB."""

    class __mongometa__:
        session = session
        name = 'embedding_matrix'

    _id = Field(schema.ObjectId)
    bert = Field(schema.String)
    seq = Field(schema.String)
    embedding = Field(schema.Array(schema.Array(schema.Float)))
    update_timestamp = Field(datetime, if_missing=datetime.utcnow)


class Embedder:
    def __init__(self, bert: str):
        self.bert = bert
        self.tf_hub = hub.Module(bert, trainable=True)

        t_info = self.tf_hub(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            # vocab_file is a BERT-global, stable mapping {tokens: ids}
            vocab_file, do_lower_case = sess.run(
                [t_info["vocab_file"],
                 t_info["do_lower_case"]])
            self.tokenizer = token.FullTokenizer(
                vocab_file=vocab_file,
                do_lower_case=do_lower_case)

    def GetEmbedding(self, seq: str):
        # fetch from MongoDB
        obj = Embedding.m.get(bert=self.bert, seq=seq)
        # if DNE, create and store in MongoDB
        if not obj:
            logging.info("Embedding matrix for bert=%s, seq=%s "
                         "not found in cache. Generating..." %
                         (self.bert, seq))
            matrix = self.embed(seq)
            print(type(matrix))
            print(type(matrix[0]))
            print(type(matrix[0][0]))
            # convert numpyArray to list
            l_matrix = matrix.tolist()
            print(type(l_matrix))
            print(type(l_matrix[0]))
            print(type(l_matrix[0][0]))

            e = Embedding.make(
                dict(bert=self.bert, seq=seq, embedding=l_matrix))
            e.m.save()
            logging.info("Embedding matrix stored in MongoDB.")
            return matrix

        logging.info("Using embedding matrix fetched from MongoDB...")
        # convert list back to numpyArray
        matrix = numpy.array(obj.embedding)
        print(type(matrix))
        print(type(matrix[0]))
        print(type(matrix[0][0]))
        logging.debug(matrix)
        return matrix

    def embed(self, seq: str):
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
        bert_outputs = self.tf_hub(inputs=bert_inputs, signature="tokens",
                                 as_dict=True)

        seq_output = bert_outputs["sequence_output"]

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            out = sess.run(seq_output)[0][0:len(tokens)]
            return out
