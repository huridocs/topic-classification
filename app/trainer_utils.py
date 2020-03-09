from typing import Any, List

import numpy as np
import pandas as pd
import tensorflow as tf
from bert import tokenization

from app.embedder import MAX_SEQ_LENGTH, Embedder


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid: str, text_a: str, labels: List[int]) -> None:
        """Constructs a InputExample.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        Only must be specified for sequence pair tasks.
        labels: (Optional) [string]. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 embeddings: np.array,
                 label_ids: Any,
                 is_real_example: bool = True) -> None:
        self.hidden_size = embeddings.shape[-1]
        self.embeddings = np.zeros([MAX_SEQ_LENGTH, self.hidden_size],
                                   dtype=np.float32)
        self.input_mask = np.zeros([MAX_SEQ_LENGTH], dtype=np.int32)
        self.embeddings[:len(embeddings)] = embeddings
        self.input_mask[:len(embeddings)] = 1
        self.label_id = np.array(label_ids, dtype=np.int32)

        self.is_real_example = is_real_example


def create_examples(data: pd.DataFrame,
                    num_classes: int,
                    set_type: str = 'train') -> List[InputExample]:
    examples = []
    for index, row in data.iterrows():
        guid = '%s-%s' % (set_type, index)
        text_a = tokenization.convert_to_unicode(row.seq)
        if set_type == 'test':
            labels = [0] * num_classes
        else:
            labels = row.one_hot_labels
        examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
    return examples


def convert_examples_to_features(examples: List[InputExample],
                                 embedder: Embedder) -> List[InputFeatures]:
    """Loads a data file into a list of `InputBatch`s."""

    embeddings = embedder.get_embedding([e.text_a for e in examples])

    return [
        InputFeatures(embeddings=em, label_ids=e.labels)
        for e, em in zip(examples, embeddings)
    ]


def input_fn_builder(examples: List[InputExample],
                     num_classes: int,
                     embedder: Embedder,
                     is_training: bool = True,
                     drop_remainder: bool = True,
                     batch_size: int = 256) -> Any:
    """Creates an `input_fn` closure to be passed to Estimator."""

    features = convert_examples_to_features(examples, embedder)

    total_size = len(features)
    hidden_size = features[0].hidden_size
    all_input_mask = np.zeros([total_size, MAX_SEQ_LENGTH], dtype=np.int32)
    all_embeddings = np.zeros([total_size, MAX_SEQ_LENGTH, hidden_size],
                              dtype=np.float32)
    all_label_ids = np.zeros([total_size, num_classes], dtype=np.int32)

    for i in range(len(features)):
        all_input_mask[i] = features[i].input_mask
        all_embeddings[i] = features[i].embeddings
        all_label_ids[i] = features[i].label_id

    def input_fn() -> Any:
        """The actual input function."""
        dataset = tf.data.Dataset.from_tensor_slices(({
            'input_mask': tf.convert_to_tensor(all_input_mask),
            'embeddings': tf.convert_to_tensor(all_embeddings)
        }, {
            'label_ids': tf.convert_to_tensor(all_label_ids),
        }))

        if is_training:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=100)

        dataset = dataset.batch(batch_size=batch_size,
                                drop_remainder=drop_remainder)
        return dataset

    return input_fn


def save_model(output_dir: str, estimator: tf.estimator.Estimator) -> None:

    def serving_input_receiver_fn() -> Any:
        embeddings = tf.placeholder(dtype=tf.float32,
                                    shape=[None, None, 768],
                                    name='embeddings')
        input_mask = tf.placeholder(dtype=tf.int32,
                                    shape=[None, None],
                                    name='input_mask')
        features = {'embeddings': embeddings, 'input_mask': input_mask}
        return tf.estimator.export.ServingInputReceiver(
            features=features, receiver_tensors=features)

    estimator.export_saved_model(output_dir, serving_input_receiver_fn)
