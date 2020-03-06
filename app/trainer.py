import json
import logging
import os
from datetime import datetime
from distutils.dir_util import copy_tree
from typing import Any, Callable, List, Set

import numpy as np
import pandas as pd
import tensorflow as tf
from bert import tokenization
from flask import Blueprint
from models import class_based_attention

import app.tasks as tasks
from app.classifier import Classifier, ClassifierCache
from app.embedder import MAX_SEQ_LENGTH, Embedder

train_bp = Blueprint('train_bp', __name__)

# LEARNING_RATE = 2e-3
LEARNING_RATE = 0.02
NUM_WARMUP_STEPS = 100
WARMUP_PROPORTION = 0.0
DROPOUT = 0.1
# SHARED_SIZE = 512
SHARED_SIZE = 0
BATCH_SIZE = 256


def _one_hot_labels(label_list: List[str], all_labels: List[str]) -> List[int]:
    labels = [0] * len(all_labels)
    indices = [all_labels.index(label) for label in label_list]
    for ind in indices:
        labels[ind] = 1
    return labels


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
                     drop_remainder: bool = True) -> Any:
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

        dataset = dataset.batch(batch_size=BATCH_SIZE,
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


class Trainer:

    def __init__(self, base_classifier_dir: str, model_name: str) -> None:
        self.logger = logging.getLogger()
        self.model_name = model_name
        self.base_classifier_dir = base_classifier_dir
        self.model_config_path = os.path.join(base_classifier_dir, model_name)
        if not os.path.isdir(base_classifier_dir):
            raise Exception('Invalid base_classifier_dir: %s' %
                            base_classifier_dir)

    def train(self,
              embedder: Embedder,
              vocab: List[str],
              seqs: List[str],
              training_labels: List[Set[str]],
              forced_instance: str = '',
              train_ratio: float = 0.9,
              num_train_steps: int = 1000,
              status_logger: Callable[[str], None] = print) -> Classifier:
        assert len(seqs) == len(training_labels)
        # timestamp = str(1578318208)
        timestamp = str(int(datetime.utcnow().timestamp()))
        if forced_instance:
            timestamp = forced_instance
        instance_path = os.path.join(self.model_config_path, timestamp)
        train_path = os.path.join(instance_path, 'train')
        if 'nan' not in vocab:
            vocab.append('nan')
        num_classes = len(vocab)

        data = pd.DataFrame(data=dict(seq=seqs,
                                      one_hot_labels=[
                                          _one_hot_labels(list(tl), vocab)
                                          for tl in training_labels
                                      ]))

        if not os.path.exists(instance_path):
            os.makedirs(instance_path, exist_ok=True)
        if not os.path.exists(train_path):
            os.makedirs(train_path, exist_ok=True)

        with open(os.path.join(instance_path, 'label.vocab'), 'w') as f:
            f.writelines([label + '\n' for label in vocab])

        config = dict(bert=embedder.bert,
                      vocab='label.vocab',
                      is_released=False,
                      subset_file='test_seqs.csv',
                      description='From Trainer.train')
        with open(os.path.join(instance_path, 'config.json'), 'w') as f:
            json.dump(config, f)

        train_values = data.sample(frac=train_ratio, random_state=42)
        test_values = data.drop(train_values.index)

        train_values['seq'].to_csv(os.path.join(instance_path,
                                                'train_seqs.csv'),
                                   index=False,
                                   header='text')
        test_values['seq'].to_csv(os.path.join(instance_path, 'test_seqs.csv'),
                                  index=False,
                                  header='text')

        # dev_values = test_values.sample(frac=0.50, random_state=42)
        # test_values = test_values.drop(dev_values.index)

        train_examples = create_examples(train_values, num_classes, 'train')
        # eval_examples = create_examples(dev_values, num_classes, 'dev')

        params = dict(num_classes=num_classes,
                      learning_rate=LEARNING_RATE,
                      num_warmup_steps=NUM_WARMUP_STEPS,
                      dropout=DROPOUT,
                      class_based_attention=True,
                      shared_size=SHARED_SIZE,
                      num_train_steps=num_train_steps)

        run_config = tf.estimator.RunConfig(model_dir=train_path,
                                            save_checkpoints_steps=min(
                                                [num_train_steps, 500]))

        model_fn = class_based_attention.model_fn_builder(use_tpu=False)

        estimator = tf.estimator.Estimator(model_fn=model_fn,
                                           config=run_config,
                                           params=params)

        status_logger('***** Started training at {} *****'.format(
            datetime.now()))
        train_input_fn = input_fn_builder(train_examples,
                                          num_classes,
                                          embedder,
                                          is_training=True,
                                          drop_remainder=False)

        saved_model_path = os.path.join(train_path, 'saved_models')
        estimator.train(input_fn=train_input_fn, steps=num_train_steps)
        save_model(saved_model_path, estimator)
        status_logger('***** Finished training at {} *****'.format(
            datetime.now()))

        copy_tree(
            os.path.join(saved_model_path,
                         sorted(os.listdir(saved_model_path), reverse=True)[0]),
            instance_path)

        c = Classifier(self.base_classifier_dir,
                       self.model_name,
                       forced_instance=timestamp)
        c.refresh_thresholds([seqs[i] for i in test_values.index],
                             [training_labels[i] for i in test_values.index])

        config['is_released'] = True
        with open(os.path.join(instance_path, 'config.json'), 'w') as f:
            json.dump(config, f)
        ClassifierCache.clear(self.base_classifier_dir, self.model_name)
        return c


DEFAULT_BERT = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'


class _TrainModel(tasks.TaskProvider):

    def __init__(self, json: Any):
        super().__init__(json)
        self.base_classifier_dir = json['base_classifier_dir']
        self.model = json['model']
        self.vocab = json['vocab'] if 'vocab' in json else None
        self.bert = json['bert'] if 'bert' in json else None
        self.num_train_steps = json[
            'num_train_steps'] if 'num_train_steps' in json else 1000
        self.train_ratio = json['train_ratio'] if 'train_ratio' in json else 0.9
        self.seqs = [sample['seq'] for sample in json['samples']]
        self.training_labels = [
            set(sample['training_labels']) for sample in json['samples']
        ]

    def Run(self, status_holder: tasks.StatusHolder) -> None:
        status_holder.status = 'Training model ' + self.model
        # Don't use the cache for long-running operations
        if not self.bert or not self.vocab:
            try:
                c = Classifier(self.base_classifier_dir, self.model)
                if not self.bert:
                    self.bert = c.embedder.bert
                if not self.vocab:
                    self.vocab = c.vocab
            except Exception:
                if not self.bert:
                    self.bert = DEFAULT_BERT
                if not self.vocab:
                    raise Exception('Cannot run without vocab!')

        e = Embedder(self.bert)
        t = Trainer(self.base_classifier_dir, self.model)
        c = t.train(embedder=e,
                    vocab=self.vocab,
                    seqs=self.seqs,
                    training_labels=self.training_labels,
                    num_train_steps=self.num_train_steps,
                    train_ratio=self.train_ratio,
                    status_logger=status_holder.SetStatus)
        status_holder.status = ('Trained model {}'.format(' '.join([
            '{}: {}'.format(tn, str(ti)) for tn, ti in c.topic_infos.items()
        ])))


tasks.providers['TrainModel'] = _TrainModel
