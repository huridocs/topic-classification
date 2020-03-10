import json
import logging
import os
from datetime import datetime
from distutils.dir_util import copy_tree
from typing import Any, Callable, List, Set

import pandas as pd
import tensorflow as tf
from flask import Blueprint
from models import class_based_attention

import app.tasks as tasks
from app.classifier import Classifier, ClassifierCache
from app.embedder import Embedder
from app.trainer_utils import create_examples, input_fn_builder, save_model

train_bp = Blueprint('train_bp', __name__)

# LEARNING_RATE = 2e-3
LEARNING_RATE = 0.02
NUM_WARMUP_STEPS = 0
DROPOUT = 0.1
# SHARED_SIZE = 512
SHARED_SIZE = 0

# In case of OOM errors, reduce this.
BATCH_SIZE = 256


def _one_hot_labels(label_list: List[str], all_labels: List[str]) -> List[int]:
    labels = [0] * len(all_labels)
    indices = [all_labels.index(label) for label in label_list]
    for ind in indices:
        labels[ind] = 1
    return labels


class LogStatusHook(tf.estimator.SessionRunHook):

    def __init__(self, num_train_steps: int,
                 status_logger: Callable[[str], None]):
        super().__init__()
        self.done_steps = 0
        self.num_train_steps = num_train_steps
        self.status_logger = status_logger

    def after_run(self, _run_context: Any, _run_values: Any) -> None:
        self.done_steps += 1
        self.status_logger('Trained {} of {} steps.'.format(
            self.done_steps, self.num_train_steps))


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
              labels: List[str],
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
        if 'nan' not in labels:
            labels.append('nan')
        num_classes = len(labels)

        data = pd.DataFrame(data=dict(seq=seqs,
                                      one_hot_labels=[
                                          _one_hot_labels(list(tl), labels)
                                          for tl in training_labels
                                      ]))

        train_values = data.sample(frac=train_ratio, random_state=42)
        test_values = data.drop(train_values.index)

        if not os.path.exists(instance_path):
            os.makedirs(instance_path, exist_ok=True)
        if not os.path.exists(train_path):
            os.makedirs(train_path, exist_ok=True)

        with open(os.path.join(instance_path, 'label.vocab'), 'w') as f:
            f.writelines([label + '\n' for label in labels])

        config = dict(bert=embedder.bert,
                      labels='label.vocab',
                      is_released=False,
                      subset_file='test_seqs.csv',
                      num_train=len(train_values),
                      num_test=len(test_values),
                      description='From Trainer.train')
        with open(os.path.join(instance_path, 'config.json'), 'w') as f:
            json.dump(config, f)

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
                                          drop_remainder=False,
                                          batch_size=BATCH_SIZE)

        saved_model_path = os.path.join(train_path, 'saved_models')

        estimator.train(input_fn=train_input_fn,
                        steps=num_train_steps,
                        hooks=[LogStatusHook(num_train_steps, status_logger)])
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
        self.labels = json['labels'] if 'labels' in json else None
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
        if not self.bert or not self.labels:
            try:
                c = Classifier(self.base_classifier_dir, self.model)
                if not self.bert:
                    self.bert = c.embedder.bert
                if not self.labels:
                    self.labels = c.labels
            except Exception:
                if not self.bert:
                    self.bert = DEFAULT_BERT
                if not self.labels:
                    raise Exception('Cannot run without labels!')

        e = Embedder(self.bert)
        t = Trainer(self.base_classifier_dir, self.model)
        c = t.train(embedder=e,
                    labels=self.labels,
                    seqs=self.seqs,
                    training_labels=self.training_labels,
                    num_train_steps=self.num_train_steps,
                    train_ratio=self.train_ratio,
                    status_logger=status_holder.SetStatus)
        status_holder.status = ('Trained model {}'.format(' '.join([
            '{}: {}'.format(tn, str(ti)) for tn, ti in c.topic_infos.items()
        ])))


tasks.providers['TrainModel'] = _TrainModel
