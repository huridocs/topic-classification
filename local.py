"""local.py runs the app's code without starting a server, not used in prod."""

import csv
import os
from typing import Any, Dict, List, Tuple

from absl import app, flags

from app import classifier, embedder, model_fetcher, trainer
from app.models import ClassificationSample, hasher, session, sessionLock

FLAGS = flags.FLAGS

flags.DEFINE_string('bert',
                    'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1',
                    'The bert model to use')
flags.DEFINE_string('classifier_dir', './classifier_models',
                    'The dir containing classifier models.')
flags.DEFINE_string('model', 'UPR_2percent_ps0',
                    'The model trained for a particular label set.')
flags.DEFINE_string('instance', '',
                    'If set, force the given instance dir, e.g. "1578385362".')
flags.DEFINE_string('seq', 'increase efforts to end forced disappearance',
                    'The string sequence to process')
flags.DEFINE_string(
    'fetch_config_path', './static/model_fetching_config.json',
    'Path to the JSON config file describe where to fetch '
    'saved models from and where to copy them to.')
flags.DEFINE_integer('limit', 2000,
                     'Max number of classification samples to use')
flags.DEFINE_integer('train_steps', 1000, 'Number of training iterations.')
flags.DEFINE_float('train_ratio', 0.7, 'Train / eval split of labeled data.')

flags.DEFINE_boolean(
    'probs', False,
    'If true, output raw probabilities, without using thresholds.')

flags.DEFINE_boolean(
    'csv_diff_only', True,
    'exclude csv output if training and predicted_sure match.')
flags.DEFINE_float('csv_sure', 0.6,
                   'Precision threshold for "sure" output in csv.')
flags.DEFINE_string(
    'subset_file', '',
    'If set, perform threshold learning only on samples which have a sequence '
    'containing one of the sequences in this csv file.')

flags.DEFINE_enum('mode', 'classify', [
    'embed', 'classify', 'prefetch', 'thresholds', 'predict', 'csv', 'import',
    'train'
], 'The operation to perform.')

flags.DEFINE_string(
    'csv', '',
    'If a path to a csv file is given, its data will be loaded, classified '
    'and the results are written to a new csv file')

flags.DEFINE_string('text_col', 'text',
                    'column name of the text data in a csv file')

flags.DEFINE_string('label_col', '',
                    'column name of the label data in a csv file')

flags.DEFINE_string('sharedId_col', '',
                    'column name of the sharedId in the csv file')


def outputCsv(c: classifier.Classifier) -> None:
    filename = '/tmp/%s_%d%s.csv' % (FLAGS.model, FLAGS.limit,
                                     '_diff' if FLAGS.csv_diff_only else '')
    with open(filename, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([
            'sharedId', 'sequence', 'training_labels', 'predicted_sure',
            'predicted_unsure', 'revised_training_labels'
        ])
        with sessionLock:
            samples: List[ClassificationSample] = list(
                ClassificationSample.query.find(
                    dict(model=FLAGS.model, use_for_training=True)).sort([
                        ('seqHash', -1)
                    ]).limit(FLAGS.limit))
        predicted = c.classify([s.seq for s in samples])
        for sample, pred in zip(samples, predicted):
            train_str = ';'.join([l.topic for l in sample.training_labels])
            sorted_pred: List[Tuple[str, float]] = sorted(pred.items(),
                                                          key=lambda e: -e[1])
            pred_sure_str = ';'.join(
                [t for t, q in sorted_pred if q >= FLAGS.csv_sure])
            pred_unsure_str = ';'.join(
                [t for t, q in sorted_pred if q < FLAGS.csv_sure])
            if not FLAGS.csv_diff_only or train_str != pred_sure_str:
                writer.writerow([
                    sample.sharedId, sample.seq, train_str, pred_sure_str,
                    pred_unsure_str, ''
                ])
    print('Wrote %s.' % filename)


def importData(path: str, text_col: str, label_col: str,
               sharedId_col: str) -> None:
    with open(path, 'r') as csvFile, sessionLock:
        for row in csv.DictReader(csvFile):
            seq = row[text_col]
            seqHash = hasher(seq)

            training_labels: List[Dict[str, float]] = []
            if label_col != '':
                training_label_list = eval(row[label_col])
                training_labels = [dict(topic=l) for l in training_label_list]

            sharedId = ''
            if sharedId_col != '':
                sharedId = row[sharedId_col]

            existing: ClassificationSample = ClassificationSample.query.get(
                model=FLAGS.model, seqHash=seqHash)
            if not existing:
                existing = ClassificationSample(model=FLAGS.model,
                                                seq=seq,
                                                seqHash=seqHash,
                                                training_labels=training_labels,
                                                sharedId=sharedId)
            else:
                existing.training_labels = training_labels
                existing.sharedId = sharedId
            existing.use_for_training = len(training_labels) > 0
        session.flush()


def main(_: Any) -> None:
    if FLAGS.mode == 'embed':
        e = embedder.Embedder(FLAGS.bert)
        seqs = [FLAGS.seq, FLAGS.seq + ' 2']
        ms = e.get_embedding(seqs)
        print([(seq, len(m.tostring())) for seq, m in zip(seqs, ms)])
    elif FLAGS.mode == 'classify':
        c = classifier.Classifier(
            FLAGS.classifier_dir,
            FLAGS.model,
            forced_instance=FLAGS.instance,
        )
        if FLAGS.probs:
            print(c._classify_probs([FLAGS.seq]))
        else:
            print(c.classify([FLAGS.seq]))
    elif FLAGS.mode == 'thresholds':
        c = classifier.Classifier(
            FLAGS.classifier_dir,
            FLAGS.model,
            forced_instance=FLAGS.instance,
        )
        c.refresh_thresholds(FLAGS.limit, FLAGS.subset_file)
    elif FLAGS.mode == 'predict':
        c = classifier.Classifier(
            FLAGS.classifier_dir,
            FLAGS.model,
            forced_instance=FLAGS.instance,
        )
        c.refresh_predictions(FLAGS.limit)
    elif FLAGS.mode == 'csv':
        c = classifier.Classifier(
            FLAGS.classifier_dir,
            FLAGS.model,
            forced_instance=FLAGS.instance,
        )
        outputCsv(c)
    elif FLAGS.mode == 'prefetch':
        f = model_fetcher.Fetcher(FLAGS.fetch_config_path, FLAGS.model)
        dst = f.fetchAll()
        for l in dst:
            print(l)
    elif FLAGS.mode == 'import':
        importData(FLAGS.csv, FLAGS.text_col, FLAGS.label_col,
                   FLAGS.sharedId_col)
    elif FLAGS.mode == 'train':
        c = classifier.Classifier(FLAGS.classifier_dir, FLAGS.model)
        e = embedder.Embedder(FLAGS.bert)
        t = trainer.Trainer(FLAGS.classifier_dir, FLAGS.model)
        t.train(embedder=e,
                vocab=c.vocab,
                limit=FLAGS.limit,
                forced_instance=FLAGS.instance,
                train_ratio=FLAGS.train_ratio,
                num_train_steps=FLAGS.train_steps)
    return


if __name__ == '__main__':
    os.environ['TFHUB_CACHE_DIR'] = os.getcwd() + '/bert_models'
    app.run(main)
