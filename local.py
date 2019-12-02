"""local.py runs the app's code without starting a server, not used in prod."""

import csv
import os
from typing import Any, List

from absl import app, flags

from app import classifier, embedder, model_fetcher
from app.models import ClassificationSample, sessionLock

FLAGS = flags.FLAGS

flags.DEFINE_string('bert',
                    'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1',
                    'The bert model to use')
flags.DEFINE_string('classifier_dir', './classifier_models',
                    'The dir containing classifier models.')
flags.DEFINE_string('model', 'UPR_2percent_ps0',
                    'The model trained for a particular label set.')
flags.DEFINE_string('seq', 'increase efforts to end forced disappearance',
                    'The string sequence to process')
flags.DEFINE_string(
    'fetch_config_path', './static/model_fetching_config.json',
    'Path to the JSON config file describe where to fetch '
    'saved models from and where to copy them to.')
flags.DEFINE_integer('limit', 2000,
                     'Max number of classification samples to use')

flags.DEFINE_enum(
    'mode', 'classify',
    ['embed', 'classify', 'prefetch', 'thresholds', 'predict', 'csv'],
    'The operation to perform.')


def outputCsv() -> None:
    filename = '/tmp/%s_%d.csv' % (FLAGS.model, FLAGS.limit)
    with sessionLock, open(filename, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([
            'sequence', 'training_labels', 'predicted_sure', 'predicted_unsure',
            'revised_training_labels'
        ])
        samples: List[ClassificationSample] = list(
            ClassificationSample.query.find(
                dict(model=FLAGS.model,
                     use_for_training=True)).sort('-seqHash').limit(
                         FLAGS.limit))
        for sample in samples:
            writer.writerow([
                sample.seq, ';'.join([l.topic for l in sample.training_labels]),
                ';'.join([
                    l.topic for l in sample.predicted_labels if l.quality >= 0.6
                ]), ';'.join([
                    l.topic for l in sample.predicted_labels if l.quality < 0.6
                ]), ''
            ])
    print('Wrote %s.' % filename)


def main(_: Any) -> None:
    if FLAGS.mode == 'embed':
        e = embedder.Embedder(FLAGS.bert)
        seqs = [FLAGS.seq, FLAGS.seq + ' 2']
        ms = e.get_embedding(seqs)
        print([(seq, len(m.tostring())) for seq, m in zip(seqs, ms)])
    elif FLAGS.mode == 'classify':
        c = classifier.Classifier(FLAGS.classifier_dir, FLAGS.model)
        print(c.classify([FLAGS.seq, FLAGS.seq + ' 2']))
    elif FLAGS.mode == 'thresholds':
        c = classifier.Classifier(FLAGS.classifier_dir, FLAGS.model)
        c.refresh_thresholds(FLAGS.limit)
    elif FLAGS.mode == 'predict':
        c = classifier.Classifier(FLAGS.classifier_dir, FLAGS.model)
        c.refresh_predictions(FLAGS.limit)
    elif FLAGS.mode == 'csv':
        outputCsv()
    elif FLAGS.mode == 'prefetch':
        f = model_fetcher.Fetcher(FLAGS.fetch_config_path, FLAGS.model)
        dst = f.fetchAll()
        for l in dst:
            print(l)
    return


if __name__ == '__main__':
    os.environ['TFHUB_CACHE_DIR'] = os.getcwd() + '/bert_models'
    app.run(main)
