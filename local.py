"""local.py is a way to test the app's functionality without starting a server, not used in prod."""

import os
from typing import Any

from absl import app, flags

from app import classifier, embedder, model_fetcher

FLAGS = flags.FLAGS

flags.DEFINE_string('bert', 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1',
                    'The bert model to use')
flags.DEFINE_string('classifier_dir', './classifier_models',
                    'The dir containing classifier models.')
flags.DEFINE_string('model', 'UPR_2percent_ps0', 'The model trained for a particular label set.')
flags.DEFINE_string('seq', 'increase efforts to end forced disappearance',
                    'The string sequence to process')
flags.DEFINE_string(
    'fetch_config_path', './static/model_fetching_config.json',
    'Path to the JSON config file describe where to fetch '
    'saved models from and where to copy them to.')
flags.DEFINE_integer('limit', 2000, 'Max number of classification samples to use')

flags.DEFINE_enum('mode', 'embed', ['embed', 'classify', 'prefetch', 'thresholds'],
                  'The operation to perform.')


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
    elif FLAGS.mode == 'prefetch':
        f = model_fetcher.Fetcher(FLAGS.fetch_config_path)
        dst = f.fetchAll()
        for l in dst:
            print(l)
    return


if __name__ == '__main__':
    os.environ['TFHUB_CACHE_DIR'] = os.getcwd() + '/bert_models'
    app.run(main)
