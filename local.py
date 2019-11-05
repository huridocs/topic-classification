"local.py is a way to test the app's functionality without starting a server, not used in prod."

import os
from app import embedder
from app import classifier
from absl import app
from absl import flags
import tensorflow_hub as hub

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert", "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", "The bert model to use")
flags.DEFINE_string(
    "classifier", os.getcwd() + "/classifier_models/",
    "The classifier model to use.")
flags.DEFINE_string("seq", "", "The sequence to handle")
flags.DEFINE_enum("mode", "embed", ["embed", "classify"], "The operation to perform.")


def main(argv):
    if FLAGS.mode == "embed":
        e = embedder.Embedder(FLAGS.bert)
        m = e.GetEmbedding(FLAGS.seq)
        print(len(m.tostring()))
    elif FLAGS.mode == "classify":
        c = classifier.Classifier((FLAGS.bert, FLAGS.classifier))
        print(c.classify(FLAGS.seq))
        pass


if __name__ == '__main__':
    os.environ['TFHUB_CACHE_DIR'] = os.getcwd() + "/bert_models"
    app.run(main)
