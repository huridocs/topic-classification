"local.py is a way to test the app's functionality without starting a server, not used in prod."

import os
from app import embedder
from app import classifier
from absl import app
from absl import flags
import tensorflow_hub as hub

BASE_CLASSIFIER_DIR = os.getcwd() + "/classifier_models"
CLASSIFIER = "UPR_2percent_ps0_1573031002"
CLASSIFIER_DIR = BASE_CLASSIFIER_DIR + "/" + CLASSIFIER

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert", "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", "The bert model to use")
flags.DEFINE_string(
    "classifier", CLASSIFIER_DIR, "The classifier model to use.")
flags.DEFINE_string(
    "vocab", CLASSIFIER_DIR + "/saved_model_label.vocab",
    "The label vocab file used to create the classifier.")
flags.DEFINE_string("seq", "", "The sequence to handle")
flags.DEFINE_enum("mode", "embed", ["embed", "classify"], "The operation to perform.")


def main(argv):
    if FLAGS.mode == "embed":
        e = embedder.Embedder(FLAGS.bert)
        m = e.GetEmbedding(FLAGS.seq)
        print(len(m.tostring()))
    elif FLAGS.mode == "classify":
        c = classifier.Classifier(FLAGS.bert, FLAGS.classifier, FLAGS.vocab)
        print(c.classify(FLAGS.seq))
    return


if __name__ == '__main__':
    os.environ['TFHUB_CACHE_DIR'] = os.getcwd() + "/bert_models"
    app.run(main)
