"local.py is a way to test the app's functionality without starting a server, not used in prod."

import os
from app import embedder
from app import classifier
from app import model_config as mc
from app import model_fetcher
from absl import app
from absl import flags

# TODO: Label models as "released" and remove hard-coded IDs here.
DEFAULT_MODEL = "UPR_2percent_ps0"
BASE_CLASSIFIER_DIR = "./classifier_models"

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert", "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", "The bert model to use")
flags.DEFINE_string(
    "classifier_dir", BASE_CLASSIFIER_DIR, "The dir containing classifier models.")
flags.DEFINE_string("seq", "", "The string sequence to process")
flags.DEFINE_enum("mode", "embed", ["embed", "classify", "prefetch"], "The operation to perform.")


def main(argv):

    if FLAGS.mode == "embed":
        e = embedder.Embedder(FLAGS.bert)
        m = e.GetEmbedding(FLAGS.seq)
        print(len(m.tostring()))
    elif FLAGS.mode == "classify":
        c = classifier.Classifier(FLAGS.classifier_dir, DEFAULT_MODEL)
        print(c.classify(FLAGS.seq))
    elif FLAGS.mode == "prefetch":
        f = model_fetcher.Fetcher()
        dst = f.fetchAll()
        for l in dst:
            print(l)
    return


if __name__ == '__main__':
    os.environ['TFHUB_CACHE_DIR'] = os.getcwd() + "/bert_models"
    app.run(main)
