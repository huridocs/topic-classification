"local.py is a way to test the app's functionality without starting a server, not used in prod."

import os
from app import embedder
from app import classifier
from app import model_fetcher
from absl import app
from absl import flags

BUCKET = "bert_classification_models"

MODEL_PREFIX_PATH = "multilabel/{model_name}/saved_model"
INSTANCE_PREFIX_PATH = MODEL_PREFIX_PATH + "/{training_instance}"

LABEL_BLOB_FORMAT = MODEL_PREFIX_PATH + "/label.vocab"
MODEL_BLOB_FORMAT = INSTANCE_PREFIX_PATH + "/saved_model.pb"
#BASE_CLASSIFIER_DIR = os.getcwd() + "/classifier_models"
BASE_CLASSIFIER_DIR = os.path.join(os.getcwd(), "tmp", "multilabel")
#CLASSIFIER = "UPR_2percent_ps0_1573031002"
CLASSIFIER = "UPR_2percent_ps0"
INSTANCE = "1573031002"
CLASSIFIER_DIR = os.path.join(
    BASE_CLASSIFIER_DIR, CLASSIFIER, "saved_model", INSTANCE)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert", "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", "The bert model to use")
flags.DEFINE_string(
    "classifier", CLASSIFIER_DIR, "The classifier model to use.")
flags.DEFINE_string(
    "vocab", os.path.join(
        os.getcwd(), "tmp", MODEL_PREFIX_PATH.format(model_name=CLASSIFIER), "label.vocab"),
    "The label vocab file used to create the classifier.")
flags.DEFINE_string("seq", "", "The string sequence to process")
flags.DEFINE_enum("mode", "embed", ["embed", "classify", "prefetch"], "The operation to perform.")


def main(argv):

    if FLAGS.mode == "embed":
        e = embedder.Embedder(FLAGS.bert)
        m = e.GetEmbedding(FLAGS.seq)
        print(len(m.tostring()))
    elif FLAGS.mode == "classify":
        c = classifier.Classifier(FLAGS.bert, FLAGS.classifier, FLAGS.vocab)
        print(c.classify(FLAGS.seq))
    elif FLAGS.mode == "prefetch":
        f = model_fetcher.Fetcher()
        print(f.fetchModel())
        print(f.fetchLabel())
        print(f.fetchVariables())
    return


if __name__ == '__main__':
    os.environ['TFHUB_CACHE_DIR'] = os.getcwd() + "/bert_models"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        "/home/samschaevitz/Downloads/BERT Classification-9a8b5ef88627.json")
    app.run(main)
