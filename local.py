import os
from app import embed
from absl import app
from absl import flags
import zlib
import tensorflow_hub as hub

FLAGS = flags.FLAGS

flags.DEFINE_string("bert", "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", "The bert model to use")
flags.DEFINE_string("seq", "", "The sequence to handle")
flags.DEFINE_enum("mode", "embed", ["embed"], "The operation to perform.")


def main(argv):
    if FLAGS.mode == "embed":
        e = embed.Embedder(FLAGS.bert)
        print(e.embed(FLAGS.seq))


if __name__ == '__main__':
    os.environ['TFHUB_CACHE_DIR'] = os.getcwd() + "/bert_models"
    app.run(main)
