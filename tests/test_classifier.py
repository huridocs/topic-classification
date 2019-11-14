import pytest
import tensorflow as tf
import tensorflow_hub as hub
from unittest.mock import MagicMock
from unittest.mock import patch

from app import classifier as c
from app import embedder as e


def dummy_predictor(path):
    return True


def dummy_hub_module(path, trainable):
    def f(signature=None,
          as_dict=None): pass
    return f


def dummy_sess_run():
    class dummy_sess:
        def run(self): pass
    return dummy_sess

def dummy_embedder(path):
    return True


class TestClassifer:
    @patch.object(tf.contrib.predictor, 'from_saved_model', dummy_predictor)
    @patch.object(e, 'Embedder', dummy_embedder)
    def test_initialize_ok(self, fs):
        fs.create_file("path/to/bert")

        fs.create_file("path/to/classifer/model")
        fs.create_file("path/to/vocab.file")
        classifier = c.Classifier(
            "path/to/bert", "path/to/classifier/model", "path/to/vocab.file")
        assert classifier.vocab is not None
        assert classifier.embedder is not None

        assert classifier.predictor is not None

