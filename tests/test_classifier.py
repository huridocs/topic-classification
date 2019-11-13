import pytest
from unittest.mock import MagicMock

from app import classifier as c


class TestClassifer:
    def test_initialize_ok(self, fs):
        fs.create_file("path/to/bert")
        fs.create_file("path/to/classifer/model")
        fs.create_file("path/to/vocab.file")
        classifier = c.Classifier(
            "path/to/bert", "path/to/classifier/model", "path/to/vocab.file")
        assert classifier.vocab
        assert classifier.embedder
        assert classifier.predictor
