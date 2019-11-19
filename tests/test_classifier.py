import os
from pyfakefs import fake_filesystem
import pytest
import tensorflow as tf
import tensorflow_hub as hub
from unittest.mock import MagicMock
from unittest.mock import patch

from app.classifier import Classifier
from app.embedder import Embedder


class TestClassifer:
    BASE_CLASSIFIER_PATH = "./testdata"

    def test_initialize_ok(self):
        c = Classifier("./testdata")
        assert c

    def test_classify(self, fs) -> None:
        fs.add_real_directory("./testdata/test_model/test_instance")
        fs.add_real_directory("./testdata/test_model/test_instance_unreleased")
        c = Classifier("./testdata")

        result = c.classify(["Where is my medical book?"], "test_model")

        assert c.vocab is not None
        assert c.embedder is not None
        assert c.predictor is not None

        assert c.instance == "test_instance"
        print(result)
        assert result
        # result ~ {'seq': [(topic, probability), (topic2, probability)...], ...}
        for seq, prob_vector in result.items():
            for topic, _ in prob_vector:
                assert topic in c.vocab

    def test_missing_base_classify_dir(self) -> None:
        fake_classifier_path = "./fake_testdata"
        with pytest.raises(
                Exception, match="Invalid path_to_classifier: ./fake_testdata"):
            c = Classifier(fake_classifier_path)

    def test_missing_model_dir(self) -> None:
        c = Classifier(self.BASE_CLASSIFIER_PATH)
        with pytest.raises(
                Exception,
                match="Invalid path_to_model: ./testdata/missing_model"):
            c.classify(["foo seq"], "missing_model")

    def test_missing_instance_dir(self, fs) -> None:
        fs.add_real_directory("./testdata/test_model/test_instance_unreleased")
        c = Classifier(self.BASE_CLASSIFIER_PATH)
        with pytest.raises(
                Exception,
                match="No valid instance of model found in %s, instances were %s" % (
                        os.path.join(self.BASE_CLASSIFIER_PATH, "test_model"),
                        r"\[\'test_instance_unreleased\'\]")):
            c.classify(["foo seq"], "test_model")

    def test_missing_vocab_file(self, fs) -> None:
        fs.add_real_directory("./testdata/test_model/test_instance")
        fs.remove_object("./testdata/test_model/test_instance/label.vocab")
        c = Classifier(self.BASE_CLASSIFIER_PATH)
        with pytest.raises(
                Exception,
                match=(r"Failure to load vocab file from {0} with exception"
                       ).format("./testdata/test_model/test_instance/label.vocab")
                ):
            c.classify(["foo seq"], "test_model")

    def test_invalid_bert(self, fs) -> None:
        bad_bert_path = "./bad/path/to/bert"
        config = """
        {
            "bert": "%s",
            "vocab": "label.vocab",
            "is_released": true,
            "description": "This is the latest model from Sascha.",
            "metadata": {
                "thesaurus": "issues"
            }
        }
        """ % (bad_bert_path)
        fs.add_real_directory("./testdata/test_model/test_instance")
        fs.remove_object("./testdata/test_model/test_instance/config.json")
        fs.create_file("./testdata/test_model/test_instance/config.json",
                       contents=config)
        c = Classifier(self.BASE_CLASSIFIER_PATH)
        with pytest.raises(
                Exception,
                match=r"unsupported handle format '{0}'".format(bad_bert_path)):
            c.classify(["foo seq"], "test_model")

    def test_missing_model(self, fs) -> None:
        instance_path = os.path.join(
            self.BASE_CLASSIFIER_PATH, "test_model", "test_instance_missing_model")
        fs.add_real_directory(instance_path)

        c = Classifier(self.BASE_CLASSIFIER_PATH)
        with pytest.raises(
                Exception,
                match=(
                        "SavedModel file does not exist at: {0}"
                      ).format(instance_path)
                ):
            c.classify(["foo seq"], "test_model")

    def test_missing_variables(self, fs) -> None:
        instance_path = os.path.join(
            self.BASE_CLASSIFIER_PATH,
            "test_model",
            "test_instance_missing_variables")
        fs.add_real_directory(instance_path)

        c = Classifier(self.BASE_CLASSIFIER_PATH)
        with pytest.raises(
                Exception,
                match=(
                        "{0}/variables; No such file or directory".format(
                            instance_path)
                )):
            c.classify(["foo seq"], "test_model")
