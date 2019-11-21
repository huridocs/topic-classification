import os

import pytest
from pyfakefs.fake_filesystem import FakeFilesystem

from app.classifier import Classifier


class TestClassifer:
    BASE_CLASSIFIER_PATH = './testdata'

    def test_classify(self, fs: FakeFilesystem) -> None:
        fs.add_real_directory('./testdata/test_model/test_instance')
        fs.add_real_directory('./testdata/test_model/test_instance_unreleased')
        c = Classifier('./testdata', 'test_model')

        result = c.classify(['Where is my medical book?'])

        assert c.vocab is not None
        assert c.embedder is not None
        assert c.predictor is not None

        assert c.instance == 'test_instance'
        print(result)
        assert result
        # result ~ [{topic: probability, topic2: probability, ...}, ...]
        for topic, _ in result[0].items():
            assert topic in c.vocab
        assert result[0]['Right to education'] >= 0.7

    def test_missing_base_classify_dir(self) -> None:
        fake_classifier_path = './fake_testdata'
        with pytest.raises(
                Exception,
                match='Invalid base_classifier_dir: ./fake_testdata'):
            Classifier(fake_classifier_path, 'test_model')

    def test_missing_model_dir(self) -> None:
        with pytest.raises(
                Exception,
                match='Invalid model path: ./testdata/missing_model'):
            Classifier(self.BASE_CLASSIFIER_PATH, 'missing_model')

    def test_missing_instance_dir(self, fs: FakeFilesystem) -> None:
        fs.add_real_directory('./testdata/test_model/test_instance_unreleased')
        model_path = os.path.join(self.BASE_CLASSIFIER_PATH, 'test_model')
        with pytest.raises(Exception,
                           match=('No valid instance of model found in %s, ' +
                                  'instances were %s') %
                           (model_path, r'\[\'test_instance_unreleased\'\]')):
            Classifier(self.BASE_CLASSIFIER_PATH, 'test_model')

    def test_missing_vocab_file(self, fs: FakeFilesystem) -> None:
        fs.add_real_directory('./testdata/test_model/test_instance')
        fs.remove_object('./testdata/test_model/test_instance/label.vocab')
        with pytest.raises(
                Exception,
                match=(r'Failure to load vocab file from {0} with exception')
                .format('./testdata/test_model/test_instance/label.vocab')):
            Classifier(self.BASE_CLASSIFIER_PATH, 'test_model')

    def test_invalid_bert(self, fs: FakeFilesystem) -> None:
        bad_bert_path = './bad/path/to/bert'
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
        fs.add_real_directory('./testdata/test_model/test_instance')
        fs.remove_object('./testdata/test_model/test_instance/config.json')
        fs.create_file('./testdata/test_model/test_instance/config.json',
                       contents=config)
        with pytest.raises(
                Exception,
                match=r"unsupported handle format '{0}'".format(bad_bert_path)):
            c = Classifier(self.BASE_CLASSIFIER_PATH, 'test_model')
            # Bad bert is only used on uncached embed.
            c.classify(['some string'])

    def test_missing_model(self, fs: FakeFilesystem) -> None:
        instance_path = os.path.join(self.BASE_CLASSIFIER_PATH, 'test_model',
                                     'test_instance_missing_model')
        fs.add_real_directory(instance_path)

        with pytest.raises(Exception,
                           match=('SavedModel file does not exist at: {0}'
                                  ).format(instance_path)):
            Classifier(self.BASE_CLASSIFIER_PATH, 'test_model')

    def test_missing_variables(self, fs: FakeFilesystem) -> None:
        instance_path = os.path.join(self.BASE_CLASSIFIER_PATH, 'test_model',
                                     'test_instance_missing_variables')
        fs.add_real_directory(instance_path)

        with pytest.raises(
                Exception,
                match=('{0}/variables; No such file or directory'.format(
                    instance_path))):
            Classifier(self.BASE_CLASSIFIER_PATH, 'test_model')
