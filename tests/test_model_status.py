from os import path

from pyfakefs.fake_filesystem import FakeFilesystem

from app.model_status import ModelStatus


class TestModelStatus:
    BASE_CLASSIFIER_PATH = './testdata'

    def test_models(self, fs: FakeFilesystem) -> None:
        fs.add_real_directory('./testdata/test_model/test_instance')
        fs.add_real_directory('./testdata/test_model/test_instance_unreleased')
        fs.add_real_directory('./testdata/test_other_model/test_instance')
        fs.add_real_directory(
            './testdata/test_other_model/test_instance_unreleased')
        s = ModelStatus(self.BASE_CLASSIFIER_PATH)

        # with a filter
        result = s.list_potential_models(filter_str='^test_other')
        assert result == ['test_other_model']

        # all models
        result = s.list_potential_models()
        assert result == ['test_model', 'test_other_model']

    def test_instances(self, fs: FakeFilesystem) -> None:
        model = 'test_model'
        fs.add_real_directory(path.join(self.BASE_CLASSIFIER_PATH, model))
        s = ModelStatus(self.BASE_CLASSIFIER_PATH, model_name=model)

        # all
        result = s.list_model_instances()
        assert result == [
            'test_instance', 'test_instance_missing_model',
            'test_instance_missing_variables',
            'test_instance_missing_variables_data',
            'test_instance_missing_variables_index', 'test_instance_unreleased'
        ]

    def test_preferred_instance(self, fs: FakeFilesystem) -> None:
        model = 'test_model'
        fs.add_real_directory('./testdata/test_model/test_instance')
        fs.add_real_directory('./testdata/test_model/test_instance_unreleased')
        s = ModelStatus(self.BASE_CLASSIFIER_PATH, model)

        result = s.get_preferred_model_instance()
        assert result == 'test_instance'
