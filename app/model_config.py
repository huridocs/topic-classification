from typing import Any, Dict, List, cast

# This is the directory tree structure for models and their metadata inside of
# Google Cloud Storage. The top-level item (e.g. 'bert_classification_models')
# is called the 'bucket'.
# .
# └── bert_classification_models
#     └── multilabel
#         └── UPR_2percent_ps0
#             └── saved_model
#                 ├── label.vocab
#                 ├── training_subset.csv
#                 └── 1573031002
#                     ├── saved_model.pb
#                     └── variables
#                         ├── variables.data-00000-of-00001
#                         └── variables.index
#         └── UPR_2percent_ps512
#             └── saved_model
#                 ├── label.vocab
#                 └── 1572866595
#                     ├── saved_model.pb
#                     └── variables
#                         ├── variables.data-00000-of-00001
#                         └── variables.index


class ModelConfig(object):
    """ A ModelConfig is a configuration object whose attributes """
    """ correspond to keys in the supplied dictory."""

    def __init__(self, conf_dict: Dict[str, Any]):
        self._config = conf_dict

    def get_property(self, property_name: str) -> Any:
        if property_name not in self._config.keys():
            return None
        return self._config[property_name]

    def set_property(self, property_name: str, value: Any) -> None:
        self._config[property_name] = value


class InstanceConfig(ModelConfig):

    @property
    def bert(self) -> str:
        return cast(str, self.get_property('bert'))

    @property
    def vocab(self) -> str:
        return cast(str, self.get_property('vocab'))

    @property
    def subset(self) -> str:
        return cast(str, self.get_property('training_subset_path'))

    @property
    def is_released(self) -> bool:
        return cast(bool, self.get_property('is_released'))

    @property
    def description(self) -> str:
        return cast(str, self.get_property('description'))


class FetchConfig(ModelConfig):

    @property
    def google_acct_key_path(self) -> str:
        return cast(str, self.get_property('google_acct_key_path'))

    @property
    def bucket_name(self) -> str:
        return cast(str, self.get_property('bucket_name'))

    @property
    def model(self) -> str:
        return cast(str, self.get_property('model'))

    @property
    def instance(self) -> str:
        return cast(str, self.get_property('instance'))

    @property
    def source_files(self) -> List[str]:
        return cast(List[str], self.get_property('source_files'))

    @property
    def source_dirs(self) -> List[str]:
        return cast(List[str], self.get_property('source_dirs'))

    @property
    def dest_dir(self) -> str:
        return cast(str, self.get_property('dest_dir'))
