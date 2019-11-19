import os
from typing import cast, Any, Dict, Optional

# This is the directory tree structure for models and their metadata inside of
# Google Cloud Storage. The top-level item (e.g. "bert_classification_models")
# is called the 'bucket'.
# .
# └── bert_classification_models
#     └── multilabel
#         └── UPR_2percent_ps0
#             └── saved_model
#                 ├── label.vocab
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


class InstanceConfig(ModelConfig):
    @property
    def bert(self) -> str:
        return cast(str, self.get_property("bert"))

    @property
    def vocab(self) -> str:
        return cast(str, self.get_property("vocab"))

    @property
    def is_released(self) -> bool:
        return cast(bool, self.get_property("is_released"))

    @property
    def description(self) -> str:
        return cast(str, self.get_property("description"))


class PathConfig(ModelConfig):

    def __init__(self, conf_dict: Dict[str, Any], model: str,
                 instance: Optional[str] = None, prefix: str = ""):
        ModelConfig.__init__(self, conf_dict)
        self.model = model
        self.instance = instance
        self.prefix = prefix

    @property
    def directory(self) -> str:
        folder = os.path.join(
            self.prefix,
            self.get_property("path")).format(
                model=self.model, instance=self.instance)
        return folder

    @property
    def filename(self) -> str:
        return cast(str, self.get_property("name"))

    @property
    def fqfn(self) -> str:
        return os.path.join(
            self.directory,
            self.filename).format(
                model=self.model, instance=self.instance)


# TODO: Deduplicate In and Out config classes.
class InConfig(ModelConfig):

    @property
    def google_acct_key_path(self) -> str:
        return cast(str, self.get_property("google_acct_key_path"))

    @property
    def bucket(self) -> str:
        return cast(str, self.get_property("bucket_name"))

    @property
    def model_name(self) -> str:
        return cast(str, self.get_property("model_name"))

    @property
    def bert(self) -> str:
        return cast(str, self.get_property("bert"))

    @property
    def instance_name(self) -> str:
        return cast(str, self.get_property("instance_name"))

    @property
    def saved_model(self) -> PathConfig:
        return PathConfig(
            self.get_property("saved_model"),
            self.model_name, self.instance_name)

    @property
    def variables(self) -> PathConfig:
        return PathConfig(
            self.get_property("variables"),
            self.model_name, self.instance_name)

    @property
    def vocab(self) -> PathConfig:
        return PathConfig(self.get_property("vocab"), self.model_name)


# TODO: Deduplicate In and Out config classes.
class OutConfig(ModelConfig):

    @property
    def base_dir(self) -> str:
        return cast(str, self.get_property("base_dir"))

    @property
    def model_name(self) -> str:
        return cast(str, self.get_property("model_name"))

    @property
    def instance_name(self) -> str:
        return cast(str, self.get_property("instance_name"))

    @property
    def saved_model(self) -> PathConfig:
        return PathConfig(
            self.get_property("saved_model"),
            self.get_property("model_name"),
            instance=self.get_property("instance_name"),
            prefix=self.base_dir)

    @property
    def variables(self) -> PathConfig:
        return PathConfig(
            self.get_property("variables"),
            self.model_name,
            instance=self.instance_name,
            prefix=self.base_dir)

    @property
    def vocab(self) -> PathConfig:
        return PathConfig(
            self.get_property("vocab"),
            self.model_name,
            instance=self.instance_name,
            prefix=self.base_dir)
