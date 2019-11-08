import os

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

# TODO: Stop hardcoding this and provide sane default values for the CLI.
# IN = {
#     "bucket_name": "bert_classification_models",
#     "model_name": "UPR_2percent_ps0",
#     "instance_name": "1573031002",
#     "saved_model": {
#         "name": "saved_model.pb",
#         "path": "multilabel/{model}/saved_model/{instance}",
#     },
#     "variables": {
#         "name": "*",
#         # TODO: come up with a convention on handling recursive copying in the
#         # input configuration.
#         "path": "multilabel/{model}/saved_model/{instance}/variables",
#     },
#     "vocab": {
#         "name": "label.vocab",
#         "path": "multilabel/{model}/saved_model",
#     },
# }


class ModelConfig(object):
    """ A ModelConfig is a configuration object whose attributes """
    """ correspond to keys in the supplied dictory."""

    def __init__(self, conf_dict):
        self._config = conf_dict

    def get_property(self, property_name: str):
        if property_name not in self._config.keys():
            return None
        return self._config[property_name]


class PathConfig(ModelConfig):

    def __init__(self, conf_dict, model, instance=None, prefix=""):
        ModelConfig.__init__(self, conf_dict)
        self.model = model
        self.instance = instance
        self.prefix = prefix

    @property
    def directory(self):
        folder = os.path.join(
            self.prefix,
            self.get_property("path")).format(
                model=self.model, instance=self.instance)
        return folder

    @property
    def filename(self):
        return self.get_property("name")

    @property
    def fqfn(self):
        return os.path.join(
            self.directory,
            self.filename).format(
                model=self.model, instance=self.instance)


class InConfig(ModelConfig):

    @property
    def bucket(self):
        return self.get_property("bucket_name")

    @property
    def model_name(self):
        return self.get_property("model_name")

    @property
    def instance_name(self):
        return self.get_property("instance_name")

    @property
    def saved_model(self):
        return PathConfig(
            self.get_property("saved_model"),
            self.model_name, self.instance_name)

    @property
    def variables(self):
        return PathConfig(
            self.get_property("variables"),
            self.model_name, self.instance_name)

    @property
    def vocab(self):
        return PathConfig(self.get_property("vocab"), self.model_name)


# OUT = {
#     "base_dir": os.path.join(os.getcwd(), "classifier_models"),
#     "model_name": "UPR_2percent_ps0",
#     "instance_name": "1573031002",
#     "saved_model": {
#         "name": "saved_model.pb",
#         "path": "multilabel/{model}/saved_model/{instance}"
#     },
#     "variables": {
#         "path": "multilabel/{model}/saved_model/{instance}/variables"
#     },
#     "vocab": {
#         "name": "label.vocab",
#         # NOTE: We store the vocab file inside an instance folder
#         # in case it changes between training runs.
#         "path": "multilabel/{model}/saved_model/{instance}"
#     },
# }


class OutConfig(ModelConfig):

    @property
    def base_dir(self):
        return self.get_property("base_dir")

    @property
    def model_name(self):
        return self.get_property("model_name")

    @property
    def instance_name(self):
        return self.get_property("instance_name")

    @property
    def saved_model(self):
        return PathConfig(
            self.get_property("saved_model"),
            self.get_property("model_name"),
            instance=self.get_property("instance_name"),
            prefix=self.base_dir)

    @property
    def variables(self):
        return PathConfig(
            self.get_property("variables"),
            self.model_name,
            instance=self.instance_name,
            prefix=self.base_dir)

    @property
    def vocab(self):
        return PathConfig(
            self.get_property("vocab"),
            self.model_name,
            instance=self.instance_name,
            prefix=self.base_dir)
