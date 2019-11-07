import logging
import ntpath
import os

from google.cloud import storage


BUCKET = "bert_classification_models"

MODEL_PREFIX_PATH = "multilabel/{model_name}/saved_model"
INSTANCE_PREFIX_PATH = MODEL_PREFIX_PATH + "/{training_instance}"
VARIABLES_PREFIX_PATH = INSTANCE_PREFIX_PATH + "/variables"

LABEL_BLOB_FORMAT = MODEL_PREFIX_PATH + "/label.vocab"
MODEL_BLOB_FORMAT = INSTANCE_PREFIX_PATH + "/saved_model.pb"


class Fetcher(object):
    def __init__(
            self,
            bucket=BUCKET,
            local_dir="./tmp",
            model_name="UPR_2percent_ps512",
            training_instance="1572868595"):
        # Fetch classifier files from Google Cloud Storage
        self.logger = logging.getLogger("app.logger")
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(bucket)
        self.local_dir = local_dir
        self.model_name = model_name
        self.training_instance = training_instance

        os.makedirs(self.local_dir, exist_ok=True)

    def fetchLabel(self):
        source_blob_name = LABEL_BLOB_FORMAT.format(model_name=self.model_name)
        blob = self.bucket.blob(source_blob_name)

        destination_file_name = os.path.join(self.local_dir, source_blob_name)
        local_dest_path = os.path.split(destination_file_name)[0]
        self.logger.info("local path=%s" % local_dest_path)
        os.makedirs(local_dest_path, exist_ok=True)

        blob.download_to_filename(destination_file_name)

        self.logger.info('Blob {} downloaded to {}.'.format(
            source_blob_name, destination_file_name))

    def fetchModel(self):
        source_blob_name = MODEL_BLOB_FORMAT.format(
            model_name=self.model_name, training_instance=self.training_instance)
        blob = self.bucket.blob(source_blob_name)

        destination_file_name = os.path.join(self.local_dir, source_blob_name)
        local_dest_path = os.path.split(destination_file_name)[0]
        os.makedirs(local_dest_path, exist_ok=True)

        blob.download_to_filename(destination_file_name)

        self.logger.info('Blob {} downloaded to {}.'.format(
            source_blob_name, destination_file_name))

    def fetchVariables(self):
        prefix = VARIABLES_PREFIX_PATH.format(
            model_name=self.model_name, training_instance=self.training_instance)
        local_prefix = os.path.join(self.local_dir, prefix)

        os.makedirs(local_prefix, exist_ok=True)
        blobs = list(self.bucket.list_blobs(prefix=prefix + "/", delimiter="/"))
        for b in blobs:
            blob = self.bucket.blob(b.name)
            if blob.name == prefix + "/":
                self.logger.info(
                    "Discarding blob with name equal to "
                    "the directory prefix: %s" % b.name)
                continue
            self.logger.info("blob name=%s" % b.name)

            destination_file_name = os.path.join(
                local_prefix, ntpath.basename(b.name))
            self.logger.info(
                "Downloading variables file to " + destination_file_name)
            blob.download_to_filename(destination_file_name)
            self.logger.info('Blob {} downloaded to {}.'.format(
                 blob.name, destination_file_name))
        return [b.name for b in blobs]
