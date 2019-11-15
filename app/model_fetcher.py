import json
import logging
import ntpath
import os

from flask import current_app as app
from google.cloud import storage

from app import model_config as mc


class Fetcher(object):
    """ Fetches classifier model files from Google Cloud Storage."""

    def __init__(self,
                 config_path="./static/model_config.json",
                 src_config=None, dest_config=None):
        self.logger = logging.getLogger("app.logger")
        self.client = storage.Client()
        if src_config and dest_config:
            self.src_config = src_config
            self.dst_config = dest_config
        else:
            with open(config_path) as f:
                data = json.loads(f.read())
                self.src_config = mc.InConfig(f["in"])
                self.dst_config = mc.OutConfig(f["out"])

        self.bucket = self.client.get_bucket(self.src_config.bucket)
        os.makedirs(self.dst_config.base_dir, exist_ok=True)

    def _fetch(self, src_blob, dest_fqfn: str) -> [str]:
        blob = self.bucket.blob(src_blob)

        dest_file = dest_fqfn
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)

        blob.download_to_filename(dest_file)
        self.logger.info('Blob {} downloaded to {}.'.format(
            src_blob, dest_file))

        return [dest_file]

    def fetchVocab(self) -> [str]:
        return self._fetch(self.src_config.vocab.fqfn,
                           self.dst_config.vocab.fqfn)

    def fetchModel(self) -> [str]:
        return self._fetch(self.src_config.saved_model.fqfn,
                           self.dst_config.saved_model.fqfn)

    def fetchVariables(self) -> [str]:
        var_blob_dir = self.src_config.variables.directory + "/"
        blobs = list(self.bucket.list_blobs(
            prefix=var_blob_dir, delimiter="/"))

        new_files = []
        for b in blobs:
            blob = self.bucket.blob(b.name)

            # Note: github.com/googleapis/google-cloud-python/issues/5163
            if blob.name == var_blob_dir:
                self.logger.info(
                    "Discarding blob with name equal to "
                    "the directory prefix: %s" % b.name)
                continue

            dest_var_fqfn = os.path.join(
                self.dst_config.variables.directory,
                ntpath.basename(b.name))
            new_files += self._fetch(b.name, dest_var_fqfn)

        return new_files

    def fetchAll(self) -> [str]:
        return (
            ["=====Model====="]
            + self.fetchModel()
            + ["=====Label====="]
            + self.fetchVocab()
            + ["===Variables==="]
            + self.fetchVariables()
        )
