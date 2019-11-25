import json
import logging
import ntpath
import os
from typing import List, Optional

from google.cloud import storage

from app import model_config as mc


class Fetcher(object):
    """ Fetches classifier model files from Google Cloud Storage."""

    def __init__(self,
                 config_path: Optional[str] = None,
                 src_config: Optional[mc.InConfig] = None,
                 dest_config: Optional[mc.OutConfig] = None,
                 require_instance_config: bool = True):
        self.logger = logging.getLogger()
        self.require_instance_config = require_instance_config
        if src_config and dest_config:
            self.src_config = src_config
            self.dst_config = dest_config
        elif config_path is not None:
            with open(config_path) as f:
                data = json.loads(f.read())
                self.src_config = mc.InConfig(
                    data['source'],
                    model_name=data['model_name'],
                    instance_name=data['instance_name'])
                self.dst_config = mc.OutConfig(
                    data['destination'],
                    model_name=data['model_name'],
                    instance_name=data['instance_name'])

        self.client = storage.Client.from_service_account_json(
            self.src_config.google_acct_key_path)
        self.bucket = self.client.get_bucket(self.src_config.bucket)
        os.makedirs(self.dst_config.base_dir, exist_ok=True)

    def _fetch(self, src_blob: str, dest_fqfn: str) -> List[str]:
        blob = self.bucket.blob(src_blob)

        dest_file = dest_fqfn
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)

        blob.download_to_filename(dest_file)
        self.logger.info('Blob {} downloaded to {}.'.format(
            src_blob, dest_file))

        return [dest_file]

    def fetchVocab(self) -> List[str]:
        return self._fetch(self.src_config.vocab.fqfn,
                           self.dst_config.vocab.fqfn)

    def fetchModel(self) -> List[str]:
        return self._fetch(self.src_config.saved_model.fqfn,
                           self.dst_config.saved_model.fqfn)

    def fetchInstanceConfig(self) -> List[str]:
        result = ['']
        try:
            result = self._fetch(self.src_config.instance_config.fqfn,
                                 self.dst_config.instance_config.fqfn)
        except Exception:
            if self.require_instance_config:
                raise Exception('Expected an instance config JSON file %s in '
                                'the source directory, and it was not found.' %
                                (self.src_config.instance_config.fqfn))
        return result

    def fetchVariables(self) -> List[str]:
        var_blob_dir = self.src_config.variables.directory + '/'
        blobs = list(self.bucket.list_blobs(prefix=var_blob_dir, delimiter='/'))

        new_files: List[str] = []
        for b in blobs:
            blob = self.bucket.blob(b.name)

            # Note: github.com/googleapis/google-cloud-python/issues/5163
            if blob.name == var_blob_dir:
                self.logger.info('Discarding blob with name equal to '
                                 'the directory prefix: %s' % b.name)
                continue

            dest_var_fqfn = os.path.join(self.dst_config.variables.directory,
                                         ntpath.basename(b.name))
            new_files += self._fetch(b.name, dest_var_fqfn)

        return new_files

    def fetchAll(self) -> List[str]:
        return (['=====Config====='] + self.fetchInstanceConfig() +
                ['=====Model====='] + self.fetchModel() + ['=====Label====='] +
                self.fetchVocab() + ['===Variables==='] + self.fetchVariables())
