import fnmatch
import json
import logging
import os
import re
from typing import List, Optional

from google.cloud import storage

from app import model_config as mc


class Fetcher(object):
    """ Fetches classifier model files from Google Cloud Storage."""

    def __init__(self,
                 config_path: str,
                 override_model_name: Optional[str] = None):
        self.logger = logging.getLogger()
        with open(config_path) as f:
            data = json.loads(f.read())
            self.config = mc.FetchConfig(data)
            if override_model_name:
                self.config.set_property('model', override_model_name)

        self.client = storage.Client.from_service_account_json(
            self.config.google_acct_key_path)
        self.bucket = self.client.get_bucket(self.config.bucket_name)

        if not self.config.instance:
            self.config.set_property('instance', self._get_latest_instance())

    def _get_latest_instance(self) -> str:
        latest_instance = ''
        for src_path in self.config.source_files:
            if '{instance}' not in src_path:
                continue
            src_path = src_path.format(model=self.config.model,
                                       instance='{instance}')
            src_path_regex = '^' + src_path.format(instance='[0-9]+') + '$'
            prefix = src_path[:src_path.find('{instance}')]
            for blob in self.bucket.list_blobs(prefix=prefix):
                if not re.match(src_path_regex, blob.name):
                    continue
                suffix = blob.name[len(prefix):]
                suffix_dir = suffix[:suffix.find('/')]
                if suffix_dir > latest_instance:
                    latest_instance = suffix_dir
        if not latest_instance:
            raise Exception('Could not find latest instance dir!')
        self.logger.info('Resolved latest instance of {} to {}'.format(
            self.config.model, latest_instance))
        return latest_instance

    def _fetch(self, src_blob: str, dest_file: str) -> List[str]:
        blob = self.bucket.blob(src_blob)

        os.makedirs(os.path.dirname(dest_file), exist_ok=True)

        try:
            blob.download_to_filename(dest_file)
            self.logger.info('Blob {} downloaded to {}.'.format(
                src_blob, dest_file))
            return [dest_file]
        except IsADirectoryError:
            return []

    def fetchAll(self) -> List[str]:
        new_files: List[str] = []
        dest_dir = self.config.dest_dir.format(model=self.config.model,
                                               instance=self.config.instance)
        for blob in self.bucket.list_blobs():
            for src_path in self.config.source_files:
                src_path = src_path.format(model=self.config.model,
                                           instance=self.config.instance)
                if fnmatch.fnmatch(blob.name, src_path):
                    new_files += self._fetch(
                        blob.name,
                        os.path.join(dest_dir, os.path.basename(blob.name)))
                    break
        for src_path in self.config.source_dirs:
            src_path = src_path.format(model=self.config.model,
                                       instance=self.config.instance)
            for blob in self.bucket.list_blobs(prefix=src_path):
                # Note: github.com/googleapis/google-cloud-python/issues/5163
                if blob.name == src_path + '/':
                    continue
                new_files += self._fetch(
                    blob.name,
                    blob.name.replace(
                        src_path,
                        os.path.join(dest_dir, os.path.basename(src_path))))

        return new_files
