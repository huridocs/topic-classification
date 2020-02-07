import json
from typing import Any

import requests

from app import tasks

# from flask import Blueprint, jsonify, request
# uwazi_bp = Blueprint('uwazi_bp', __name__)


class _SyncPredictionsTask(tasks.TaskProvider):

    def __init__(self, json: Any):
        super().__init__(json)
        self.uwazi_url = (json['uwazi_url']
                          if 'uwazi_url' in json else json['remote_url'])

    def Run(self, status_holder: tasks.StatusHolder) -> None:
        status_holder.status = 'Loading thesauri from ' + self.uwazi_url
        templates = {
            t['_id']: t
            for t in requests.get('%s/api/templates' %
                                  self.uwazi_url).json()['rows']
        }
        thesauri = [
            t for t in requests.get('%s/api/thesauris' %
                                    self.uwazi_url).json()['rows']
            if 'enable_classification' in t and t['enable_classification']
        ]
        response = {
            'uwazi_url': self.uwazi_url,
        }
        for thesaurus in thesauri:
            tmpl_props = [
                p for t in templates.values() for p in t['properties']
                if 'content' in p and thesaurus['_id'] == p['content']
            ]
            if not tmpl_props:
                continue
            prop_name = tmpl_props[0]['name']
            filters = json.dumps({
                '_' + prop_name: dict(values=['missing']),
                prop_name: dict(values=['missing'])
            })
            print(filters)
            sharedIds = [
                r['sharedId']
                for r in requests.get('%s/api/search' % self.uwazi_url,
                                      params=dict(filters=filters,
                                                  includeUnpublished='true',
                                                  select='["sharedId"]'),
                                      cookies={
                                          'connect.sid': ''
                                      }).json()['rows']
            ]
            response[
                thesaurus['name']] = tmpl_props[0]['name'] + ' ' + sharedIds[0]
        status_holder.status = json.dumps(response)


tasks.providers['SyncPredictions'] = _SyncPredictionsTask
