import threading
import traceback
from typing import Any, Dict, Optional, Type

import requests
from flask import Blueprint, abort
from flask import current_app as app
from flask import jsonify, request

from app import tasks

uwazi_bp = Blueprint('uwazi_bp', __name__)


@uwazi_bp.route('/sync_uwazi', methods=['PUT'])
def sync_uwazi() -> Any:
    data = request.get_json()
    uwazi_url = data['uwazi_url'] if 'uwazi_url' in data else (
        'http://%s:%d' %
        (request.environ['REMOTE_ADDR'], request.environ['REMOTE_PORT']))
    templates = {
        t['_id']: t
        for t in requests.get('%s/api/templates' % uwazi_url).json()['rows']
    }
    thesauri = [
        t for t in requests.get('%s/api/thesauris' % uwazi_url).json()['rows']
        if 'enable_classification' in t and t['enable_classification']
    ]
    response = {
        'uwazi_url': uwazi_url,
    }
    for thesaurus in thesauri:
        tmpl_props = [
            p for t in templates.values() for p in t['properties']
            if 'content' in p and thesaurus['_id'] == p['content']
        ]
        if not tmpl_props:
            continue
        response[thesaurus['name']] = tmpl_props[0]['name']
    return jsonify(response)
