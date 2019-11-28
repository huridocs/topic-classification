import json
import logging
import os
from typing import Any, List

from flask import Blueprint
from flask import current_app as app
from flask import jsonify, request

model_status_bp = Blueprint('model_status_bp', __name__)


class ModelStatus:

    def __init__(self, base_classifier_dir: str) -> None:
        self.logger = logging.getLogger()
        self.base_classifier_dir = base_classifier_dir

    def list_potential_models(self) -> List[str]:
        return sorted(os.listdir(self.base_classifier_dir))

    def list_model_instances(self, model_name: str) -> List[str]:
        model_dir = os.path.join(self.base_classifier_dir, model_name)
        return sorted(os.listdir(model_dir))

    def get_preferred_model_instance(self, model_name: str) -> str:
        instances = self.list_model_instances(model_name)
        for i in sorted(instances, reverse=True):
            with open(
                    os.path.join(self.base_classifier_dir, model_name, i,
                                 'config.json')) as f:
                try:
                    d = json.loads(f.read())
                    if d['is_released']:
                        return i
                except Exception:
                    self.logger.info("Instance config wasn\'t loadable.")
        return ''


@model_status_bp.route('/models', methods=['GET'])
def getModels() -> Any:
    # request.get_json: {
    #     'model': 'UPR_2percent_ps0'
    # }
    args = request.args

    status = ModelStatus(app.config['BASE_CLASSIFIER_DIR'])
    models = status.list_potential_models()
    if not args.get('model', default=''):
        return jsonify(models=models)

    instances = status.list_model_instances(args['model'])
    preffered = status.get_preferred_model_instance(args['model'])

    return jsonify(instances=instances,
                   preffered=preffered)
