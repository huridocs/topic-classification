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
            try:
                with open(
                        os.path.join(self.base_classifier_dir, model_name, i,
                                    'config.json')) as f:
                        d = json.loads(f.read())
                        if d['is_released']:
                            return i
            except Exception:
                self.logger.info("Instance config wasn\'t loadable.")
        return ''

    def get_bert(self, model_name: str, instance_name: str) -> str:
        # TODO: make this real
        return 'a bert'

    def get_num_training_samples(
            self, model_name: str, instance_name: str) -> int:
        # TODO: make this real
        return 1000

    def get_completeness_score(
            self, model_name: str, instance_name: str) -> float:
        # TODO: make this real
        return 0.99

    def get_extraneous_wrong_values(
            self, model_name: str, instance_name: str) -> float:
        # TODO: make this real
        return 1


@model_status_bp.route('/models', methods=['GET'])
def getModels() -> Any:
    # request.get_json: {
    #     'model': 'UPR_2percent_ps0'
    # }
    args = request.args

    status = ModelStatus(app.config['BASE_CLASSIFIER_DIR'])
    models = status.list_potential_models()
    model = args.get('model', default='')
    verbose = args.get('verbose', default=True)
    if not verbose:
        return jsonify(models=models)

    if model:
        instances = status.list_model_instances(model)
        preferred = status.get_preferred_model_instance(model)

        bert = status.get_bert(model, preferred)
        samples = status.get_num_training_samples(model, preferred)
        completeness = status.get_completeness_score(model, preferred)
        extraneous = status.get_extraneous_wrong_values(model, preferred)
        return jsonify(instances=instances,
                       preferred=preferred,
                       bert=bert,
                       samples=samples,
                       completeness=completeness,
                       extraneous=extraneous)

    results = {}
    # TODO: Add typesjson to fix type problems in the results dict.
    for m in models:
        instances = status.list_model_instances(m)
        preferred = status.get_preferred_model_instance(m)
        results[m] = {
             'instances': instances,
             'preferred': preferred,
        }
        if preferred:
            results[m].update({
                'bert': status.get_bert(m, preferred),
                'samples': status.get_num_training_samples(m, preferred),
                'completeness': status.get_completeness_score(m, preferred),
                'extraneous': status.get_extraneous_wrong_values(m, preferred),
            })
    return jsonify(results)
