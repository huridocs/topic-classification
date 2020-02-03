import logging
import os
import re
from typing import Any, Dict, List, Optional

from flask import Blueprint
from flask import current_app as app
from flask import jsonify, request

from app.classifier import Classifier, TopicInfo

model_status_bp = Blueprint('model_status_bp', __name__)


class ModelStatus:

    class TopicStatus:

        def __init__(self, topic_name: str, topic_infos: TopicInfo):
            self.topic_name = topic_name
            self.info = topic_infos

    def __init__(self, base_classifier_dir: str, model_name: str = '') -> None:
        self.logger = logging.getLogger()
        self.base_classifier_dir = base_classifier_dir
        self.model_name = model_name
        self.classifier: Optional[Classifier] = None
        if self.model_name:
            try:
                self.classifier = Classifier(self.base_classifier_dir,
                                             model_name)
                self.topic_infos: Dict[str, ModelStatus.TopicStatus] = {}
                for t, ti in self.classifier.topic_infos.items():
                    self.topic_infos[t] = ModelStatus.TopicStatus(t, ti)
            except Exception:
                self.logger.info(
                    'No model %s found in classifier directory=%s' %
                    (model_name, self.base_classifier_dir))

    def list_potential_models(self, filter_str: str = '.*') -> List[str]:
        filter_exp = re.compile(filter_str)
        return sorted([
            model for model in os.listdir(self.base_classifier_dir)
            if filter_exp.match(model)
        ])

    def list_model_instances(self) -> List[str]:
        try:
            model_dir = os.path.join(self.base_classifier_dir, self.model_name)
            return sorted(os.listdir(model_dir))
        except Exception:
            self.logger.info('No model %s found in classifier directory=%s' %
                             (self.model_name, self.base_classifier_dir))
            return []

    def get_preferred_model_instance(self) -> str:
        if not self.classifier:
            return ''
        try:
            if self.classifier is not None:
                return self.classifier.instance if self.model_name else ''
        except Exception:
            self.logger.info('No preferred instance found for model %s' %
                             self.model_name)
        return ''

    def get_bert(self) -> str:
        if not self.classifier:
            return ''
        try:
            if self.classifier is not None:
                return self.classifier.instance_config.bert if (
                    self.model_name) else ''
        except Exception:
            self.logger.info(
                'No preferred instance with BERT specified found for model %s' %
                self.model_name)
        return ''

    def _build_status_dict(self) -> Dict[str, Any]:
        bert = self.get_bert()
        instances = self.list_model_instances()
        if not instances or self.classifier is None:
            return {
                'name': self.model_name,
                'error': 'Invalid model name %s' % self.model_name
            }
        preferred = self.get_preferred_model_instance()
        topics = {}
        quality_at_precision: Dict[str, Any] = {}
        if self.classifier:
            quality_at_precision = self.classifier.quality_infos.get(
                app.config['DESIRED_CLASSIFIER_PRECISION'], {})
            for t, ti in self.classifier.topic_infos.items():
                topics[t] = {
                    'name':
                        t,
                    'samples':
                        ti.num_samples,
                    'quality':
                        ti.recalls.get(
                            app.config['DESIRED_CLASSIFIER_PRECISION'], 0.0),
                }
        return {
            'name': self.model_name,
            'instances': instances,
            'preferred': preferred,
            'completeness': quality_at_precision.get('completeness', 0.0),
            'extraneous': quality_at_precision.get('extra', 0.0),
            'bert': bert,
            'topics': topics
        }


@model_status_bp.route('/models', methods=['GET'])
def getModel() -> Any:
    # request.get_json: {
    #     'model': 'UPR_2percent_ps0'
    # }
    args = request.args

    model = args.get('model', default='')
    verbose = args.get('verbose', default=True)

    status = ModelStatus(app.config['BASE_CLASSIFIER_DIR'])
    models = status.list_potential_models()
    if not verbose:
        return jsonify(models=models)

    if model:
        status = ModelStatus(app.config['BASE_CLASSIFIER_DIR'],
                             model_name=model)
        return jsonify(status._build_status_dict())

    # TODO: Add typesjson to fix type problems in the results dict.
    results: Dict[str, Any] = {}
    for m in models:
        status = ModelStatus(app.config['BASE_CLASSIFIER_DIR'], model_name=m)
        results[m] = status._build_status_dict()
    return jsonify(results)


@model_status_bp.route('/models/list', methods=['GET'])
def listModels() -> Any:
    # request.get_json: {
    #     'filter': 'upr-topics'
    # }
    args = request.args

    model_filter = args.get('filter', default='.*')

    status = ModelStatus(app.config['BASE_CLASSIFIER_DIR'])
    models = status.list_potential_models(model_filter)
    return jsonify(models=models)
