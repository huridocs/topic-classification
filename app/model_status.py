import logging
import os
from typing import Any, Dict, List, Optional

from flask import Blueprint
from flask import current_app as app
from flask import jsonify, request

from app.classifier import Classifier, TopicInfo

PRECISION = 30

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

    def list_potential_models(self) -> List[str]:
        return sorted(os.listdir(self.base_classifier_dir))

    def list_model_instances(self, model_name: str) -> List[str]:
        try:
            model_dir = os.path.join(self.base_classifier_dir, model_name)
            return sorted(os.listdir(model_dir))
        except Exception:
            self.logger.info('No model %s found in classifier directory=%s' %
                             (model_name, self.base_classifier_dir))
            return []

    def get_preferred_model_instance(self) -> str:
        return self.classifier.instance if self.model_name else ''

    def get_bert(self) -> str:
        return self.classifier.instance_config.bert if self.model_name else ''

    def get_num_training_samples(self, topic: str) -> int:
        # TODO: make this real
        return 1000

    def get_completeness_score(self, topic: str) -> float:
        # TODO: make this real
        return 0.99

    def get_extraneous_wrong_values(self, topic: str) -> float:
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
        status = ModelStatus(app.config['BASE_CLASSIFIER_DIR'],
                             model_name=model)
        instances = status.list_model_instances(model)
        if not instances:
            return jsonify(name=model, error='Invalid model name %s' % model)
        preferred = status.get_preferred_model_instance()

        bert = status.get_bert()
        topics = {}
        # TODO: don't refresh on every query
        quality_at_precision = status.classifier.refresh_thresholds()[PRECISION]
        for t, ti in status.classifier.topic_infos.items():
            print(ti.recalls.__dir__())
            print(ti.recalls.__class__)
            topics[t] = {
                'name': t,
                'samples': ti.num_samples,
                'quality': ti.recalls.get(PRECISION, 0.0),
            }
        return jsonify(name=model,
                       instances=instances,
                       preferred=preferred,
                       completeness=quality_at_precision['completeness'],
                       extraneous=quality_at_precision['extra'],
                       bert=bert,
                       topics=topics)

    results: Dict[str, Any] = {}
    # TODO: Add typesjson to fix type problems in the results dict.
    for m in models:
        status = ModelStatus(app.config['BASE_CLASSIFIER_DIR'], model_name=m)
        instances = status.list_model_instances(m)
        preferred = status.get_preferred_model_instance()
        results[m] = {
            'name': m,
            'instances': instances,
            'preferred': preferred,
        }
        if preferred:
            # TODO: don't refresh on every query
            quality_at_precision = (
                status.classifier.refresh_thresholds()[PRECISION])
            topics = {}
            for t, ti in status.classifier.topic_infos.items():
                topics[t] = {
                    'name': t,
                    'samples': ti.num_samples,
                    'completeness':
                        (quality_at_precision['completeness']),
                    'extraneous': quality_at_precision['extra'],
                }

            results[m]['topics'] = topics
    return jsonify(results)
