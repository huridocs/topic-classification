import logging
import os

from flask import Blueprint, Flask

errors = Blueprint('errors', __name__)

# @errors.app_errorhandler(Exception)
# def handle_generic_error(error: Any) -> Any:
#     status_code = 500
#     response = {
#         'success': False,
#         'error': '%s: %s' % (str(type(error)), str(error))
#     }
#     return jsonify(response), status_code


def create_app() -> Flask:
    app = Flask(__name__)

    # initialize configuration values
    config_name = os.getenv('FLASK_ENV', 'default')
    app.logger.debug('Reading ' + config_name + ' configuration...')
    app.logger.debug('App config:')

    # hard-code some configuration
    app.config['BASE_CLASSIFIER_DIR'] = './classifier_models'
    for k, v in app.config.items():
        app.logger.debug('%s: %s' % (k, v))

    # configure logging
    if not app.debug:
        app.logger.setLevel(logging.INFO)

    with app.app_context():
        # Include our Routes
        from . import classifier
        from . import embedder
        from . import model_status
        from . import task_routes
        # from . import uwazi

        # Register Blueprints
        app.register_blueprint(classifier.classify_bp)
        app.register_blueprint(embedder.embed_bp)
        app.register_blueprint(model_status.model_status_bp)
        app.register_blueprint(task_routes.task_bp)
        # app.register_blueprint(uwazi.uwazi_bp)
        app.register_blueprint(errors)

        return app
