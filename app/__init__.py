import logging
import os

from flask import Flask, current_app

from app import embedder
from app import classifier
from app import flask_config


def create_app():
    app = Flask(__name__)

    # initialize configuration values
    config_name = os.getenv("FLASK_ENV", 'default')
    app.logger.debug("Reading " + config_name + " configuration...")

    # configure logging
    if not app.debug:
        app.logger.setLevel(logging.INFO)

    with app.app_context():
        # Include our Routes
        from . import classifier
        from . import embedder
        from . import model_fetcher
        from . import task_routes

        # Register Blueprints
        app.register_blueprint(classifier.classify_bp)
        app.register_blueprint(embedder.embed_bp)
        app.register_blueprint(task_routes.task_bp)

        return app
