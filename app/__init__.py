import logging
import os

from flask import Flask, current_app

from app import flask_config as config


CONFIG = {
    "development": "app.flask_config.DevelopmentConfig",
    "test": "app.flask_config.TestConfig",
    "production": "app.flask_config.ProductionConfig",
    "default": "app.flask_config.DevelopmentConfig",
}

app = Flask(__name__)

# initialize configuration values
config_name = os.getenv("FLASK_ENV", 'default')
app.logger.debug("Reading " + config_name + " configuration...")

config_obj = CONFIG[config_name]
app.logger.debug("config obj: " + config_obj)
app.config.from_object(config_obj)

if app.config.from_pyfile('flask_config.py'):
    app.logger.info("Successfully read instance configuration file.")
app.logger.info("Flask config:")
app.logger.info(app.config.items())

# configure logging
if not app.debug:
    app.logger.setLevel(logging.INFO)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "/home/samschaevitz/Downloads/BERT Classification-9a8b5ef88627.json")

from app import routes  # nopep8
from app import task_routes  # nopep8
