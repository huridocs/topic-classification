import os

from flask import Flask

from app import config


CONFIG = {
    "development": "app.config.DevelopmentConfig",
    "test": "app.config.TestConfig",
    "production": "app.config.ProductionConfig",
    "default": "app.config.DevelopmentConfig",
}

app = Flask(__name__)

# initialize configuration values
config_name = os.getenv("FLASK_ENV", 'default')
app.logger.debug("Reading " + config_name + " configuration...")

config_obj = CONFIG[config_name]
app.logger.debug("config obj: " + config_obj)
app.config.from_object(config_obj)

if app.config.from_pyfile('config.py'):
    app.logger.info("Successfully read instance configuration file.")
app.logger.info("Flask config:")
app.logger.info(app.config.items())

from app import routes  # nopep8
from app import task_routes  # nopep8
