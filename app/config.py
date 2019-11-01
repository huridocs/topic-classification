"""This configuration is read at Flask initialization,
and conditioned on the FLASK_ENV environment variable
(usually configured in ../.flaskenv)."""


class Config(object):
    DEBUG = False
    TESTING = False


class ProductionConfig(Config):
    DATABASE_URI = 'mongodb://prod'


class DevelopmentConfig(Config):
    DATABASE_URI = 'mongodb://local'
    DEBUG = False


class TestConfig(Config):
    DATABASE_URI = 'mongodb://test'
    TESTING = True
