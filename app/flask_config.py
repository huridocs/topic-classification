"""This configuration is read at Flask initialization, and conditioned on the
FLASK_ENV environment variable (usually configured in ../.flaskenv)."""


class Config(object):
    DEBUG = False
    TESTING = False
    BERT = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    MODEL = "UPR_2percent_ps0"
    INSTANCE = "1573031002"
    GOOGLE_ACCT_KEY_PATH = "./.credz/bert-classification-key.json"
    MODEL_CONFIG_PATH = "./static/model_config.json"


class ProductionConfig(Config):
    DATABASE_URI = 'mongodb://prod'


class DevelopmentConfig(Config):
    DATABASE_URI = 'mongodb://local'
    DEBUG = True


class TestConfig(Config):
    DATABASE_URI = 'mongodb://test'
    TESTING = True
