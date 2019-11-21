from typing import Any

import pytest
from flask import Flask

from app import create_app
from app.classifier import ClassifierCache
from app.models import DATABASE_NAME, datastore

pytest_plugins = ('pyfakefs',)


@pytest.fixture(scope='function', autouse=True)
def clear_all() -> None:
    """Make sure test data does not bleed over to other tests."""
    datastore.db.client.drop_database(DATABASE_NAME)
    ClassifierCache.clear_all()


@pytest.fixture
def app() -> Flask:
    app = create_app()
    app.config.from_pyfile('../tests/flask_test_cfg.py')
    return app


def pytest_runtest_makereport(item: Any, call: Any) -> None:
    if 'incremental' in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item


def pytest_runtest_setup(item: Any) -> None:
    if 'incremental' in item.keywords:
        previousfailed = getattr(item.parent, '_previousfailed', None)
        if previousfailed is not None:
            pytest.xfail('previous test failed ({})'.format(
                previousfailed.name))
