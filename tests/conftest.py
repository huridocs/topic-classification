import pytest
from typing import Any

from flask import Flask

from app import create_app


@pytest.fixture
def app() -> Flask:
    app = create_app()
    app.config.from_pyfile("../tests/flask_test_cfg.py")
    return app


def pytest_runtest_makereport(item: Any, call: Any) -> None:
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item


def pytest_runtest_setup(item: Any) -> None:
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail(
                "previous test failed ({})".format(previousfailed.name))
