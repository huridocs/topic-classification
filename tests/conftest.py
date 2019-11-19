import pytest

from app import create_app

pytest_plugins = ("pyfakefs",)


@pytest.fixture
def app():
    app = create_app()
    app.config.from_pyfile("../tests/flask_test_cfg.py")
    return app


def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item


def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail(
                "previous test failed ({})".format(previousfailed.name))
