import json
import pytest

from app import create_app


def test_hello(app):
    client = app.test_client()

    with app.test_request_context():
        resp = client.get('/')
    assert resp.status == '200 OK'

    data = resp.data
    assert data == b"Hello, World!"
