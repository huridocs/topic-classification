import json
import pytest


def test_classify(app):
    client = app.test_client()

    with app.test_request_context():
        resp = client.post(
            '/classify',
            data=json.dumps(dict(seq='hello world!')),
            content_type='application/json')
    assert resp.status == '200 OK'


def test_embed(app):
    client = app.test_client()

    with app.test_request_context():
        resp = client.post(
            '/embed',
            data=json.dumps(dict(seq='hello world!')),
            content_type='application/json')
    assert resp.status == '200 OK'
