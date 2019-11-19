import json
import pytest

from flask import Flask


def test_classify(app: Flask) -> None:
    client = app.test_client()

    with app.test_request_context():
        data = {'model': 'test_model'}
        resp = client.post(
            '/classify?model=test_model',
            data=json.dumps({'seqs': ['hello world!']}),
            content_type='application/json')
    assert resp.status == '200 OK'


def test_embed(app: Flask) -> None:
    client = app.test_client()

    with app.test_request_context():
        resp = client.post(
            '/embed',
            data=json.dumps(
                {'seqs': ['hello world!'],
                 'bert': ('https://tfhub.dev/google/'
                          'bert_uncased_L-12_H-768_A-12/1')}),
            content_type='application/json')
    assert resp.status == '200 OK'
