import json
import os
import pytest


def test_classify(app, fs):
    base_classifier_path = "./testdata"
    instance_path = os.path.join(
        base_classifier_path,
        "test_model",
        "test_instance")
    fs.add_real_directory(instance_path)

    client = app.test_client()

    with app.test_request_context():
        data = {'model': 'test_model'}
        resp = client.post(
            '/classify?model=test_model',
            data=json.dumps({'seq': 'hello world!'}),
            content_type='application/json')
    assert resp.status == '200 OK'


def test_embed(app):
    client = app.test_client()

    with app.test_request_context():
        resp = client.post(
            '/embed',
            data=json.dumps(
                    {'seq': 'hello world!',
                     'bert': ('https://tfhub.dev/google/'
                              'bert_uncased_L-12_H-768_A-12/1')}),
            content_type='application/json')
    assert resp.status == '200 OK'
