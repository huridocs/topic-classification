import json
import os

import pytest
from flask import Flask
from pyfakefs.fake_filesystem import FakeFilesystem


def test_classify(app: Flask, fs: FakeFilesystem) -> None:
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
