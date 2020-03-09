import json
import os
import shutil
import time

from flask import Flask
from flask.testing import FlaskClient
from pyfakefs.fake_filesystem import FakeFilesystem


def wait_for_task(client: FlaskClient, name: str) -> None:
    while True:
        time.sleep(0.1)
        resp = client.get('/task',
                          data=json.dumps({'name': name}),
                          content_type='application/json')
        assert resp.status == '200 OK'
        state = json.loads(resp.data)['state']
        assert state != 'failed'
        if state == 'done':
            break


def test_e2e(app: Flask, fs: FakeFilesystem) -> None:
    fs.pause()
    app.config['BASE_CLASSIFIER_DIR'] = '/tmp/tctest/testdata'
    if os.path.exists(app.config['BASE_CLASSIFIER_DIR']):
        shutil.rmtree(app.config['BASE_CLASSIFIER_DIR'])
    os.makedirs(app.config['BASE_CLASSIFIER_DIR'], exist_ok=True)
    seq_pattern = ('Continue working for the eradiction of poverty %d')
    client = app.test_client()
    with app.test_request_context():
        assert client.post('/task',
                           data=json.dumps({
                               'provider':
                                   'TrainModel',
                               'name':
                                   'train-model',
                               'model':
                                   'trained_model',
                               'labels': ['a', 'b', 'c'],
                               'num_train_steps':
                                   10,
                               'train_ratio':
                                   0.5,
                               'samples': [
                                   dict(seq=seq_pattern % i,
                                        training_labels=['a'])
                                   for i in range(0, 30)
                               ]
                           }),
                           content_type='application/json').status == '200 OK'
        wait_for_task(client, 'train-model')

        # without threshold file default confidence is set to 0.3
        resp = client.post('/classify?model=trained_model',
                           data=json.dumps(
                               dict(samples=[dict(seq=seq_pattern % 1)])),
                           content_type='application/json')
        assert resp.status == '200 OK'
        data = json.loads(resp.data)
        print(data)
        assert len(data['samples']) == 1
        assert data['samples'][0]['predicted_labels'][0]['topic'] == 'a'
        assert data['samples'][0]['predicted_labels'][0]['quality'] >= 0.7
