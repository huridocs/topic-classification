import json
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
        status = json.loads(resp.data)['status']
        print(status)
        assert 'Failed' not in status
        if 'Done' in status:
            break


def test_e2e(app: Flask, fs: FakeFilesystem) -> None:
    seq_pattern = (
        'react more swiftly to comply with international instruments %d')
    instance_path = './testdata/test_model/test_instance'
    fs.add_real_directory(instance_path)
    fs.remove_object('./testdata/test_model/test_instance/quality.json')
    fs.remove_object('./testdata/test_model/test_instance/thresholds.json')
    client = app.test_client()
    with app.test_request_context():
        # initial classify returns no topics because we deleted thresholds.json.
        resp = client.post('/classify?model=test_model',
                           data=json.dumps({'seqs': [seq_pattern % 1]}),
                           content_type='application/json')
        assert resp.status == '200 OK'
        assert len(json.loads(resp.data)[0]) == 0

        # now we add training labels
        assert client.put(
            '/classification_sample?model=test_model',
            data=json.dumps({
                'samples': [{
                    'seq': seq_pattern % i,
                    'training_labels': [{
                        'topic': 'International instruments'
                    }]
                } for i in range(20)]
            }),
            content_type='application/json').status == '200 OK'

        resp = client.get('/classification_sample?model=test_model&seq=*')
        assert resp.status == '200 OK'
        assert len(json.loads(resp.data)) == 20

        assert client.post('/task',
                           data=json.dumps({
                               'provider': 'RefreshThresholds',
                               'name': 'thres',
                               'model': 'test_model'
                           }),
                           content_type='application/json').status == '200 OK'
        wait_for_task(client, 'thres')

        for i in range(20):
            resp = client.post('/classify?model=test_model',
                               data=json.dumps({'seqs': [seq_pattern % i]}),
                               content_type='application/json')
            assert resp.status == '200 OK'
            assert len(json.loads(resp.data)[0]) == 1

        assert client.post('/task',
                           data=json.dumps({
                               'provider': 'RefreshPredictions',
                               'name': 'pred',
                               'model': 'test_model'
                           }),
                           content_type='application/json').status == '200 OK'
        wait_for_task(client, 'pred')

        resp = client.get('/classification_sample?model=test_model&seq=*')
        assert resp.status == '200 OK'
        data = json.loads(resp.data)
        assert len(data) == 20
        assert data[0]['predicted_labels'][0][
            'topic'] == 'International instruments'
        assert data[0]['predicted_labels'][0]['quality'] >= 0.5
