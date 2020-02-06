import json
import os

from flask import Flask
from pyfakefs.fake_filesystem import FakeFilesystem


def test_embed(app: Flask) -> None:
    client = app.test_client()

    with app.test_request_context():
        resp = client.post('/embed',
                           data=json.dumps({
                               'seqs': ['hello world!'],
                               'bert': ('https://tfhub.dev/google/'
                                        'bert_uncased_L-12_H-768_A-12/1')
                           }),
                           content_type='application/json')
    assert resp.status == '200 OK'
    result = json.loads(resp.data)
    assert result
    assert result[0] > 10000


def test_classify(app: Flask, fs: FakeFilesystem) -> None:
    instance_path = os.path.join('./testdata/test_model/test_instance')
    fs.add_real_directory(instance_path)

    client = app.test_client()

    with app.test_request_context():
        resp = client.post(
            '/classify?model=test_model',
            data=json.dumps(
                {'seqs': ['improve access to health care for children']}),
            content_type='application/json')
    assert resp.status == '200 OK'
    result = json.loads(resp.data)
    assert len(result) == 1
    assert result[0]['Right to health'] >= 0.5


def test_all_model_status(app: Flask, fs: FakeFilesystem) -> None:
    fs.add_real_directory('./testdata/test_model/test_instance')
    fs.add_real_directory('./testdata/test_model/test_instance_unreleased')
    fs.add_real_directory('./testdata/test_other_model/test_instance')

    client = app.test_client()

    with app.test_request_context():
        resp = client.get('/models', content_type='application/json')
    assert resp.status == '200 OK'

    result = json.loads(resp.data)
    print(result)

    assert result
    assert len(result.keys()) == 2
    assert set(result.keys()) == set(['test_model', 'test_other_model'])
    assert result['test_other_model']['instances'] == ['test_instance']
    assert result['test_model']['instances'] == [
        'test_instance', 'test_instance_unreleased'
    ]
    assert result['test_model']['preferred'] == 'test_instance'
    assert result['test_model']['completeness'] >= 0.9
    # Pick random test topics to assert
    assert result['test_model']['topics']['Poverty'] == {
        'name': 'Poverty',
        'quality': 0.81,
        'samples': 178
    }
    # Pick topic with not enough samples for threshold optimization
    assert result['test_model']['topics']['nan'] == {
        'name': 'nan',
        'samples': 0,
        'quality': 0.0
    }


def test_list_models(app: Flask, fs: FakeFilesystem) -> None:
    fs.add_real_directory('./testdata/test_model/test_instance')
    fs.add_real_directory('./testdata/test_model/test_instance_unreleased')
    fs.add_real_directory('./testdata/test_other_model/test_instance')

    client = app.test_client()

    with app.test_request_context():
        resp = client.get('/models/list', content_type='application/json')
    assert resp.status == '200 OK'

    result = json.loads(resp.data)

    assert result
    assert len(result.keys()) == 1
    assert set(result['models']) == set(['test_model', 'test_other_model'])


def test_list_models_filtered(app: Flask, fs: FakeFilesystem) -> None:
    fs.add_real_directory('./testdata/test_model/test_instance')
    fs.add_real_directory('./testdata/test_model/test_instance_unreleased')
    fs.add_real_directory('./testdata/test_other_model/test_instance')

    client = app.test_client()

    with app.test_request_context():
        resp = client.get('/models/list?filter=test_other',
                          content_type='application/json')
    assert resp.status == '200 OK'

    result = json.loads(resp.data)

    assert result
    assert len(result.keys()) == 1
    assert set(result['models']) == set(['test_other_model'])


def test_model_status(app: Flask, fs: FakeFilesystem) -> None:
    fs.add_real_directory('./testdata/test_model/test_instance')
    fs.add_real_directory('./testdata/test_model/test_instance_unreleased')

    client = app.test_client()

    with app.test_request_context():
        resp = client.get('/models?model=test_model',
                          content_type='application/json')
    assert resp.status == '200 OK'

    result = json.loads(resp.data)

    assert result
    assert result['instances'] == ['test_instance', 'test_instance_unreleased']
    assert result['preferred'] == 'test_instance'
    # Pick random test topics to assert
    assert result['topics']['Poverty'] == {
        'name': 'Poverty',
        'quality': 0.81,
        'samples': 178
    }
    # Pick topic with not enough samples for threshold optimization
    assert result['topics']['nan'] == {
        'name': 'nan',
        'samples': 0,
        'quality': 0.0
    }
