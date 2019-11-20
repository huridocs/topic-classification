import json
import time

import pytest
from flask import Flask


@pytest.mark.incremental  # Make it obvious which request is problematic if fail
class TestWait1:

    def test_task_not_found(self, app: Flask) -> None:
        client = app.test_client()
        with app.test_request_context():
            resp = client.get('/task',
                              data=json.dumps({'name': 'wait1'}),
                              content_type='application/json')
        assert resp.status_code == 404

    def test_create_new_task(self, app: Flask) -> None:
        client = app.test_client()
        with app.test_request_context():
            resp = client.post('/task',
                               data=json.dumps({
                                   'provider': 'Wait',
                                   'name': 'wait1',
                                   'time': 0.2
                               }),
                               content_type='application/json')
        assert resp.status_code == 200
        data = json.loads(resp.get_data(as_text=True))
        assert data['status'] == 'Started'

    def test_get_new_task(self, app: Flask) -> None:
        client = app.test_client()
        time.sleep(0.1)
        with app.test_request_context():
            resp = client.get('/task',
                              data=json.dumps({'name': 'wait1'}),
                              content_type='application/json')
        assert resp.status_code == 200
        data = json.loads(resp.get_data(as_text=True))
        assert 'Waited for' in data['status']

    def test_delete_task(self, app: Flask) -> None:
        client = app.test_client()
        with app.test_request_context():
            resp = client.delete('/task',
                                 data=json.dumps({'name': 'wait1'}),
                                 content_type='application/json')
        assert resp.status_code == 200

    def test_get_done_task(self, app: Flask) -> None:
        client = app.test_client()
        time.sleep(0.1)
        with app.test_request_context():
            resp = client.get('/task',
                              data=json.dumps({'name': 'wait1'}),
                              content_type='application/json')
        assert resp.status_code == 200
        data = json.loads(resp.get_data(as_text=True))
        assert data['status'] == 'Done (Cancelled)'
