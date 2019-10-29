from app import app
import unittest
import json
import time


class TestRoutes(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_wait1(self):
        resp = self.client.get(
            '/task', data=json.dumps({"name": "wait1"}), content_type='application/json')
        self.assertEqual(resp.status_code, 404)

        resp = self.client.post(
            '/task',
            data=json.dumps({"provider": "Wait", "name": "wait1", "time": 0.2}),
            content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.get_data(as_text=True))
        self.assertEqual(data['status'], 'Started')
        time.sleep(0.1)

        resp = self.client.get(
            '/task', data=json.dumps({"name": "wait1"}), content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.get_data(as_text=True))
        assert 'Waited for' in data['status']

        resp = self.client.delete(
            '/task', data=json.dumps({"name": "wait1"}), content_type='application/json')
        self.assertEqual(resp.status_code, 200)

        time.sleep(0.1)

        resp = self.client.get(
            '/task', data=json.dumps({"name": "wait1"}), content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.get_data(as_text=True))
        self.assertEqual(data['status'], 'Done (Cancelled)')


if __name__ == '__main__':
    unittest.main()
