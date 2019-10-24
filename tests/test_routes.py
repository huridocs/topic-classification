from app import app
import unittest
import json


class TestRoutes(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_hello(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status, '200 OK')

        data = resp.data
        self.assertEqual(data, b"Hello, World!")


if __name__ == '__main__':
    unittest.main()
