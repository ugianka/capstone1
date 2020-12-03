#!/usr/bin/env python
"""
api tests

these tests use the requests package however similar requests can be made with curl

e.g.

data = '{"key":"value"}'
curl -X POST -H "Content-Type: application/json" -d "%s" http://localhost:8080/predict'%(data)
"""

import sys
import os
import unittest
import requests
import re
from ast import literal_eval
import numpy as np
import json
import datetime as dt

port = 5000

try:
    requests.post('http://127.0.0.1:{}/predict'.format(port))
    server_available = True
except:
    server_available = False

# test class for the main window function


class ApiTest(unittest.TestCase):
    """
    test the essential functionality
    """

    @unittest.skipUnless(server_available, "local server is not running")
    def test_01_train(self):
        """
        test the train functionality
        """

        # request_json = {'mode': 'test'}
        r = requests.get('http://127.0.0.1:{}/train?env=test'.format(port))
        # print('train API result:', r.text)
        jresp = json.loads(r.text)
        rc = jresp['result']
        self.assertEqual(rc, 0)

    @unittest.skipUnless(server_available, "local server is not running")
    def test_02_predict_empty(self):
        """
        check that predict with no input returns error
        """
        # provide no data at all
        r = requests.get('http://127.0.0.1:{}/predict'.format(port))
        self.assertEqual(r.status_code, 200)
        # print('predict no data response: ', r.text)
        jresp = json.loads(r.text)
        rc = jresp['result']
        self.assertEqual(rc, -1)

    @unittest.skipUnless(server_available, "local server is not running")
    def test_03_predict_not_empty(self):
        """
        check that predict works
        """
        query_string = 'country=united_kingdom&year=2018&month=10&day=20'
        r = requests.get('http://127.0.0.1:{}/predict?{}'.format(port, query_string))
        self.assertEqual(r.status_code, 200)
        print('predict with data: ', r.text)
        jresp = json.loads(r.text)
        rc = jresp['result']
        y_pred = jresp['prediction']
        self.assertEqual(rc, 0)
        self.assertTrue(len(y_pred) > 0)
        self.assertTrue(y_pred[0] >= 0)

    @unittest.skipUnless(server_available, "local server is not running")
    def test_04_logs(self):
        """
        check that logs are returned from log API
        """
        file_name = 'train-test.log'
        request_json = {'file': 'train-test.log'}

        today = dt.datetime.now()
        isodate = today.strftime('%Y-%m-%d')
        query_string = 'type=train&env=test&date={}'.format(isodate)
        r = requests.get('http://127.0.0.1:{}/getLog?{}'.format(port, query_string))
        resj = json.loads(r.text)
        self.assertTrue(resj['result'] >= 0)


# Run the tests
if __name__ == '__main__':
    unittest.main()
