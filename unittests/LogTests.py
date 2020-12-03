#!/usr/bin/env python

import os
import sys
import csv
import unittest
from ast import literal_eval
import pandas as pd
import datetime as dt
sys.path.insert(1, os.path.join('..', os.getcwd()))

# import model specific functions and variables
from log import update_train_log, update_predict_log


class LogTest(unittest.TestCase):
    """
    test the essential functionality
    """

    def test_01_train(self):
        """
        ensure log file is created
        """

        today = dt.datetime.today()

        log_file = os.path.join("logs", "train-test-{}-{}-{}.log".format(today.year, today.month, today.day))
        if os.path.exists(log_file):
            os.remove(log_file)

        # update the log
        startDate = '2018-10-1'
        endDate = '2018-10-30'

        update_train_log(country='monaco', val='3.73', startDate=today, endDate=endDate, MODEL_VERSION='0.1', MODEL_VERSION_NOTE='a note', runtime=100.0)

        self.assertTrue(os.path.exists(log_file))

    def test_02_train(self):
        """
        ensure that content can be retrieved from log file
        """

        today = dt.datetime.today()

        # update the log
        log_file = os.path.join("logs", "train-test-{}-{}-{}.log".format(today.year, today.month, today.day))
        if os.path.exists(log_file):
            os.remove(log_file)

        # update the log
        startDate = '2018-10-1'
        endDate = '2018-10-30'

        update_train_log(country="monaco", val='3.73', startDate=startDate, endDate=endDate, MODEL_VERSION='0.1', MODEL_VERSION_NOTE='a note', runtime=100.0)

        df = pd.read_csv(log_file)
        self.assertEqual(df.shape[0], 1)
        sdate = df.iloc[0:1, :]['start_date'].values[0]
        self.assertEqual(sdate, startDate)

    def createPredictLog():
        today = dt.datetime.today()

        update_predict_log('monaco', 12300, 0.65, '2018-02-01',
                           100.0, '0.1', test=test)

    def test_03_predict(self):
        """
        ensure log file is created
        """

        today = dt.datetime.today()
        fname = "pred-test-{}-{}-{}.log".format(today.year, today.month, today.day)
        log_file = os.path.join(".", "logs", fname)
        if os.path.exists(log_file):
            os.remove(log_file)

        # update the log

        update_predict_log('monaco', 12300, 0.65, '2018-02-01',
                           100.0, '0.1', test=True)

        self.assertTrue(os.path.exists(log_file))

    def test_04_predict(self):
        """
        ensure that content can be retrieved from log file
        """

        today = dt.datetime.today()
        fname = "pred-test-{}-{}-{}.log".format(today.year, today.month, today.day)
        log_file = os.path.join(".", "logs", fname)
        if os.path.exists(log_file):
            os.remove(log_file)

        # update the log
        y_pred = 12300

        update_predict_log('monaco', y_pred, 0.65, '2018-02-01',
                           100.0, '0.1', test=True)

        df = pd.read_csv(log_file)
        logged_y_pred = df.loc[0, 'y_pred']
        self.assertEqual(y_pred, logged_y_pred)


# Run the tests
if __name__ == '__main__':
    unittest.main()
