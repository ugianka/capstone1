import datetime as dt
import os
from os.path import join
import csv
import uuid


# update log
def update_train_log(startDate, endDate, val, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test=True):
    # name the logfile using something that cycles with date (day, month, year)
    today = dt.datetime.today()
    if test:
        logfile = os.path.join("logs", "train-test.log")
    else:
        logfile = os.path.join(
            "logs", "train-{}-{}.log".format(today.year, today.month))

    # write the data to a csv file
    header = ['unique_id', 'timestamp', 'start_date', 'end_date', 'val,' 'model_version',
              'model_version_note', 'runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str, [uuid.uuid4(), today.time(), startDate, endDate, val,
                             MODEL_VERSION, MODEL_VERSION_NOTE, runtime])
        writer.writerow(to_write)


def update_predict_log(country, y_pred, y_proba, target_date, runtime, MODEL_VERSION, test):
    """
    update predict log file
    """
    today = dt.datetime.now()

    log_dir = join('.', 'logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file_path = join(
        log_dir, 'revenue_pred_-{}-{}-{}.log').format(today.year, today.month, today.day)
    # name the logfile using something that cycles with date (day, month, year)

    # write the data to a csv file
    header = ['uuid', 'country', 'timestamp', 'y_pred',
              'y_proba', 'targetdate', 'model_version', 'runtime']
    write_header = False
    if not os.path.exists(log_file_path):
        write_header = True
    with open(log_file_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        if write_header:
            writer.writerow(header)

        to_write = map(str, [uuid.uuid4(), country, today.time(), y_pred,
                             y_proba, target_date, MODEL_VERSION, runtime])
        writer.writerow(to_write)
