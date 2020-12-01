from data_ingestion import getAllTS, ingestTrainData
from features_extraction import engineer_features
from log import update_predict_log, update_train_log
import time
import os
from os.path import join, exists
import re
import csv
import sys
import uuid
import joblib
from datetime import date
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import unittest


import sys
sys.path.insert(1, './src')


# model specific variables (iterate the version and note with each change)
TSDIR = 'work-data'

MODEL_DIR = "models"
MODEL_VERSION = '0.1'
MODEL_VERSION_NOTE = "supervised learing model for time-series"


def _model_train(df, tag, model_dir=None, test=False):
    """
    example funtion to train model

    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file 

    """

    # start timer for runtime
    time_start = time.time()

    X, y, dates = engineer_features(df)

    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]), n_samples,
                                          replace=False).astype(int)
        mask = np.in1d(np.arange(y.size), subset_indices)
        y = y[mask]
        X = X[mask]
        dates = dates[mask]

    # Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        shuffle=True, random_state=42)
    # train a random forest model
    param_grid_rf = {
        'rf__criterion': ['mse', 'mae'],
        'rf__n_estimators': [10, 15, 20, 25]
    }

    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
                              ('rf', RandomForestRegressor())])

    grid = GridSearchCV(pipe_rf, param_grid=param_grid_rf,
                        cv=5, iid=False, n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    eval_rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)))

    # retrain using all data
    grid.fit(X, y)
    model_name = re.sub("\.", "_", str(MODEL_VERSION))
    if test:
        saved_model = os.path.join(model_dir,
                                   "test-{}-{}.joblib".format(tag, model_name))
        print("... saving test version of model: {}".format(saved_model))
    else:
        saved_model = os.path.join(model_dir,
                                   "sl-{}-{}.joblib".format(tag, model_name))
        print("... saving model: {}".format(saved_model))

    joblib.dump(grid, saved_model)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)

    # update log
    update_train_log(tag, (str(dates[0]), str(dates[-1])), {'rmse': eval_rmse}, runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE, test=True)


def model_train(data_dir, test=False, model_dir=None, force_data_load=True):
    """
    funtion to train model given a df

    'mode' -  can be used to subset data essentially simulating a train
    """

    work_dir = os.path.join(data_dir, TSDIR)
    inp_dir = os.path.join(data_dir, 'cs-train')

    if not model_dir:
        model_dir = MODEL_DIR

    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if test:
        print("... test flag on")
        print("...... subseting data")
        print("...... subseting countries")

    idf = pd.DataFrame()
    if(force_data_load):
        print('loading data from ', inp_dir)
        idf = ingestTrainData(inp_dir)
    else:
        train_path = os.path.join(work_dir, 'train-data-cleaned.csv')
        idf = pd.read_csv(train_path)

    # fetch time-series formatted data
    ts_data = getAllTS(idf, work_dir)

    # train a different model for each data sets
    for country, df in ts_data.items():

        if test and country not in ['all', 'united_kingdom']:
            continue

        _model_train(df, country, model_dir=model_dir, test=test)


def model_predict(country, year, month, day, all_models=None, test=False, data_dir=None, model_dir=None):
    """
    example funtion to predict from model
    """

    print('model directory: ', model_dir)

    if not data_dir:
        data_dir = join('.', 'data')

    if not model_dir:
        model_dir = join('.', 'models')

    # start timer for runtime
    time_start = time.time()

    all_models = {}
    all_data = {}

    # load model if needed
    if not all_models:
        all_data, all_models = model_load(
            training=False, data_dir=data_dir, model_dir=model_dir)

    # input checks
    print('all_models.keys', all_models.keys())
    print('country:', country)
    if country not in all_models.keys():
        raise Exception(
            "ERROR (model_predict) - model for country '{}' could not be found".format(country))

    for d in [year, month, day]:
        if re.search("\D", d):
            raise Exception(
                "ERROR (model_predict) - invalid year, month or day")

    # load data
    model = all_models[country]
    data = all_data[country]

    # check date
    target_date = "{}-{}-{}".format(year,
                                    str(month).zfill(2), str(day).zfill(2))
    print(target_date)

    if target_date not in data['dates']:
        raise Exception("ERROR (model_predict) - date {} not in range {}-{}".format(target_date,
                                                                                    data['dates'][0],
                                                                                    data['dates'][-1]))
    date_indx = np.where(data['dates'] == target_date)[0][0]
    query = data['X'].iloc[[date_indx]]

    # sanity check
    if data['dates'].shape[0] != data['X'].shape[0]:
        raise Exception("ERROR (model_predict) - dimensions mismatch")

    # make prediction and gather data for log entry
    y_pred = model.predict(query)
    y_proba = None
    if 'predict_proba' in dir(model) and 'probability' in dir(model):
        if model.probability == True:
            y_proba = model.predict_proba(query)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)

    # update predict log
    update_predict_log(country, y_pred, y_proba, target_date,
                       runtime, MODEL_VERSION, test=test)

    return({'y_pred': y_pred, 'y_proba': y_proba})


def model_load(prefix='sl', data_dir=None, model_dir=None, training=True):
    """
    example funtion to load model

    The prefix allows the loading of different models
    """

    if not data_dir:
        data_dir = os.path.join(".", "data")

    if not model_dir:
        model_dir = os.path.join(".", "models")

    inp_dir = os.path.join(data_dir, "cs-train")
    work_dir = os.path.join(data_dir, "work-data")

    print('seraching models in: '+model_dir)
    models = [f for f in os.listdir(model_dir) if re.search("sl", f)]
    print('models loaded: ', models)

    if len(models) == 0:
        raise Exception(
            "Models with prefix '{}' cannot be found did you train?".format(prefix))

    all_models = {}
    for model in models:
        all_models[re.split("-", model)[1]
                   ] = joblib.load(os.path.join(model_dir, model))

    # load data
    idf = ingestTrainData(inp_dir)
    ts_data = getAllTS(idf, work_dir)
    all_data = {}
    for country, df in ts_data.items():
        X, y, dates = engineer_features(df, training=training)
        dates = np.array([str(d) for d in dates])
        all_data[country] = {"X": X, "y": y, "dates": dates}

    return(all_data, all_models)


if __name__ == "__main__":

    """
    basic test procedure for model.py
    """

    # train the model
    print("TRAINING MODELS")
    data_dir = os.path.join(os.path.join(".", 'data'), "cs-train")
    model_train(data_dir, test=True)
    # train also the production mopdels
    model_train(data_dir, test=False)

    # load the model
    print("LOADING MODELS")
    all_data, all_models = model_load(data_dir=data_dir)
    print("... models loaded: ", ",".join(all_models.keys()))

    # test predict
    country = 'all'
    year = '2018'
    month = '01'
    day = '05'
    result = model_predict(country, year, month, day)
    print(result)


class TestModelTrain(unittest.TestCase):
    def test_file_creation(self):
        data_dir = join('..', 'data')
        work_dir = join(data_dir, 'work-data')
        models_dir = join('..', 'models')
        inpfile = join(work_dir, 'train-data-cleaned.csv')
        force_data_load = True
        if exists(inpfile):
            force_data_load = False
        model_train(data_dir=data_dir, test=True, model_dir=models_dir,
                    force_data_load=force_data_load)
        outfile = join(work_dir, 'test-all-0_1')
        return exists(outfile)


class TestModelPredict(unittest.TestCase):
    def test_result_is_numeric(self):
        data_dir = join('..', 'data')
        models_dir = join('..', 'models')
        work_dir = join(data_dir, 'work-data')
        inpfile = join(work_dir, 'train-data-cleaned.csv')
        force_data_load = True
        if exists(inpfile):
            force_data_load = False

        ukmodelfile = join(models_dir, 'sl-united_kingdom-' +
                           MODEL_VERSION+'.joblib')
        if not exists(ukmodelfile):
            model_train(data_dir=data_dir, model_dir=models_dir, test=False,
                        force_data_load=force_data_load)
        pred = model_predict('united_kingdom', '2018', '1', '1', data_dir=data_dir,
                             model_dir=models_dir, test=True)
        print(pred)
        return pred['y_pred'] > 0
