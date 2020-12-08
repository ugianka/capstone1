
from features_extraction import engineer_features
from data_ingestion import getAllTS, ingestTrainData
import os
from os.path import join, exists
import sys
import pandas as pd

import shutil

import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from model import load_data, model_predict
from data_ingestion import getTimeSeries

import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np


sys.path.insert(1, './src')


def trainClf(clf, param_grid_rf):

    data_dir = os.path.join(os.path.join('.', 'data'), 'cs-train')
    work_dir = os.path.join(os.path.join('.', 'data'), 'work-data')

    aDf = ingestTrainData(data_dir)

    data = getAllTS(aDf, work_dir)

    X, y, dates = engineer_features(data['united_kingdom'])

    # Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        shuffle=True, random_state=42)
    # train a random forest model

    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
                              ('clf', clf)])

    grid = GridSearchCV(pipe_rf, param_grid=param_grid_rf,
                        cv=5, iid=False, n_jobs=-1)
    grid.fit(X_train, y_train)
    y_train_pred = grid.predict(X_train)
    y_pred = grid.predict(X_test)
    eval_train_rmse = round(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    eval_rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)))
    print('eval_rmse_test', eval_rmse)
    print('eval_rmse_train', eval_train_rmse)
    print(grid.best_estimator_)
    return grid


def getFeatures():
    # load time series
    work_dir = join(data_dir, 'work-data')

    ts_file_path = join(work_dir, 'ts-data-all.csv')
    df = pd.DataFrame()
    if not exists(ts_file_path):
        # create time series
        idf = ingestTrainData(join('.', 'data'))
        df = getTimeSeries(idf)
    else:
        df = pd.read_csv(ts_file_path)

    X, y, dates = engineer_features(df)
    return X, y, dates


def evaluateModel(model=None, model_name=None, inp_dir=None, work_dir=None, selection_dir=None):

    X, y, dates = getFeatures()

    all_data = load_data(inp_dir=inp_dir, work_dir=work_dir)

    all_models = {
        'all': model
    }

    results = pd.DataFrame(columns=['date', 'y_pred', 'y', 'diff'])
    for idx, d in enumerate(dates):
        date = pd.to_datetime(d)
        error = False
        answ = None
        try:
            answ = model_predict('all', str(date.year), str(date.month), str(date.day), test=False, all_data=all_data, all_models=all_models)
        except:
            print('system error:' + str(sys.exc_info()[1]))
            error = True
        y_pred = None
        diff = None
        yt = y[idx]
        if not error:
            y_pred = answ['y_pred'][0]

            diff = abs(y_pred - yt)

        results = results.append({
            'date': date,
            'y_pred': y_pred,
            'y': yt,
            'diff': diff
        }, ignore_index=True)
        # take only the last dates
    today = dt.datetime.today()

    monname = "{}-results-{}-{}-{}".format(model_name, today.year, today.month, today.day)
    results.to_csv(join(selection_dir, monname + ".csv"))

    rmse = mean_squared_error(results['y'].values, results['y_pred'].values, squared=False)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(15, 8)
    ax.set_title(' error distribution  ')
    sns.distplot(results['diff'], bins=50, color='#008899', ax=ax)
    fig.savefig(join(selection_dir, monname) + '.png', dpi=200)

    statistics_path = join(selection_dir, '{}_statistics.csv'.format(model_name))
    statDF = pd.DataFrame()
    today_iso = today.strftime('%y-%m-%d')

    mse = mean_squared_error(results['y'].values, results['y_pred'].values)
    if exists(statistics_path):
        statDF = pd.read_csv(statistics_path)
        found = statDF[statDF['date'] == today_iso]
        if(found.shape[0] > 0):
            statDF.loc[statDF['date'] == today_iso, ['mse']] = mse
    else:
        statDF = statDF.append(
            {
                'date': today_iso,
                'mse': mse
            }, ignore_index=True
        )
    statDF.to_csv(statistics_path)


if __name__ == "__main__":
    np.random.seed(41)

    data_dir = join('.', 'data')
    model_dir = join('.', 'models')
    selection_dir = join('.', 'selection')
    imp_dir = join(data_dir, 'cs-train')
    work_dir = join(data_dir, 'work-data')

    if not exists(selection_dir):
        os.mkdir(selection_dir)

    randomForestClf = None
    modelpath = join(selection_dir, 'rf.joiblib')
    if exists(modelpath):
        randomForestClf = joblib.load(modelpath)
    else:
        param_grid_rf = {
            'clf__criterion': ['mse', 'mae'],
            'clf__n_estimators': [10, 15, 20, 25]
        }
        print('random_forest')
        randomForestClf = trainClf(clf=RandomForestRegressor(), param_grid_rf=param_grid_rf)
        joblib.dump(randomForestClf, modelpath)

    svmClf = None
    modelpath = join(selection_dir, 'svm.joiblib')
    if exists(modelpath):
        svmClf = joblib.load(modelpath)
    else:
        param_grid_rf = {
            'clf__kernel': ['poly', 'rbf'],
            'clf__degree': [2, 3, 4, 5],
            'clf__epsilon': [0.05, 0.1, 0.2, 0.3]
        }

        svmClf = trainClf(clf=SVR(), param_grid_rf=param_grid_rf)
        joblib.dump(svmClf, modelpath)

    mlpReg = None
    modelpath = join(selection_dir, 'mlp.joiblib')
    if exists(modelpath):
        mlpReg = joblib.load(modelpath)
    else:
        print('mlp')
        param_grid_rf = {
            'clf__hidden_layer_sizes': [[100, 200, 400, 400, 300, 300, 200, 200, 200, 200, 150, 150, 100]],
            'clf__alpha': [0.0001, 0.001],
            'clf__beta_1': [0.6, 0.7, 0.9],
            'clf__beta_2': [0.995, 0.999, 0.9995]
        }
        mlpReg = trainClf(clf=MLPRegressor(), param_grid_rf=param_grid_rf)
        joblib.dump(mlpReg, modelpath)

    models = {
        'random_forest': randomForestClf,
        'svm': svmClf,
        'mlp': mlpReg
    }
    for m in models:
        model = models[m]
        evaluateModel(model=model, model_name=m, inp_dir=imp_dir, work_dir=work_dir, selection_dir=selection_dir)
