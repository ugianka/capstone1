
import os
from os import stat
from os.path import join, exists
import pandas as pd
from data_ingestion import getTimeSeries, ingestTrainData
from features_extraction import engineer_features
from model import model_predict, model_load, load_data
import sys
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
# compare the projected revenue with the real revenue


def monitoring():

    # load time series
    data_dir = join('.', 'data')
    model_dir = join('.', 'models')
    monitor_dir = join('.', 'monitor')
    if not exists(monitor_dir):
        os.mkdir(monitor_dir)
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

    all_data, all_models = model_load(
        training=False, data_dir=data_dir, model_dir=model_dir, test=False)

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

    monname = "model-monitoring-{}-{}-{}".format(today.year, today.month, today.day)
    results.to_csv(join(monitor_dir, monname + ".csv"))

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(15, 8)
    ax.set_title('prediction error distribution')
    sns.distplot(results['diff'], bins=50, color='#008899', ax=ax)
    fig.savefig(join(monitor_dir, monname) + '.png', dpi=200)

    statistics_path = join(monitor_dir, 'monitor_statistics.csv')
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


if __name__ == '__main__':
    monitoring()
