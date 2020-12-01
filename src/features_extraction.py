import datetime as dt
from itertools import count
import pandas as pd
import numpy as np
from collections import defaultdict
import re
import os

# get the amount of last month revenue given a DataFrame with the data
# and a date


def engineer_features(df, training=True):
    date_type = 'datetime64[D]'
    dates = df['date'].values.astype(date_type).copy()
    previous = [7, 14, 28, 70]  # [7, 14, 21, 28, 35, 42, 49, 56, 63, 70]
    eng_features = defaultdict(list)
    y = np.zeros(dates.size)
    for idx, day in enumerate(dates):
        current = np.datetime64(day, 'D')
        for num in previous:

            end = current
            start = end - np.timedelta64(num, 'D')

            range = np.arange(start, end, dtype=date_type)
            mask = np.in1d(dates, range)
            filtered = df[mask]
            eng_features["previous_{}".format(num)].append(
                filtered['revenues'].sum())
        # get get the target revenue
        plus_30 = current + np.timedelta64(30, 'D')
        mask = np.in1d(dates, np.arange(
            current, plus_30, dtype='datetime64[D]'))
        y[idx] = df[mask]['revenues'].sum()

        # attempt to capture monthly trend with previous years data (if present)
        start_date = current - np.timedelta64(365, 'D')
        stop_date = plus_30 - np.timedelta64(365, 'D')
        mask = np.in1d(dates, np.arange(
            start_date, stop_date, dtype='datetime64[D]'))
        eng_features['previous_year'].append(df[mask]['revenues'].sum())

        # add some non-revenue features
        minus_30 = current - np.timedelta64(30, 'D')
        mask = np.in1d(dates, np.arange(
            minus_30, current, dtype='datetime64[D]'))
        eng_features['recent_invoices'].append(
            df[mask]['invoices'].mean())
        eng_features['recent_views'].append(df[mask]['views'].mean())
    X = pd.DataFrame(eng_features)
    # combine features in to df and remove rows with all zeros
    X.fillna(0, inplace=True)
    mask = X.sum(axis=1) > 0
    X = X[mask]
    y = y[mask]
    dates = dates[mask]
    X.reset_index(drop=True, inplace=True)

    if training == True:
        # remove the last 30 days (because the target is not reliable)
        mask = np.arange(X.shape[0]) < np.arange(X.shape[0])[-30]
        X = X[mask]
        y = y[mask]
        dates = dates[mask]
        X.reset_index(drop=True, inplace=True)

    return(X, y, dates)


# df = pd.read_csv('./work-data/ts-data-United Kingdom.csv')
# X, y, dates = engineer_features(df)
# for i, d in enumerate(dates):
#     print(d, X.iloc[i:i+1, :], y[i])
