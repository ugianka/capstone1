
import os
import pandas as pd
import re
import math
import numpy as np

# extracts the first integer number encountered in a string
# will be used to clean the stream_id and to remove the ones not
# containing a stream visualization fee (bank transactions etc..)


def extractNumber(astring):
    inpstring = str(astring)
    # if(not inpstring.isnumeric()):
    #     print('nn')
    nums = re.findall('[0-9]+', inpstring)
    if(len(nums) > 0):
        return str(nums[0])
    else:
        return math.nan

# create the Training DataFrame from the training data path passed as
# parameter


def ingestTrainData(data_path):

    # this is a dictionary with the mispelled columns found in some of the months
    # of 2018 and the relative correct ones
    bogus = {
        'StreamID': 'stream_id',
        'TimesViewed': 'times_viewed',
        'total_price': 'price'
    }

    # create a list with all the file path contained in the train data  directory
    onlyfiles = [os.path.join(data_path, f)
                 for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    # the response dataframe
    inpDf = pd.DataFrame()

    # for each of the training Json files paths
    for idx, fname in enumerate(onlyfiles):
        print('loading input file:', fname)

        # read the contents in a dataframe
        # print('reading json from: ', fname)
        df = pd.read_json(fname)

        cols = df.columns

        # replace wrong columns (if present) with the correct ones
        for c in bogus:
            if(c in cols):
                df.rename(columns=bogus, inplace=True)
        # clean streamIds from non numeric characters the ones that have no number inside
        # will be NaN
        df.stream_id = df.stream_id.apply(lambda x: extractNumber(x))

        # remove the rows with stream Nan as those are not relative
        # to pay per view but are bank transactions and other stuff
        df = df[~df.stream_id.isnull()]

        # append the dataframe to the final one
        inpDf = inpDf.append(df)

    # add a column with the full date of trx
    # inpDf['date'] = pd.to_datetime(
    #     (inpDf.year*10000+inpDf.month*100+inpDf.day).apply(str), format='%Y%m%d')
    inpDf['date'] = pd.to_datetime(inpDf[['year', 'month', 'day']])
    return inpDf


def getTimeSeries(inpDataFtame, country=None):
    df = inpDataFtame
    if country:
        df = df[df['country'] == country]
    date_type = 'datetime64[D]'
    mindate = df['date'].values.astype(date_type).min()
    maxdate = df['date'].values.astype(date_type).max()
    dates_range = np.arange(mindate, maxdate, dtype=date_type)
    date_indices = df.date.values.astype(date_type)

    invoices = [np.unique(df[date_indices == day]['invoice']
                          ).size for day in dates_range]
    streams = [np.unique(df[date_indices == day]['stream_id']
                         ).size for day in dates_range]
    purchases = [df[date_indices == day].shape[0] for day in dates_range]
    views = [(df[date_indices == day]['times_viewed']).values.sum()
             for day in dates_range]
    revenues = [(df[date_indices == day]['price']).values.sum()
                for day in dates_range]
    year_months = year_month = [
        "-".join(re.split("-", str(day))[:2]) for day in dates_range]

    return pd.DataFrame({
        'date': dates_range,
        'invoices': invoices,
        'streams': streams,
        'purchases': purchases,
        'views': views,
        'revenues': revenues,
        'year_month': year_months
    })


# takes the original dataframe in input and creates the time series by country
def getAllTS(aDf, output_directory_path):
    # get the top 10 countries by revenue

    if not os.path.exists(output_directory_path):
        os.mkdir(output_directory_path)

    rev_by_country = aDf[['country', 'price']].groupby('country').sum()
    top_countries = rev_by_country.sort_values(
        'price', ascending=False).iloc[0:10, :].index.values

    data = {}
    data['all'] = getTimeSeries(aDf)
    for country in top_countries:
        country_id = re.sub("\s+", "_", country.lower())
        country_ts = getTimeSeries(aDf, country=country)
        data[country_id] = country_ts

    for country, ts in data.items():
        country_id = re.sub("\s+", "_", country.lower())
        ts.to_csv(os.path.join(output_directory_path,
                               'ts-data-'+country_id+'.csv'))
    return data


# data_path = os.path.join('.', 'cs-train')
# out_path = os.path.join('.', 'work-data')
# df = ingestTrainData(data_path)

# # save the ingested data in the work-data directory
# df.to_csv(os.path.join(out_path, 'train-data-cleaned.csv'))


# getAllTS(df, out_path)
