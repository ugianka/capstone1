from os import listdir
from os.path import isfile, join
import pandas as pd


def ingestTrainData(data_path):

    bogus = {
        'StreamID': 'stream_id',
        'TimesViewed': 'times_viewed',
        'total_price': 'price'
    }

    # create a list with all the file path contained in the train data  directory
    onlyfiles = [join(data_path, f)
                 for f in listdir(data_path) if isfile(join(data_path, f))]

    inpDf = pd.DataFrame()

    for idx, fname in enumerate(onlyfiles):
        df = pd.read_json(fname)
        cols = df.columns
        for c in bogus:
            if(c in cols):
                df.rename(columns=bogus, inplace=True)
        inpDf = inpDf.append(df)
    return inpDf


data_path = './cs-train'
df = ingestTrainData(data_path)
df.to_csv('.cleaned.csv')
