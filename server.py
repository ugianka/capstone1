from flask import Flask, request, Response
import sys
import os
from os.path import join
import json
import datetime as dt
import pandas as pd
import unittest


from model import model_predict, model_train
sys.path.insert(1, './src')


app = Flask(__name__)


data_dir = os.path.join('.', 'data')


# welcome and help page of the service
@app.route('/')
def hello_world():
    return '''

    <style>
        body{
            font-family:'Helvetica';
            font-size:13px;
        }
        span{
            padding: 10px;
        }
        .left_align{
            display:flex;
            flex-direction:row;
            justify-content:flex-start;
            border-style:solid;
            border-color:rgba(0,0,0,0.1);
            border-width:1px;
        }

    </style>
    <title>AAVAIL revenue predictor service</title>
    <h1>AAVAIL revenue predictor service</h1>

    <span><h2>available API's:</h2></span><br>
    <table>
    <tr>
    <th><span class="left_align"><b>URL Path<b></span></th>
     <th><span class="left_align"><b>Params<b></span></th>
    <th><span class="left_align"><b>Description<b></span></th>
    </tr>

    <tr>
    <td><span class="left_align" style="color:blue;"><b>/train</b></span></td>
    <td><span class="left_align" style="color:blue;"><i>none</i></span></td>
    <td><span class="left_align"><i>trains the models for the different countries</i></span></td>
    </tr>
    <tr>
    <td><span class="left_align" style="color:blue;"><b>/predict</b></span></td>
    <td><span class="left_align" style="color:blue;">country,year,month,day</span></b></td>
    <td><span class="left_align"><i>predicts the revenue in a specific date</i></span></td>
    </tr>
    </table>

    '''


@app.route('/train')
def trainModel():
    response = {
        'result': -1
    }

    try:
        test = False
        env = request.args.get('env')
        if (env == 'test'):
            test = True

        model_train(data_dir, test=test)
        response['result'] = 0
    except:
        response['error'] = 'system error:' + str(sys.exc_info()[1])

    json_object = json.dumps(response, indent=4)
    return Response(str(json_object),
                    mimetype="application/json")


@app.route('/predict')
def predict():
    response = {
        'result': -1
    }
    try:
        country = request.args.get('country')
        year = request.args.get('year')
        month = request.args.get('month')
        day = request.args.get('day')
        res = model_predict(country, year, month, day)

        # text = '<h1 style="color:blue;">result'+str(res)+'</h1>'
        # print('type(res: ', type(res))
        if isinstance(res, dict):
            if 'y_pred' in res.keys():
                response['prediction'] = res['y_pred'].tolist()
                response['result'] = 0
        else:
            response['error'] = str(res)
        print('response: \n', response)

    except:
        response['error'] = 'system error:' + str(sys.exc_info()[1])
    json_object = json.dumps(response, indent=4)
    return Response(str(json_object),
                    mimetype="application/json")


@app.route('/getLog')
def log():
    response = {
        'result': -1
    }
    try:
        test = False
        date = request.args.get('date')
        today = dt.date.fromisoformat(date)
        logtype = request.args.get('type')
        env = request.args.get('env')
        if env == 'test':
            test = True
        log_dir = join('.', 'logs')

        if logtype and logtype in ['pred', 'train']:
            a = 1
        else:
            raise Exception(
                "ERROR (fetch logs) - type must be either pred or train")
        test_suffix = ''
        if test:
            test_suffix = '-test'
        log_file_path = join(
            log_dir, '{}{}-{}-{}-{}.log').format(logtype, test_suffix, today.year, today.month, today.day)

        if not os.path.exists(log_file_path):
            response['result'] = 0
            response['data'] = []
        else:
            df = pd.read_csv(log_file_path)
            response['result'] = 0
            response['data'] = df.to_dict(orient='records')

    except:
        response['error'] = 'system error:' + str(sys.exc_info()[1])
    json_object = json.dumps(response, indent=4)
    return Response(str(json_object),
                    mimetype="application/json")


class TestServer(unittest.TestCase):
    def test_train_API(self):
        request = {}
        req = request()

        request['args'] = ''
        trainModel()
