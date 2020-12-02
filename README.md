
# <span style="color:rgba(255,150,0,1);"> AAVAIL revenue predictor service</span>
# <span style="color:rgba(255,150,0,1);"> installation</span> 
## install in docker
There is a dockerfile on the root of the project to create the docker image.
Be sure to have docker running on your machine.
Then execute from the root directory:
```
docker build -t <image-name> .
```
this will create an image with name \<image-name> in your docker registry.

To run the image type:
```
docker run -d -p 5000:5000 --name <container-name> <image-name>
```
\<container-name> is the name you want to give to the container.

The --name \<aname> parameter is optional if you don't specify it docker will create a name for the container.

The -p 5000:5000 will create a NAT in your machine on port 5000 to the port 5000 of the container. You will be able to send requests to the flask server in the container addressing your machine like this for example:

```
curl -X GET  http://127.0.0.1:5000/train
```

to verify that the container is running:
```
docker ps
```
this should show something like:
```
CONTAINER ID        IMAGE                  COMMAND               CREATED             STATUS              PORTS                    NAMES
be97439e4684        aavail_predict:1.0.9   "./start-server.sh"   5 seconds ago       Up 3 seconds        0.0.0.0:5000->5000/tcp   aavail
(
```

# Service interface

These are the path of the different services, all services accept **GET** method and receive all parameters in **query string**



|  path   |  message  | parameters| description |
|:-------|:---------:|:---------:|:-----------|
|  /train | GET       | train_params      | train all the models in production mode | 
| /predict | GET | predict_params | predict the revenue for the month after the date passed as parameter|
| /getLog | GET | getlog_params | get the logs for the date passed as parameter


where

## train_params are:
| param name| description |
|:----------|:------------|
| env   | can be either test or prod if not present env will be set to prod|

## predict_params are:
| param name| description |
|:----------|:------------|
| country   | the name of the country is all lowercase and spaces must be replaced by _ |
| year      | the year of the date for wich we want the prediction of the revenue of the next month    |
| month | the month of the date |
| day | the day of the date |
| env   | can be either test or prod if not present env will be set to prod|



## getlog_params are:
| param name| description |
|:----------|:------------|
| type   | either pred for prediction logs or train for train logs |
| date      | the date in ISO format like 2020-12-01 for December 1st 2020   |


# service response
all services respond with a json. Status code of the http response is always 200 even on failure.
The field result in the JSON response is 0 if the API has executed without errors -1 otherwise .
In case of error also the field error is present with a description of the error



# services invocation examples
### example of train
this will train all models for the top 10 countries

```
curl -X GET http://127.0.0.1:5000/train?env=test
```

### example of prediction

for example if we want the prediction of the revenue for november 2018 for the country United Kingdom:
```
curl -X GET "http://127.0.0.1:5000/predict?country=united_kingdom&year=2018&month=10&day=31"
```

### example of log fetch

this will retrieve all the prediction logs for the day: 2018-01-01

```
curl -X GET "http://127.0.0.1:5000/getLog?type=pred&date=2020-12-01"
```

# Unit Tests

Unit Tests are runned by the  unit_test.sh shell in the root directory.
Once you have run the application in a docker container you can execute unit tests with the command:
```
docker exec -ti <container-name> /usr/src/aavail_predict/unit_test.sh
```

where \<container-name> is the name or the id of the container running the application