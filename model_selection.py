
from features_extraction import engineer_features
from data_ingestion import getAllTS, ingestTrainData
import os
import sys

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor

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


np.random.seed(41)

param_grid_rf = {
    'clf__criterion': ['mse', 'mae'],
    'clf__n_estimators': [10, 15, 20, 25]
}

clf = RandomForestRegressor()

# print('random_forest')
# randomForestClf = trainClf(clf=clf, param_grid_rf=param_grid_rf)
# print('svm')
# param_grid_rf = {
#     'clf__kernel': ['poly', 'rbf'],
#     'clf__degree': [2, 3, 4, 5],
#     'clf__epsilon': [0.05, 0.1, 0.2, 0.3]
# }
# svmClf = trainClf(clf=SVR(), param_grid_rf=param_grid_rf)
# print('sgd')
# param_grid_rf = {
# }
# svmClf = trainClf(clf=SGDRegressor(), param_grid_rf=param_grid_rf)


print('mlp')
param_grid_rf = {
    'clf__hidden_layer_sizes': [[100, 200, 400, 400, 300, 300, 200, 200, 200, 200, 150, 150, 100]],
    'clf__alpha': [0.0001, 0.001],
    'clf__beta_1': [0.6, 0.7, 0.9],
    'clf__beta_2': [0.995, 0.999, 0.9995]
}
svmClf = trainClf(clf=MLPRegressor(), param_grid_rf=param_grid_rf)
