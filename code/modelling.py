#%%
import argument_parser as ap
import numpy as np 
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

from tabular_data import ModelData
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

rstate=100
pre_scale=True

def save_model(model_choice, export_path='../models/', model_type='regression', model_name='SGDregression'):
    model_name = model_name + '.joblib'
    model_path = Path(export_path, model_type, model_name)
    joblib.dump(model_choice, model_path)

def grid_search(model_class, X_train, y_train, param_grid):

    mse = make_scorer(mean_squared_error,greater_is_better=False)

    pipe = make_pipeline(StandardScaler(), model_class())

    search = GridSearchCV(pipe, param_grid, scoring=mse)

    search.fit(X_train, y_train)

    search_results = {'best_estimator': search.best_estimator_,
                      'best_params': search.best_params_,
                      'validation_RMSE': np.sqrt(-search.best_score_)}

    return search_results

def select_model(model_name):
    if model_name.lower() == 'sgdregressor':
        return SGDRegressor
    elif model_name.lower() == 'linearregression':
        return LinearRegression
    else:
        raise ValueError(f'{model_name} not a valid model name')

def make_parameter_grid(model_name):
    if model_name.lower() == 'sgdregressor':
        return {'sgdregressor__alpha': [1e-05, 0.0001, 0.001, 0.01, 0.1],
                'sgdregressor__loss': ['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive'],
                'sgdregressor__penalty': ['elasticnet'],
                'sgdregressor__l1_ratio': [0, 0.2, 0.4, 0.6, 0.8, 1]}


# %%



if __name__ == '__main__':
    args = ap.make_args()
    #args = {'import_path': '../data/intermediate/airbnb-property-listings/tabular_data/',
    #        'file_name': 'listing.csv'}
    print(args)

    model_data = ModelData()

    df = model_data.load_tabular_data(args['file_name'], args['import_path'])

    to_model = model_data.extract_label(df, 'Price_Night', numeric_only=True)

    X_train, X_test, y_train, y_test = train_test_split(to_model[0], to_model[1], test_size=0.25, random_state=rstate)

    model_class = select_model(args['ml_model'])

    # get baseline performance
    pipe = make_pipeline(StandardScaler(), model_class())
    pipe.fit(X_train, y_train)
    baseline_performance = {'best_estimator': pipe,
                            'best_params': pipe.get_params(),
                            'validation_RMSE': np.sqrt(mean_squared_error(y_test, pipe.predict(X_test)))}
    
    save_model(baseline_performance, model_name='baseline_'+args['ml_model'])
    print(baseline_performance)


    # get best hyperparameters
    # TODO: parameters should be passed as namelist rather tham function
    parameters = make_parameter_grid(args['ml_model'])  
    best_params = grid_search(model_class, X_train, y_train, parameters)
    save_model(best_params, model_name='best_'+args['ml_model'])
    print(best_params)
