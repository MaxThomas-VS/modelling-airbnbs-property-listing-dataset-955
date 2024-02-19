#%%
import argument_parser as ap
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
import warnings
from sklearn.exceptions import ConvergenceWarning



from tabular_data import ModelData
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

rstate=100
pre_scale=True

def save_model(best_params, args, extra_title='baseline_'):
    model_name = extra_title + args['ml_model'] + '.joblib'
    model_path = Path(args['export_path'], model_name)
    joblib.dump(best_params, model_path)

def quality_control_figure(y_test, X_test, best_params, args):
    plt.figure()
    plt.scatter(y_test, best_params['best_estimator'].predict(X_test), color='black')
    plt.title(args['ml_model'])
    plt.savefig(Path(args['export_path'], args['ml_model']+'.png'))

def tune_regression_hyperparameters(model_class, X_train, y_train, param_grid):

    mse = make_scorer(mean_squared_error, greater_is_better=False)

    pipe = make_pipeline(StandardScaler(), model_class())

    #search = GridSearchCV(pipe, param_grid, scoring=mse)
    search = GridSearchCV(pipe, param_grid, scoring='neg_root_mean_squared_error', n_jobs=-1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        search.fit(X_train, y_train)

    search_results = {'best_estimator': search.best_estimator_,
                      'best_params': search.best_params_,
                      'validation_RMSE': -search.best_score_}

    return search_results

def tune_classification_hyperparameters(model_class, X_train, y_train, param_grid):

    mse = make_scorer(mean_squared_error, greater_is_better=False)

    pipe = make_pipeline(StandardScaler(), model_class())

    #search = GridSearchCV(pipe, param_grid, scoring=mse)
    search = GridSearchCV(pipe, param_grid, scoring='neg_root_mean_squared_error', n_jobs=-1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        search.fit(X_train, y_train)

    search_results = {'best_estimator': search.best_estimator_,
                      'best_params': search.best_params_,
                      'validation_RMSE': -search.best_score_}

    return search_results

def select_model(model_name):
    if model_name.lower() == 'sgdregressor':
        from sklearn.linear_model import SGDRegressor
        return SGDRegressor
    elif model_name.lower() == 'linearregressor':
        from sklearn.linear_model import LinearRegression
        return LinearRegression
    elif model_name.lower() == 'decisiontreeregressor':
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor
    elif model_name.lower() == 'randomforestregressor':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor
    elif model_name.lower() == 'gradientboostingregressor':
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor
    elif model_name.lower() == 'svr':
        from sklearn.svm import SVR
        return SVR
    elif model_name.lower() == 'kernelridge':
        from sklearn.kernel_ridge import KernelRidge
        return KernelRidge
    elif model_name.lower() == 'bayesianridge':
        from sklearn.linear_model import BayesianRidge
        return BayesianRidge
    elif model_name.lower() == 'logisticregression':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression
    else:
        raise ValueError(f'{model_name} not a valid model name')

def make_parameter_grid(model_name):
    if model_name.lower() == 'sgdregressor':
        return {'sgdregressor__alpha': [1e-05, 0.0001, 0.001, 0.01, 0.1],
                'sgdregressor__loss': ['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive'],
                'sgdregressor__penalty': ['elasticnet'],
                'sgdregressor__l1_ratio': [0, 0.2, 0.4, 0.6, 0.8, 1]}
    elif model_name.lower() == 'decisiontreeregressor':
        return {'decisiontreeregressor__max_depth': [2, 4, 6, 8, 10],
                'decisiontreeregressor__min_samples_split': [2, 4, 6, 8, 10],
                'decisiontreeregressor__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'decisiontreeregressor__min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}
    elif model_name.lower() == 'randomforestregressor':
        return {'randomforestregressor__n_estimators': [100, 350, 500],
                'randomforestregressor__max_depth': [2, 6, 10],
                'randomforestregressor__min_samples_split': [2, 6, 10],
                'randomforestregressor__min_samples_leaf': [1, 5, 9],
                'randomforestregressor__min_impurity_decrease': [0.0, 0.25, 0.5]}
    elif model_name.lower == 'gradientboostingregressor':
        return {'gradientboostingregressor__n_estimators': [100, 300, 500],
                'gradientboostingregressor__max_depth': [2, 6, 10],
                'gradientboostingregressor__min_samples_split': [2, 6, 10],
                'gradientboostingregressor__min_samples_leaf': [1, 5, 9],
                'gradientboostingregressor__min_impurity_decrease': [0.0, 0.25, 0.5]}
    elif model_name.lower() == 'linearregressor':
        return {'linearregressor__fit_intercept': [True, False]}
    elif model_name.lower() == 'xgbrregressor':
        return {'xgbrregressor__n_estimators': [100, 300, 500],
                'xgbrregressor__max_depth': [2, 6, 10],
                'xgbrregressor__min_samples_split': [2, 6, 10],
                'xgbrregressor__min_samples_leaf': [1, 5, 9],
                'xgbrregressor__min_impurity_decrease': [0.0, 0.25, 0.5]}
    elif model_name.lower() == 'gradientboostingregressor':
        return {'gradientboostingregressor__n_estimators': [100, 300, 500],
                'gradientboostingregressor__max_depth': [2, 6, 10],
                'gradientboostingregressor__min_samples_split': [2, 6, 10],
                'gradientboostingregressor__min_samples_leaf': [1, 5, 9],
                'gradientboostingregressor__min_impurity_decrease': [0.0, 0.25, 0.5]}
    elif model_name.lower() == 'svr':
        return {'svr__C': [0.1, 1, 10, 100],
                'svr__gamma': [1, 0.1, 0.01, 0.001],
                'svr__kernel': ['rbf', 'poly', 'sigmoid']}
    elif model_name.lower() == 'kernelridge':
        return {'kernelridge__alpha': [1e-05, 0.0001, 0.001, 0.01, 0.1],
                'kernelridge__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'kernelridge__gamma': [1, 0.1, 0.01, 0.001]}
    elif model_name.lower() == 'bayesianridge':
        return {'bayesianridge__alpha_1': [1e-05, 0.0001, 0.001, 0.01, 0.1],
                'bayesianridge__alpha_2': [1e-05, 0.0001, 0.001, 0.01, 0.1],
                'bayesianridge__lambda_1': [1e-05, 0.0001, 0.001, 0.01, 0.1],
                'bayesianridge__lambda_2': [1e-05, 0.0001, 0.001, 0.01, 0.1]}
    elif model_name.lower() == 'lgbmregressor':
        return {'lgbmregressor__n_estimators': [100, 300, 500],
                'lgbmregressor__max_depth': [2, 6, 10],
                'lgbmregressor__min_child_samples': [2, 6, 10],
                'lgbmregressor__min_child_weight': [1e-05, 0.0001, 0.01, 0.1],
                'lgbmregressor__subsample': [0.1, 0.5, 1],
                'lgbmregressor__colsample_bytree': [0.1, 0.5, 1]}
    elif model_name.lower() == 'catboostregressor':
        return {'catboostregressor__n_estimators': [100, 300, 500],
                'catboostregressor__max_depth': [2, 6, 10],
                'catboostregressor__learning_rate': [0.1, 0.5, 1],
                'catboostregressor__l2_leaf_reg': [1e-05, 0.0001, 0.01, 0.1]}
    elif model_name.lower() == 'logisticregression':
        return {'logisticregression__l1_ratio': [0, 0.2, 0.4, 0.6, 0.8, 1],
                'logisticregression__C': [0.1, 1, 10, 100]}


def get_tabular_data(args, test_size=0.2, rstate=None, column='Price_Night', one_hot_encode_labels=False):
    if rstate is None:
        rstate = args['random_state']

    model_data = ModelData()

    df = model_data.load_tabular_data(args['file_name'], args['import_path'])

    to_model = model_data.extract_label(df, column, numeric_only=True)

    X_train, X_test, y_train, y_test = train_test_split(to_model[0], to_model[1], test_size=test_size, random_state=rstate)

    if one_hot_encode_labels:
        y_train = one_hot_encode_column(y_train)
        y_test = one_hot_encode_column(y_test)

    return X_train, X_test, y_train, y_test

def tune_one_model(args, X_train, y_train):

    model_class = select_model(args['ml_model'])

    # get baseline performance
    pipe = make_pipeline(StandardScaler(), model_class())
    pipe.fit(X_train, y_train)
    baseline_performance = {'best_estimator': pipe,
                            'best_params': pipe.get_params()}
    
    save_model(baseline_performance, args, 'baseline_')
    #print("baseline RMSE: %0.3f" % baseline_performance['validation_RMSE'])


    # get best hyperparameters
    # TODO: parameters should be passed as namelist rather tham function
    parameters = make_parameter_grid(args['ml_model'])  
    best_params = tune_regression_hyperparameters(model_class, X_train, y_train, parameters)

    save_model(best_params, args, 'tuned_')
    #print("tuned RMSE (validation): %0.3f" % best_params['validation_RMSE'])
    #print('tuned RMSE: '+ np.sqrt(mean_squared_error(y_test, best_params['best_estimator'].predict(X_test))))
    print('finishd tuning model: ' + args['ml_model'])


def evaluate_one_model(args, X_test, y_test):

    tuned_model = joblib.load(Path(args['export_path'], 'tuned_'+args['ml_model']+'.joblib'))
    baseline_model = joblib.load(Path(args['export_path'], 'baseline_'+args['ml_model']+'.joblib'))

    baseline_RMSE = np.sqrt(mean_squared_error(y_test, baseline_model['best_estimator'].predict(X_test)))
    tuned_RMSE = np.sqrt(mean_squared_error(y_test, tuned_model['best_estimator'].predict(X_test)))
    
    quality_control_figure(y_test, X_test, tuned_model, args)

    print('For ' + args['ml_model'])
    print(f'baseline RMSE: {baseline_RMSE}')
    print(f'tuned RMSE: {tuned_RMSE}')

    return tuned_RMSE

def one_hot_encode_column(pds):
    #oh_encoder = OneHotEncoder(sparse_output=False, drop='first')
    oh_encoder = OneHotEncoder(sparse_output=False)
    label_encoder = LabelEncoder()
    pds_2 = label_encoder.fit_transform(pds).reshape(-1,1)
    oh_encoder.fit(pds_2)
    oh_labels = oh_encoder.transform(pds_2)
    return np.argmax(oh_labels, axis=1)
    #return oh_labels

#%%
if __name__ == '__main__':
    
    arguments = ap.make_args()
        
    X_train, X_test, y_train, y_test = get_tabular_data(arguments)
    
    model_stats = {}

    for model in arguments['ml_model']:

        arguments['ml_model'] = model

        tune_one_model(arguments, X_train, y_train)

        model_stats[model] = evaluate_one_model(arguments, X_test, y_test)

    print(model_stats)
    print('See also quality control figures in '+arguments['export_path'])


    # rough logisitc regression
    model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    X_train, X_test, y_train, y_test = get_tabular_data(arguments, column='Category', one_hot_encode_labels=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    parmaeters = {'penalty': ['l1','l2','none'], 'C': [0.001,0.01,0.1,1,10,100,1000]}
    search = GridSearchCV(model, param_grid=parmaeters, n_jobs=-1)
    search.fit(X_train, y_train)
    y_pred_tuned = search.best_estimator_.predict(X_test)
    print(classification_report(y_test, y_pred_tuned))



    
# %%
