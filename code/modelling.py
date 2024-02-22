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
import os



from tabular_data import ModelData
from tabular_data import get_tabular_data
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import datetime

import scikitplot as skplt

from nn_train_eval import train_and_evaluate_nn


def results_dir(args):
    # make results directory and return path to it
    # the results are stored in ../models/<label>/<ml_model>/
    put_results_here = Path(args['export_path'], args['label'], get_model_type(args['ml_model']))  
    put_results_here = Path(put_results_here, args['ml_model'])
    os.system('mkdir -p ' + str(put_results_here))
    return put_results_here

def dummy_args():
    # dummy arguments for testing and jupyter notebooks
    return {'file_name': 'listing.csv', 
            'import_path': '../data/intermediate/airbnb-property-listings/tabular_data/', 
            'export_path': '../models/', 
            'label': 'Price_Night', 
            'ml_model': ['sgdregressor'], 
            'random_state': 42, 
            'nn_n_configs': 16, 
            'nn_epochs': 50, 
            'nn_single_config': None,
            'batch_size': 64, 
            'nn_val_split_size': 0.2,
            'do_training': True, 
            'do_evaluation': True, 
            'script': 'modelling.py',
            'run_time': datetime.datetime.now()}

def save_model(best_params, args, extra_title='baseline_'):
    # save the best parameters to a file in appropriate results dir
    # the file name is <extra_title><ml_model>.joblib
    # the extra_title can be baseline_ or tuned_
    
    model_name = extra_title + args['ml_model'] + '.joblib'
    
    model_path = Path(results_dir(args), model_name)

    joblib.dump(best_params, model_path)

def quality_control_figure(y_test, X_test, best_params, args):
    # make a quality control figure and save it to the results directory
    plt.figure(dpi=300)
    y_pred = best_params['best_estimator'].predict(X_test)
    title = args['ml_model'] + ' performance for ' + args['label']
    if get_model_type(args['ml_model']) == 'regression':
        plt.scatter(y_test, y_pred, color='black')
        plt.ylabel('Predicted')
        plt.xlabel('True')
        plt.title(title)
    elif get_model_type(args['ml_model']) == 'classification':
        skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
        plt.title(title)

    plt.savefig(Path(results_dir(args), args['ml_model']+'.png'))

def tune_regression_hyperparameters(model_class, X_train, y_train, param_grid):
    '''
    Tune hyperparameters for a regression model using grid search
    
    Arguments:
    model_class: the class of the model to be tuned (e.g. RandomForestRegressor)
    X_train: the training data
    y_train: the training labels
    param_grid: the hyperparameters to be tuned
    
    Returns:
    search_results: a dictionary containing the best estimator, best parameters and validation RMSE'''
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
    '''
    Tune hyperparameters for a classification model using grid search
    
    Arguments:
    model_class: the class of the model to be tuned (e.g. RandomForestClassifier)
    X_train: the training data
    y_train: the training labels
    param_grid: the hyperparameters to be tuned
    
    Returns:
    search_results: a dictionary containing the best estimator, best parameters and validation accuracy
    '''
   # if model_class == 'LogisticRegression':
   #     model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    #pipe = make_pipeline(StandardScaler(), model_class(multi_class='multinomial', solver='saga'))
    pipe = make_pipeline(StandardScaler(), model_class())

    #search = GridSearchCV(pipe, param_grid, scoring=mse)
    search = GridSearchCV(pipe, param_grid, n_jobs=-1)

    search.fit(X_train, y_train)

    search_results = {'best_estimator': search.best_estimator_,
                      'best_params': search.best_params_,
                      'validation_accuracy': -search.best_score_}

    return search_results


def select_model(model_name):
    # load model class for a given model name
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
    elif model_name.lower() == 'decisiontreeclassifier':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier
    elif model_name.lower() == 'randomforestclassifier':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier
    elif model_name.lower() == 'gradientboostingclassifier':
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier
    else:
        raise ValueError(f'{model_name} not a valid model name')
    
def get_model_type(model_name):
    # return the type of model for a given model name
    regressors = ['sgdregressor', 'decisiontreeregressor', 'randomforestregressor', 'gradientboostingregressor', 'svr', 'kernelridge', 'bayesianridge']
    classifiers = ['logisticregression', 'decisiontreeclassifier', 'randomforestclassifier', 'gradientboostingclassifier']
    neural_networks = 'nn'
    if model_name.lower() in regressors:
        return 'regression'
    elif model_name.lower() in classifiers:
        return 'classification'
    elif model_name.lower() == neural_networks:
        return 'neural_network'
    else:
        raise ValueError(f'{model_name} not a valid model name')
    

def make_parameter_grid(model_name):
    # return a parameter grid for a given model name
    if model_name.lower() == 'sgdregressor':
        return {'sgdregressor__alpha': [1e-05, 0.0001, 0.001, 0.01, 0.1],
                'sgdregressor__loss': ['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive'],
                'sgdregressor__penalty': ['elasticnet'],
                'sgdregressor__l1_ratio': [0, 0.05, 0.15, 0.2, 0.4, 0.6, 0.8, 1]}
    elif model_name.lower() == 'decisiontreeregressor':
        return {'decisiontreeregressor__max_depth': [2, 4, 6, 8, 10],
                'decisiontreeregressor__min_samples_split': [2, 4, 6, 8, 10],
                'decisiontreeregressor__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'decisiontreeregressor__min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}
    elif model_name.lower() == 'randomforestregressor':
        return {'randomforestregressor__n_estimators': [10, 50, 100, 200, 350, 500, 1000],
                'randomforestregressor__max_depth': [None, 2, 4, 6, 8, 10],
                'randomforestregressor__min_samples_split': [2, 6, 10],
                'randomforestregressor__min_samples_leaf': [1, 6, 9],
                'randomforestregressor__min_impurity_decrease': [0.0, 0.25, 0.5]}
    elif model_name.lower == 'gradientboostingregressor':
        return {'gradientboostingregressor__n_estimators': [10, 100, 350, 500, 1000],
                'gradientboostingregressor__max_depth': [2, 4, 6, 8, 10],
                'gradientboostingregressor__min_samples_split': [2, 4, 6, 8, 10],
                'gradientboostingregressor__min_samples_leaf': [1, 3, 5, 7, 9],
                'gradientboostingregressor__min_impurity_decrease': [0.0, 0.25, 0.5]}
    elif model_name.lower() == 'xgbrregressor':
        return {'xgbrregressor__n_estimators': [10, 100, 350, 500, 1000],
                'xgbrregressor__max_depth': [2, 6, 10],
                'xgbrregressor__min_samples_split': [2, 6, 10],
                'xgbrregressor__min_samples_leaf': [1, 6, 9],
                'xgbrregressor__min_impurity_decrease': [0.0, 0.25, 0.5]}
    elif model_name.lower() == 'gradientboostingregressor':
        return {'gradientboostingregressor__n_estimators': [10, 50, 100, 200, 350, 500, 1000],
                'gradientboostingregressor__max_depth': [2, 3, 4, 6, 8, 10],
                'gradientboostingregressor__min_samples_split': [2, 6, 10],
                'gradientboostingregressor__min_samples_leaf': [1, 6, 9],
                'gradientboostingregressor__min_impurity_decrease': [0.0, 0.25, 0.5]}
    elif model_name.lower() == 'svr':
        return {'svr__C': np.logspace(-4, 2, 10),
                'svr__gamma': np.logspace(-4, 2, 10),
                'svr__kernel': ['sigmoid']}
    elif model_name.lower() == 'kernelridge':
        return {'kernelridge__alpha': np.logspace(-4, 2, 10),
                'kernelridge__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'kernelridge__gamma': np.logspace(-4, 2, 10)}
    elif model_name.lower() == 'bayesianridge':
        return {'bayesianridge__alpha_1': [1e-05, 0.0001, 0.001, 0.01, 0.1],
                'bayesianridge__alpha_2': [1e-05, 0.0001, 0.001, 0.01, 0.1],
                'bayesianridge__lambda_1': [1e-05, 0.0001, 0.001, 0.01, 0.1],
                'bayesianridge__lambda_2': [1e-05, 0.0001, 0.001, 0.01, 0.1]}
    elif model_name.lower() == 'lgbmregressor':
        return {'lgbmregressor__n_estimators': [10, 100, 350, 500, 1000],
                'lgbmregressor__max_depth': [2, 4, 6, 8, 10],
                'lgbmregressor__min_child_samples': [2, 4, 6, 8, 10],
                'lgbmregressor__min_child_weight': [1e-05, 0.0001, 0.01, 0.1],
                'lgbmregressor__subsample': [0.1, 0.5, 1],
                'lgbmregressor__colsample_bytree': [0.1, 0.5, 1]}
    elif model_name.lower() == 'catboostregressor':
        return {'catboostregressor__n_estimators': [100, 300, 500],
                'catboostregressor__max_depth': [2, 6, 10],
                'catboostregressor__learning_rate': [0.1, 0.5, 1],
                'catboostregressor__l2_leaf_reg': [1e-05, 0.0001, 0.01, 0.1]}
    elif model_name.lower() == 'logisticregression':
        return {'logisticregression__penalty' : ['elastic_net','none'],
                'logisticregression__C' : np.logspace(-4, 2, 10),
                'logisticregression__l1_ratio' : [0, 0.2, 0.4, 0.6, 0.8, 1]}
    elif model_name.lower() == 'decisiontreeclassifier':
        return {'decisiontreeclassifier__max_depth': [2, 4, 6, 8, 10],
                'decisiontreeclassifier__min_samples_split': [2, 4, 6, 8, 10],
                'decisiontreeclassifier__min_samples_leaf': [1, 3, 5, 7, 9],
                'decisiontreeclassifier__min_impurity_decrease': [0.0, 0.25, 0.5]}
    elif model_name.lower() == 'randomforestclassifier':
        return {'randomforestclassifier__n_estimators': [10, 50, 100, 200, 350, 500, 1000],
                'randomforestclassifier__max_depth': [None, 2, 3, 4, 6, 8, 10],
                'randomforestclassifier__min_samples_split': [2, 6, 10],
                'randomforestclassifier__min_samples_leaf': [1, 5, 9],
                'randomforestclassifier__min_impurity_decrease': [0.0, 0.25, 0.5]}
    elif model_name.lower() == 'gradientboostingclassifier':    
        return {'gradientboostingclassifier__n_estimators': [10, 50, 100, 200, 350, 500, 1000],
                'gradientboostingclassifier__max_depth': [2, 3, 4, 6, 9],
                'gradientboostingclassifier__min_samples_split': [2, 6, 10],
                'gradientboostingclassifier__min_samples_leaf': [1, 6, 10],
                'gradientboostingclassifier__min_impurity_decrease': [0.0, 0.25, 0.5]}

# def get_tabular_data(args, test_size=0.3, rstate=None, column_overide=None, one_hot_encode_labels=False):
#     '''
#     Load tabular data and split it into training and testing sets.
    
#     Arguments:
#     args: a dictionary containing the file name, import path, label, random state and ml model
#     test_size: the proportion of the data to be used for testing
#     rstate: the random state
#     column_overide: the column to be used as the label. If None, Price_Night is the label
#     one_hot_encode_labels: whether to one hot encode the labels
    
#     Returns:
#     X_train: the training data
#     X_test: the testing data
#     y_train: the training labels
#     y_test: the testing labels
#     '''
#     if rstate is None:
#         rstate = args['random_state']

#     column = args['label']

#     if column_overide is not None:
#         column = column_overide

#     model_data = ModelData()

#     df = model_data.load_tabular_data(args['file_name'], args['import_path'])

#     to_model = model_data.extract_label(df, column, numeric_only=True)

#     X_train, X_test, y_train, y_test = train_test_split(to_model[0], to_model[1], test_size=test_size, random_state=rstate)

#     if one_hot_encode_labels:
#         y_train = one_hot_encode_column(y_train)
#         y_test = one_hot_encode_column(y_test)

#     return X_train, X_test, y_train, y_test

def tune_one_regression_model(args, X_train, y_train):
    '''
    Tune a regression model and save the best parameters to a file.

    Both calculates a baseline from the standard parameters, and calls tune_regression_hyperparameters to find the best parameters.

    Data are scaled using StandardScaler before fitting.
    
    Arguments:
    args: a dictionary containing the file name, import path, label, random state and ml model
    X_train: the training data
    y_train: the training labels
    '''

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

def tune_one_classification_model(args, X_train, y_train):
    '''
    Tune a classification model and save the best parameters to a file.
    
    Both calculates a baseline from the standard parameters, and calls tune_classification_hyperparameters to find the best parameters.
    
    Data are scaled using StandardScaler before fitting.
    
    Arguments:
    args: a dictionary containing the file name, import path, label, random state and ml model
    X_train: the training data
    y_train: the training labels
    '''

    model_class = select_model(args['ml_model'])

    # get baseline performance
    if args['ml_model'] == 'LogisticRegression':
        pipe = make_pipeline(StandardScaler(), model_class(multi_class='multinomial', solver='saga'))
    else:
        pipe = make_pipeline(StandardScaler(), model_class())
    pipe.fit(X_train, y_train)
    baseline_performance = {'best_estimator': pipe,
                            'best_params': pipe.get_params()}
    
    save_model(baseline_performance, args, 'baseline_')
    #print("baseline RMSE: %0.3f" % baseline_performance['validation_RMSE'])

    # get best hyperparameters
    # TODO: parameters should be passed as namelist rather tham function
    parameters = make_parameter_grid(args['ml_model'])  
    print(parameters)
    best_params = tune_classification_hyperparameters(model_class, X_train, y_train, parameters)

    save_model(best_params, args, 'tuned_')
    #print("tuned RMSE (validation): %0.3f" % best_params['validation_RMSE'])
    #print('tuned RMSE: '+ np.sqrt(mean_squared_error(y_test, best_params['best_estimator'].predict(X_test))))
    print('finishd tuning model: ' + args['ml_model'])


def evaluate_one_regression_model(args, X_test=None, y_test=None):
    '''
    Evaluate a regression model and find the RMSE for the baseline and tuned models.
    
    Arguments:
    args: a dictionary containing the file name, import path, label, random state and ml model
    X_test: the testing data. If None, the data will be loaded from the file
    y_test: the testing labels. If None, the data will be loaded from the file
    
    Returns:
    tuned_RMSE: the RMSE for the tuned model'''
    if X_test is None:
        X_train, X_test, y_train, y_test = get_tabular_data(args)

    #tuned_model = joblib.load(Path(args['export_path'], 'tuned_'+args['ml_model']+'.joblib'))
    #baseline_model = joblib.load(Path(args['export_path'], 'baseline_'+args['ml_model']+'.joblib'))

    tuned_model = joblib.load(Path(results_dir(args), 'tuned_' +args['ml_model'] + '.joblib'))
    baseline_model = joblib.load(Path(results_dir(args), 'baseline_' +args['ml_model'] + '.joblib'))

    baseline_RMSE = np.sqrt(mean_squared_error(y_test, baseline_model['best_estimator'].predict(X_test)))
    tuned_RMSE = np.sqrt(mean_squared_error(y_test, tuned_model['best_estimator'].predict(X_test)))
        
    quality_control_figure(y_test, X_test, tuned_model, args)

    print('For ' + args['ml_model'])
    print(f'baseline RMSE: {baseline_RMSE}')
    print(f'tuned RMSE: {tuned_RMSE}')
    print('')

    return tuned_RMSE


def evaluate_one_classification_model(args, X_test=None, y_test=None, column_overide=None):
    '''
    Evaluate a classification model and find the accuracy for the baseline and tuned models.
    
    Arguments:
    args: a dictionary containing the file name, import path, label, random state and ml model
    X_test: the testing data. If None, the data will be loaded from the file
    y_test: the testing labels. If None, the data will be loaded from the file
    '''
    if X_test is None:
        X_train, X_test, y_train, y_test = get_tabular_data(args, column_overide=args['label'])

    #tuned_model = joblib.load(Path(args['export_path'], 'tuned_'+args['ml_model']+'.joblib'))
    #baseline_model = joblib.load(Path(args['export_path'], 'baseline_'+args['ml_model']+'.joblib'))
    tuned_model = joblib.load(Path(results_dir(args), 'tuned_' +args['ml_model'] + '.joblib'))
    baseline_model = joblib.load(Path(results_dir(args), 'baseline_' +args['ml_model'] + '.joblib'))

    baseline_accuracy = accuracy_score(y_test, baseline_model['best_estimator'].predict(X_test))
    tuned_accuracy = accuracy_score(y_test, tuned_model['best_estimator'].predict(X_test))
    
    quality_control_figure(y_test, X_test, tuned_model, args)

    print('For ' + args['ml_model'])
    print(f'baseline accuracy: {baseline_accuracy}')
    #print(baseline_model['best_estimator'].get_params())
    print(f'tuned accuracy: {tuned_accuracy}')
    print('')
    #print(tuned_model['best_estimator'].get_params())

    return tuned_accuracy

#def one_hot_encode_column(pds):
#    '''
#    One hot encode a pandas series.

#   Arguments:
#    pds: a pandas series to encode

#    Returns:
#    oh_labels: the one hot encoded labels as integers
#    '''
#    #oh_encoder = OneHotEncoder(sparse_output=False, drop='first')
#    oh_encoder = OneHotEncoder(sparse_output=False)
#    label_encoder = LabelEncoder()
#    pds_2 = label_encoder.fit_transform(pds).reshape(-1,1)
#    oh_encoder.fit(pds_2)
#    oh_labels = oh_encoder.transform(pds_2)
#    return np.argmax(oh_labels, axis=1)
#    #return oh_labels

def train_regression_model(arguments):
        # gets train data and calls tuning for regression
    
        X_train, X_test, y_train, y_test = get_tabular_data(arguments)
    
        tune_one_regression_model(arguments, X_train, y_train)


def train_classification_model(arguments):
        # gets train data and calls tuning for classification

        X_train, X_test, y_train, y_test = get_tabular_data(arguments, column_overide=arguments['label'])
    
        tune_one_classification_model(arguments, X_train, y_train)
    



def train_and_evaluate_nn_model(arguments):
    # trains and evaluates a neural network model based on config files in /nn_configs/
    if arguments['ml_model'].lower() == 'nn':
        train_various_configurations(arguments)


def do_training(arguments):
    # trains models based on the arguments and model type
    for model in arguments['ml_model']:
        arguments['ml_model'] = model
        if get_model_type(model) == 'regression':
            train_regression_model(arguments)
        elif get_model_type(model) == 'classification':
            train_classification_model(arguments)
        elif get_model_type(model) == 'neural_network':
            train_and_evaluate_nn(arguments)

def do_evaluation(arguments):
    # evaluates models based on the arguments and model type

    for model in arguments['ml_model']:

        arguments['ml_model'] = model

        if get_model_type(model) == 'regression':
            evaluate_one_regression_model(arguments)

        elif get_model_type(model) == 'classification':

            evaluate_one_classification_model(arguments)
        # model evaluation is done alongside tuning for nn
    

#%%
if __name__ == '__main__':
    
    arguments = ap.make_args()
    print(arguments)

    if arguments['do_training']:
        print('doing training')
        do_training(arguments.copy())

    if arguments['do_evaluation']:
        print('doing evaluation')
        do_evaluation(arguments.copy())



    
# %%
