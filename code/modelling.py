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
from tabular_data import ModelData
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt


rstate=100
pre_scale=True



# %%



#if __name__ == '__main__':
#args = ap.make_args()
args = {'import_path': '../data/intermediate/airbnb-property-listings/tabular_data/',
        'file_name': 'listing.csv'}
print(args)

#%%
model_data = ModelData()

df = model_data.load_tabular_data(args['file_name'], args['import_path'])

to_model = model_data.extract_label(df, 'Price_Night', numeric_only=True)

X_train, X_test, y_train, y_test = train_test_split(to_model[0], to_model[1], test_size=0.4, random_state=rstate)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.3, random_state=rstate)

#%%
#reg_mdl = SGDRegressor(max_iter=10000, tol=0.0001, alpha=0.1, learning_rate='constant', eta0=1e-7)
reg = make_pipeline(SGDRegressor())
reg.fit(X_train, y_train)
print(mean_squared_error(y_val, reg.predict(X_val), squared=False))


#%%
def random_parameter_search(model_class, X_train, y_train, X_val, y_val, X_test, y_test, parameters, pre_scale = False):
    param_performance = pd.DataFrame(columns=parameters.keys())
    previous_examples = []
    for ix in range(1000):
        param_example = {}
        current_example_str = ''
        for key in parameters.keys():
            param_example[key] = random.choice(parameters[key])
            current_example_str += str(param_example[key]) + ','

        if not current_example_str in previous_examples:
            previous_examples.append(current_example_str)

            if pre_scale:
                reg = make_pipeline(StandardScaler(), model_class(**param_example, random_state=rstate))
            else:
                reg = make_pipeline(model_class(**param_example, random_state=rstate))

            reg.fit(X_train, y_train)

            rmse_val = mean_squared_error(y_val, reg.predict(X_val), squared=False)
            rmse_train = mean_squared_error(y_train, reg.predict(X_train), squared=False)
            rmse_test = mean_squared_error(y_test, reg.predict(X_test), squared=False)

            param_example['validation_RMSE'] = round(rmse_val,3)
            param_example['train_RMSE'] = round(rmse_train,3)
            param_example['test_RMSE'] = round(rmse_test,3)
            param_performance = param_performance._append(param_example, ignore_index=True)
    
    print(param_performance.sort_values(by='validation_RMSE', ascending=True).head(10))

    best_params_ix = param_performance.sort_values(by='validation_RMSE', ascending=True).head(1).index 
    best_params = {}
    for col in parameters.keys():
        best_params[col] = param_performance.loc[best_params_ix, col].values[0]

    if pre_scale:
        reg = make_pipeline(StandardScaler(), model_class(**best_params, random_state=rstate))
    else:
        reg = make_pipeline(model_class(**best_params, random_state=rstate))
    reg.fit(X_train, y_train)

    rmse_val = mean_squared_error(y_val, reg.predict(X_val), squared=False)
    rmse_train = mean_squared_error(y_train, reg.predict(X_train), squared=False)
    rmse_test = mean_squared_error(y_test, reg.predict(X_test), squared=False)

    best_performance = {'validation_RMSE': round(rmse_val,3),
                        'train_RMSE': round(rmse_train,3),
                        'test_RMSE': round(rmse_test,3)}
    return reg, best_params, best_performance



# %%
parameters = {'alpha': [1e-05, 0.0001, 0.001, 0.01, 0.1],
                'loss': ['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive'],
                'penalty': ['elasticnet'],
                'l1_ratio': [0, 0.2, 0.4, 0.6, 0.8, 1]}
reg, params, performance = random_parameter_search(SGDRegressor, X_train, y_train, X_val, y_val, X_test, y_test, parameters, pre_scale=pre_scale)
print(performance)

# %%
y_val_pred = reg.predict(X_val)
y_test_pred = reg.predict(X_test)
y_train_pred = reg.predict(X_train)

plt.scatter(y_val, y_val_pred, color='black')
plt.scatter(y_test, y_test_pred, color='blue')  
plt.scatter(y_train, y_train_pred, color='red') 

lm = LinearRegression()
lm.fit(X_train, y_train)
lm_val_pred = lm.predict(X_val)

plt.scatter(y_val, lm_val_pred, color='green')


# %%
sgd = SGDRegressor()
pipe = make_pipeline(StandardScaler(), sgd)
param_grid = {'sgdregressor__alpha': [1e-05, 0.0001, 0.001, 0.01, 0.1],
                'sgdregressor__loss': ['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive'],
                'sgdregressor__penalty': ['elasticnet'],
                'sgdregressor__l1_ratio': [0, 0.2, 0.4, 0.6, 0.8, 1]}

# NOTE here we could train on set minus test_set (contrast to self implementation which has test, val, train)

def grid_search(model_class, parmaeters, X_train, y_train, pre_scale=True, random_state=rstate, search_type = 'cv'):

    if search_type == 'cv':
        Searcher = GridSearchCV
    elif search_type == 'random':
        Searcher = RandomizedSearchCV
    

    model = model_class()

    if pre_scale:
        pipe = make_pipeline(StandardScaler(), model)
    else:
        pipe = make_pipeline(model)

    search = Searcher(pipe, param_grid)

    search.fit(X_train, y_train)

    print("Best parameter (CV score=%0.3f):" % search.best_score_)

    print(search.best_params_)

    return search.best_estimator_, search.best_params_, search.best_score_

reg, params, performance = grid_search(SGDRegressor, param_grid, X_train, y_train, pre_scale=pre_scale)
y_val_pred = reg.predict(X_val)
plt.scatter(y_val, y_val_pred, color='black', marker='x')

#%%

search = GridSearchCV(pipe, param_grid)
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

reg = search.best_estimator_
y_val_pred = reg.predict(X_val)

plt.scatter(y_val, y_val_pred, color='black', marker='x')


# %%
