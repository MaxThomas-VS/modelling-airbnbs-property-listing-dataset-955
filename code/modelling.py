#%%
import argument_parser as ap
import numpy as np 
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tabular_data import ModelData
import torch


# %%



if __name__ == '__main__':
    args = ap.make_args()
    print(args)

    model_data = ModelData()

    df = model_data.load_tabular_data(args['file_name'], args['import_path'])

    to_model = model_data.extract_label(df, 'Price_Night', numeric_only=True)

    X_train, X_test, y_train, y_test = train_test_split(to_model[0], to_model[1], test_size=0.2, random_state=1)

    reg_mdl = SGDRegressor(max_iter=1000, tol=1e-3)
    reg_mdl.fit(X_train, y_train)
    print(reg_mdl.score(X_test, y_test))