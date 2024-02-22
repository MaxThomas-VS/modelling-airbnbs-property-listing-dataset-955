#%%
import argument_parser as ap
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



class CleanTabularData():

    def drop_rows_with_missing(self, df, columns):
        df = df.dropna(subset=columns)
        df.reset_index(drop=True, inplace=True)
        return df
    
    def combine_description_strings(self, df):
        description = df['Description']
        description = description.str.replace("\'About this space\', ", "")
        description = description.str.replace("\'The space\', ", "")

        description = description.str.replace("[\"","")
        description = description.str.replace("]\"","")
        description = description.str.replace("[\'","")
        description = description.str.replace("]\'","")

        description = description.str.replace("\"[","")
        description = description.str.replace("\"]","")
        description = description.str.replace("\'[","")
        description = description.str.replace("\']","")

        description_list = description.str.split()

        for ix in description_list.index:
            try:
                description_list[ix].remove(' ')
            except:
                pass

            try:
                description_list[ix] = ' '.join(description_list[ix])
            except:
                description_list[ix] = pd.NaT

        df['Description'] = description_list

        return df
    
    def drop_columns(self, df, columns):
        df = df.drop(columns, axis=1)
        return df
    
    def set_default_feature_values(self, df, columns, value):
        for col in columns:
            df[col].fillna(value, inplace=True)
        return df
    
    def make_numeric(self, df, columns):
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df


    def clean_tabular_data(self, df):
        cleaner = CleanTabularData()
        df = self.drop_columns(df, ['Unnamed: 19'])
        df = self.drop_rows_with_missing(df, ['Accuracy_rating', 'Check-in_rating', 'Value_rating', 'Location_rating'])
        df = self.combine_description_strings(df)
        df = self.set_default_feature_values(df, ['guests','beds','bathrooms','bedrooms'], 1)
        df = self.make_numeric(df, ['bedrooms'])
        df = self.drop_rows_with_missing(df, df.columns)
        return df


class ModelData():

    def load_tabular_data(self, filename, filepath):
        df = pd.read_csv(filepath+filename)
        return df

    def is_numeric(self, df):
        isnumeric = []
        for column in df.columns:
            if df[column].dtype.kind in 'biufc':
                isnumeric.append(column)
        return isnumeric

    def not_numeric(self, df):
        isntnumeric = []
        for column in df.columns:
            if not df[column].dtype.kind in 'biufc':
                isntnumeric.append(column)
        return isntnumeric

    def extract_label(self, df, column, numeric_only=True):
        label = df[column]
        if numeric_only:
            isnt_numeric = self.not_numeric(df)
            isnt_numeric.append(column)
            columns_to_drop = isnt_numeric
        else:
            columns_to_drop = label
        df = df.drop(columns_to_drop, axis=1)
        return df, label
    
def get_tabular_data(args, test_size=0.3, rstate=None, column_overide=None, one_hot_encode_labels=False):
    '''
    Load tabular data and split it into training and testing sets.
    
    Arguments:
    args: a dictionary containing the file name, import path, label, random state and ml model
    test_size: the proportion of the data to be used for testing
    rstate: the random state
    column_overide: the column to be used as the label. If None, Price_Night is the label
    one_hot_encode_labels: whether to one hot encode the labels
    
    Returns:
    X_train: the training data
    X_test: the testing data
    y_train: the training labels
    y_test: the testing labels
    '''
    if rstate is None:
        rstate = args['random_state']

    column = args['label']

    if column_overide is not None:
        column = column_overide

    model_data = ModelData()

    df = model_data.load_tabular_data(args['file_name'], args['import_path'])

    to_model = model_data.extract_label(df, column, numeric_only=True)

    X_train, X_test, y_train, y_test = train_test_split(to_model[0], to_model[1], test_size=test_size, random_state=rstate)

    if one_hot_encode_labels:
        y_train = one_hot_encode_column(y_train)
        y_test = one_hot_encode_column(y_test)

    return X_train, X_test, y_train, y_test

def one_hot_encode_column(pds):
    '''
    One hot encode a pandas series.

    Arguments:
    pds: a pandas series to encode

    Returns:
    oh_labels: the one hot encoded labels as integers
    '''
    #oh_encoder = OneHotEncoder(sparse_output=False, drop='first')
    oh_encoder = OneHotEncoder(sparse_output=False)
    label_encoder = LabelEncoder()
    pds_2 = label_encoder.fit_transform(pds).reshape(-1,1)
    oh_encoder.fit(pds_2)
    oh_labels = oh_encoder.transform(pds_2)
    return np.argmax(oh_labels, axis=1)
        

if __name__ == '__main__':
    arguments = ap.make_args()
    print(arguments)

    model_data = ModelData()
    cleaner = CleanTabularData()

    df = model_data.load_tabular_data(arguments['file_name'], arguments['import_path'])

    df = cleaner.clean_tabular_data(df)

    print(df.head())
    print(df.info())

    Path(arguments['export_path']).mkdir(parents=True, exist_ok=True)

    df.to_csv(arguments['export_path'] + arguments['file_name'], index=False)

    df.info()
    # make d


# %%
