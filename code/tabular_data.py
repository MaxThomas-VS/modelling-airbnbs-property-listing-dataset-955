#%%
import argument_parser as ap
from pathlib import Path
import pandas as pd


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
        df = self.drop_rows_with_missing(df, ['Description'])
        df = self.set_default_feature_values(df, ['guests','beds','bathrooms','bedrooms'], 1)
        df = self.make_numeric(df, ['bedrooms'])
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

    # make d


# %%
