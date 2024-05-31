# Import libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer 

# Define column types
target_col = 'binary_target'
categorical_cols = ['регион', 'использование', 'pack']
continuous_cols = ['сумма', 'частота_пополнения', 'доход', 'сегмент_arpu', 
                   'частота', 'объем_данных', 'on_net', 'продукт_1', 
                   'продукт_2', 'секретный_скор', 'pack_freq', 'binary_target']
drop_col = ['client_id', 'mrg_', 'зона_1', 'зона_2']
threshold = 0.5

def import_data(path_to_file):

    # Get input dataframe
    input_df = pd.read_csv(path_to_file)

    return input_df


# Main preprocessing function
def run_preproc(input_df):

    input_df['null_count'] = input_df.isna().sum(axis=1)
    output_df = input_df.drop(columns=drop_col, axis=1)
    output_df[categorical_cols] = output_df[categorical_cols].fillna('Пропуск')
    
    # Return resulting dataset
    return output_df