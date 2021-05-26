'''
Import Libraries
'''
import sys

import pandas as pd
import numpy as np

import sqlite3

from sklearn import model_selection
from sklearn.preprocessing import StandardScaler


'''
Defining SQL connection details
'''

sqliteConnection = sqlite3.connect('data/news_popularity.db')


'''
Defining default feature_list
'''

default_feature_list=['timedelta', 'num_hrefs', 'n_comments', 'self_reference_avg_shares', 'num_keywords', 'kw_avg_min', 'kw_avg_avg', 
                      'weekday', 'n_tokens_title', 'data_channel']


'''
Function for preprocessing
'''

def feature_preprocessing(feature_list=default_feature_list, target='shares', data_path=sqliteConnection):

    '''
    Extracting data from all the relevant tables and compiling them into a single df
    '''

    df_articles = pd.read_sql("""SELECT * from articles""", con=sqliteConnection)
    df_description = pd.read_sql("""SELECT * from description""", con=sqliteConnection)
    df_keywords = pd.read_sql("""SELECT * from keywords""", con=sqliteConnection)

    df_compiled = pd.merge(df_articles, df_description, on='ID')
    df_compiled = pd.merge(df_compiled,df_keywords, on='ID')
    df_compiled.dropna(inplace=True)
    
    '''
    Declaring x and y values based on feature list and target inputs
    '''
    x = df_compiled[feature_list]
    y = df_compiled[target]
    
    '''
    One hot encoding for categorical features in x
    '''
    if 'weekday' in feature_list:
        
        one_hot_weekday = pd.get_dummies(df_compiled['weekday'])
        x = x.drop(columns=['weekday'])
        x = pd.concat([x, one_hot_weekday], axis=1)
        
    if 'data_channel' in feature_list:
        
        one_hot_data_channel = pd.get_dummies(df_compiled['data_channel']) 
        x = x.drop(columns=['data_channel'])
        x = pd.concat([x, one_hot_data_channel], axis=1)
        
    '''
    Splitting the dataset into training and testing sets
    '''
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.1, random_state = 2020)

    '''
    Using standardscaler to standardise dataset so that a feature will not have a variance that is orders of magnitude larger than
    others as it might dominate the objective function and make the estimator unable to learn from other features correctly 
    as expected.
    '''
    
    scaler = StandardScaler()
    scaler.fit(x)
    
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    '''
    Return the values of x and y after preprocessing
    '''
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    
    output=feature_preprocessing()
    # sys.stdout(feature_preprocessing())
    # np.savetxt(sys.stdout, output)    
    print(feature_preprocessing())