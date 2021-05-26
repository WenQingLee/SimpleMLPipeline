import sys

import pandas as pd
import numpy as np

import sqlite3

from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model, svm
from sklearn import metrics, preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, BaggingClassifier

import preprocessing as pp


'''
Variable declaration based on shell script input
'''
model_select = sys.argv[1]

try:
    feature_list = [feature for feature in sys.argv[2:]]
except:
    pass

if feature_list:
    pass
else:
    feature_list = ['timedelta', 'num_hrefs', 'n_comments', 'self_reference_avg_shares', 'num_keywords', 'kw_avg_min', 'kw_avg_avg', 
                          'weekday', 'n_tokens_title', 'data_channel']


'''
Function for the ML output
'''
def ml_output(model_select, feature_list):
    
    if model_select == 'ElasticNet':
        ElasticNet_estimator(feature_list)
    elif model_select == 'GradientBoosting':
        GradientBoosting_estimator(feature_list)
    elif model_select == 'SupportVector_Bagging':
        SupportVector_Bagging_estimator(feature_list)
    elif model_select == 'LogisticRegression_Bagging':
        LogisticRegression_Bagging_estimator(feature_list)
        
        
        

'''
Hyperparameter tuning with randomisedCV
Mean Absolute Error (MAE) will be the metric used for evaluation. It is more robust to outliers and 
does not penalize the errors as extremely as mse
'''

def hyper_para(preprocess_input=pp.feature_preprocessing(), model_type='ElasticNet'):
    
    '''
    Define x and y for randomisedCV
    '''
    x_train=preprocess_input[0]
    y_train=preprocess_input[2]
    
    
    '''
    Specifying different grid parameters, metrics and estimators accordingly.
    fit_intercept is set to False as no features(e.g. timedelta) should mean no shares
    '''
    
    if model_type=='SGDRegressor':
        
        random_grid = {'alpha' : [0.01, 0.03, 0.07 , 0.1, 0.3, 0.7, 1, 3, 7, 10],
                               'fit_intercept' : [False],
                               'max_iter' : [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]}

        metric_scorer = metrics.make_scorer(metrics.mean_absolute_error)
        
        base_estimator = linear_model.SGDRegressor()
        
    elif model_type=='ElasticNet':
        
        random_grid = {'alpha' : [0.01, 0.03, 0.07 , 0.1, 0.3, 0.7, 1, 3, 7, 10],
                       'fit_intercept' : [False],
                       'max_iter' : [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]}

        metric_scorer = metrics.make_scorer(metrics.mean_absolute_error)
        
        base_estimator = linear_model.ElasticNet()

    # Fitting the RandomSearch model
    base_randomCV = RandomizedSearchCV(estimator = base_estimator, 
                                      param_distributions = random_grid, 
                                      n_iter = 20, 
                                      cv = 5, 
                                      verbose=2, 
                                      random_state=2020,
                                      scoring=metric_scorer,
                                      n_jobs = -1)

    base_randomCV_results = base_randomCV.fit(x_train, y_train)

    # Returning the results
    base_max_iter = base_randomCV_results.best_params_['max_iter']
    base_intercept = base_randomCV_results.best_params_['fit_intercept']
    base_alpha = base_randomCV_results.best_params_['alpha']

    return base_max_iter, base_intercept, base_alpha


'''
Function for metric results
'''
def metric_results(prediction, actual):
    print('---'*20)
    print('Mean Squared Error is', metrics.mean_squared_error(prediction, actual))
    print('Root Mean Squared Error is', metrics.mean_squared_error(prediction, actual, squared=False))
    print('Mean Absolute Error is', metrics.mean_absolute_error(prediction, actual))
    print('---'*20)


'''
Function to bin the shares into thousands - Used in classification models
'''

def bin_shares(y_train, y_test):
    
    y_train=[round(num, -3) for num in y_train]
    y_test=[round(num, -3) for num in y_test]
    
    return y_train, y_test




'''
Function for ElasticNet Model
'''

def ElasticNet_estimator(feature_list):
    
    # Data preprocessing
    x_train, x_test, y_train, y_test = pp.feature_preprocessing(feature_list)
    
    # Hyperparameter tuning with RandomGrid
    model_max_iter, model_intercept, model_alpha = hyper_para(pp.feature_preprocessing(feature_list))

    # Model based on hyperparameters derived
    regression_estimator = linear_model.ElasticNet(max_iter = model_max_iter, 
                                                   alpha = model_alpha,
                                                   fit_intercept = model_intercept)

    # Training the model
    regression_estimator.fit(x_train, y_train)

    # Model Prediction
    regression_predict = regression_estimator.predict(x_test)

    # Model results based on metrics
    metric_results(regression_predict, y_test)


'''
Function for Gradient Boosting model
'''
def GradientBoosting_estimator(feature_list):

    # Data preprocessing
    x_train, x_test, y_train, y_test = pp.feature_preprocessing(feature_list)
    
    # Gradient Boosting model
    grad_boost = GradientBoostingRegressor(random_state = 2020)

    # Training the model
    grad_boost.fit(x_train, y_train)
    
    # Model Prediction
    grad_boost_predict = grad_boost.predict(x_test)

    # Model results based on metrics
    metric_results(grad_boost_predict, y_test)


'''
Function for SupportVector_Bagging model
'''
def SupportVector_Bagging_estimator(feature_list):

    # Data preprocessing
    x_train, x_test, y_train, y_test = pp.feature_preprocessing(feature_list)
    
    # Binning the shares attributes
    bin_y_train, bin_y_test = bin_shares(y_train, y_test)
    
    # Bagging classifier model
    bag_class = BaggingClassifier(base_estimator=svm.SVC(random_state=2020),
                                  random_state = 2020)

    # Training the model
    bag_class.fit(x_train, bin_y_train)
    
    # Model prediction
    bag_predict = bag_class.predict(x_test)

    # Model results based on metrics
    metric_results(bag_predict, y_test)
    

'''
Function for LogisticRegression_Bagging model
'''
def LogisticRegression_Bagging_estimator(feature_list):

    # Data preprocessing
    x_train, x_test, y_train, y_test = pp.feature_preprocessing(feature_list)
    
    # Binning the shares attributes
    bin_y_train, bin_y_test = bin_shares(y_train, y_test)
    
    
    '''
    Hyperparameter tuning with randomgrid
    '''
    random_grid = {'C' : [0.01, 0.03, 0.07 , 0.1, 0.3, 0.7, 1, 3, 7, 10],
               'fit_intercept' : [False],
               'max_iter' : [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]}

    # Metric set as Fbeta instead. Fbeta of 1 equivalent to F1 score
    metric_scorer = metrics.make_scorer(metrics.fbeta_score, beta=1.0, average='weighted')

    # Logistic Regression model
    base_log = linear_model.LogisticRegression()

    base_randomCV = RandomizedSearchCV(estimator = base_log, 
                                      param_distributions = random_grid, 
                                      n_iter = 20, 
                                      cv = 5, 
                                      verbose=2, 
                                      random_state=2020,
                                      scoring=metric_scorer,
                                      n_jobs = -1)

    base_randomCV_results = base_randomCV.fit(x_train, bin_y_train)

    base_max_iter = base_randomCV_results.best_params_['max_iter']
    base_intercept = base_randomCV_results.best_params_['fit_intercept']
    base_c = base_randomCV_results.best_params_['C']
    
    # Specifying the logistic regression model with hyperparameters from randomsearch
    log_estimator = linear_model.LogisticRegression(solver = 'lbfgs', 
                                                    max_iter = base_max_iter, 
                                                    C=base_c, 
                                                    fit_intercept = base_intercept,
                                                    random_state=2020)
    
    # Bagging classifier model
    bag_class = BaggingClassifier(base_estimator=log_estimator,
                                  random_state = 2020)

    # Training the model
    bag_class.fit(x_train, bin_y_train)
    
    # Model Prediction
    bag_predict = bag_class.predict(x_test)
    
    # Model results based on metrics
    metric_results(bag_predict, y_test)



'''
Function for support vector machines - Legacy and will not be used
'''

# def svm_svc(feature_list):

#     # Data preprocessing
#     x_train, x_test, y_train, y_test = pp.feature_preprocessing(feature_list)
    
#     # Binning the shares attributes
#     bin_y_train, bin_y_test = bin_shares(y_train, y_test)

#     # Support vector machine model
#     clf=svm.SVC(random_state=2020)

#     # Training the model
#     clf.fit(x_train, bin_y_train)

#     # Model Prediction
#     svm_predict=clf.predict(x_test)

#     # Model results based on metrics
#     metric_results(svm_predict, y_test)


'''
Function for support vector machines-LinearSVC - Legacy and will not be used
'''

# def svm_LinearSVC():

#     # Data preprocessing
#     x_train, x_test, y_train, y_test = pp.feature_preprocessing(feature_list)
    
#     # Binning the shares attributes
#     bin_y_train, bin_y_test = bin_shares(y_train, y_test)

#     # Support vector machine - LinearSVC model
#     clf=svm.LinearSVC(random_state=2020)

#     # Training the model
#     clf.fit(x_train, bin_y_train)

#     # Model Prediction
#     svm_predict=clf.predict(x_test)

#     # Model results based on metrics
#     metric_results(svm_predict, y_test)
    

'''
Function for Stochastic Gradient Descent Model - Legacy and will not be used
'''
# def SGD_estimator(feature_list):

#     # Data preprocessing
#     x_train, x_test, y_train, y_test = pp.feature_preprocessing(feature_list)
    
#     # Hyperparameter tuning with RandomGrid
#     model_max_iter, model_intercept, model_alpha = hyper_para(pp.feature_preprocessing(feature_list))
    
#     # Model based on hyperparameters derived
#     regression_estimator = linear_model.SGDRegressor(max_iter = model_max_iter, 
#                                                      alpha = model_alpha,
#                                                      fit_intercept = model_intercept)

#     # Training the model
#     regression_estimator.fit(x_train, y_train)

#     # Model Prediction
#     regression_predict = regression_estimator.predict(x_test)

#     # Model results based on metrics
#     metric_results(regression_predict, y_test)



if __name__ == "__main__":

    print(ml_output(model_select, feature_list))
    print('Model selected:', model_select)
    print('Features selected:')
    print(feature_list)
