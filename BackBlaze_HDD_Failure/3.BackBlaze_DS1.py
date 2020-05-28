# -*- coding: utf-8 -*-
"""
BackBlaze HDD Failure Dataset
"""
# Import all dependencies
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
import time
import datetime as dt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
from sklearn.decomposition import PCA
from collections import Counter
from numpy import where
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

import xgboost as XGBClassifier
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
random.seed(42)


#############################################################################################################################################################
# Data Set 1 for Analysis: Choosing the 5 raw SMART data that is chosen by BackBlaze
# Split Train Test data


# Sources for SMOTE: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
# Source: https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/

# SMOTE for balancing the data
# Create Train Test data. So not to oversample the test data.
x5_scaled_train, x5_test, y5_train, y5_test = train_test_split(x5_scaled, y, test_size=0.3, stratify = y, random_state=42)
x5_scaled_train.shape, x5_test.shape, y5_train.shape, y5_test.shape
Counter(y5_train.failure), Counter(y5_test.failure)

#Dataset # 3
# Run this for checking results with NO UPSAMPLE or DOWNSAMPLE of data.
classify_hdd_failure(x5_scaled_train, x5_test, y5_train.values.ravel(), y5_test.values.ravel())

# Method 1: Upsample minority class and Downsample majority class
from imblearn.pipeline import Pipeline
oversample = SMOTE(sampling_strategy = 0.2, random_state=42)
undersample = RandomUnderSampler(sampling_strategy=0.3, random_state=42)
steps = [('o', oversample), ('u', undersample)]
pipeline = Pipeline(steps=steps)
x5_scaled_train_s, y5_train_s = pipeline.fit_resample(x5_scaled_train, y5_train)

# the Dataset has now reduced to 354K rows. This move was mainly to reduce the size of the dataset. Computational resources.
x5_scaled_train_s.shape, x5_test.shape

# After the SMOTE, the failure percentage in the data has now increased to 33%. Could look at how the results in the analysis change with this %
Counter(y5_train_s.failure)
print('The percentage of failure in the dataset is now: ', Counter(y5_train_s.failure)[1]/(Counter(y5_train_s.failure)[0]+ Counter(y5_train_s.failure)[1]))

'''
# Method 2: Upsample minority class. This method upsamples the minority class to 50% of the data.
sm = SMOTE(random_state=42)
x_scaled_sm , y_sm = sm.fit_sample(x_scaled , y)
print('The percentage of failures now in data = ',Counter(y_sm.failure)[1]/(Counter(y_sm.failure)[1]+Counter(y_sm.failure)[0]))
x_scaled_sm.shape
y_sm.shape
'''

# Data for train test 
x5_scaled_train_s, x5_test, y5_train_s, y5_test
Counter(y5_test.failure)

classify_hdd_failure(x5_scaled_train_s, x5_test, y5_train_s.values.ravel(), y5_test.values.ravel())


############################################################################################################################################################
# Data Set 2 for Analysis: Choosing 3 additional SMART as suspected by Back Blaze
# smart_9_raw, smart_12_raw, smart_189_raw

# Create Train Test data. So not to oversample the test data.
x8_scaled_train, x8_test, y8_train, y8_test = train_test_split(x8_scaled, y, test_size=0.3, stratify = y, random_state=42)
x8_scaled_train.shape, x8_test.shape, y8_train.shape, y8_test.shape
Counter(y5_train.failure), Counter(y5_test.failure)

#Dataset # 4
# Run this for checking results with NO UPSAMPLE or DOWNSAMPLE of data.
classify_hdd_failure(x8_scaled_train, x8_test, y8_train.values.ravel(), y8_test.values.ravel())


# Method 1: Upsample minority class and Downsample majority class
from imblearn.pipeline import Pipeline
oversample = SMOTE(sampling_strategy = 0.2, random_state=42)
undersample = RandomUnderSampler(sampling_strategy=0.3, random_state=42)
steps = [('o', oversample), ('u', undersample)]
pipeline = Pipeline(steps=steps)
x8_scaled_train_s, y8_train_s = pipeline.fit_resample(x8_scaled_train, y8_train)

# the Dataset has now reduced to 354K rows. This move was mainly to reduce the size of the dataset. Computational resources.
x8_scaled_train_s.shape, x8_test.shape

# After the SMOTE, the failure percentage in the data has now increased to 33%. Could look at how the results in the analysis change with this %
Counter(y8_train_s.failure)
print('The percentage of failure in the dataset is now: ', Counter(y8_train_s.failure)[1]/(Counter(y8_train_s.failure)[0]+ Counter(y8_train_s.failure)[1]))


'''
# Method 2: Upsample minority class. This method upsamples the minority class to 50% of the data.
sm = SMOTE(random_state=42)
x_scaled_sm , y_sm = sm.fit_sample(x_scaled , y)
print('The percentage of failures now in data = ',Counter(y_sm.failure)[1]/(Counter(y_sm.failure)[1]+Counter(y_sm.failure)[0]))
x_scaled_sm.shape
y_sm.shape
'''

# Data for train test 
x8_scaled_train_s, x8_test, y8_train_s, y8_test
classify_hdd_failure(x8_scaled_train_s, x8_test, y8_train_s.values.ravel(), y8_test.values.ravel())
Counter(y8_test.failure)




#########################################################################

# Function for Classification
def classify_hdd_failure(x_train, x_test, y_train, y_test):
    start_time_lr = time.time()
    ###### Log Regression #####
    print("-------------Running Logistic Regression________________")
    pipe_lr = Pipeline([('classifier', LogisticRegression())])
    #return lr

    param_grid_lr = [{'classifier' : [LogisticRegression()],
                      'classifier__penalty' : ['l1','l2'],
                      'classifier__solver' : ['newton_cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                      'classifier__C' : [0.001,0.01,0.1,10,100,1000]}]
    #return param_grid
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    #return cv
    
    lr_clf = GridSearchCV(pipe_lr, param_grid=param_grid_lr, cv=cv, verbose=True, scoring='accuracy', n_jobs=-1)
    #return lr_clf
    
    lr_clf.fit(x_train, y_train)
    
    #define best parameter variables to run on LR algo.
    lr_C = lr_clf.best_params_['classifier__C']
    lr_penalty = lr_clf.best_params_['classifier__penalty']
    lr_solver = lr_clf.best_params_['classifier__solver']
    
    #return lr_max_iter, lr_penalty, lr_solver
    
    # Now that we have the best params, we will fit the model to the training data and run
    lr_clf_best =  LogisticRegression(C = lr_C , 
                                      penalty = lr_penalty,
                                      solver = lr_solver)
    #return lr_clf_best

    lr_clf_best.fit(x_train , y_train)
    lr_clf_predict = lr_clf_best.predict(x_test)
    #return lr_clf_predict
    
    # Run all related reports
    print('LR Accuracy Score: ', accuracy_score(y_test , lr_clf_predict))
    print('LR Precision Score: ', precision_score(y_test, lr_clf_predict))
    print('LR Recall Score: ', recall_score(y_test, lr_clf_predict))
    print('LR F1 Score: ', f1_score(y_test, lr_clf_predict))
    print('LR Confusion Matrix: \n', confusion_matrix(y_test , lr_clf_predict))
    print('LR Classification Report: \n', classification_report(y_test, lr_clf_predict))
    print("------Time taken for LR: %s minutes" % ((time.time() - start_time_lr)/60))
    time_lr = (time.time() - start_time_lr)/60
    print(time_lr)
    ################################################################################
    # Random Forest Classifier
    start_time_rf = time.time()
    print("-------------Running Random Forest________________")
    pipe_rf = Pipeline([('classifier', RandomForestClassifier())])
    
    # define the parameters to be used for Grid Search
    param_grid_rf = [
    {'classifier' : [RandomForestClassifier()],
    'classifier__n_estimators' : list(range(5,20,2)),
    'classifier__max_depth' : list(range(8,22,2))}
    ]
    
    # define the cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    rf_clf = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=cv, verbose=True, scoring='accuracy', n_jobs=-1)
    
    # Fit the model to the Training Dataset
    rf_clf.fit(x_train, y_train)
    
    #define best parameter variables to run on RF algo.
    rf_n_estimators = rf_clf.best_params_['classifier__n_estimators']
    rf_max_depth = rf_clf.best_params_['classifier__max_depth']
    
    # Now that we have the best params, we will fit the model to the training data and run
    rf_clf_best = RandomForestClassifier(n_estimators=rf_n_estimators,
                                         max_depth=rf_max_depth)
    rf_clf_best.fit(x_train, y_train)
    
    # Run Test data to predict
    rf_clf_predict = rf_clf_best.predict(x_test)
    
    # Run all related reports.
    print('RF Accuracy Score: ', accuracy_score(y_test , rf_clf_predict))
    print('RF Precision Score: ', precision_score(y_test, rf_clf_predict))
    print('RF Recall Score: ', recall_score(y_test, rf_clf_predict))
    print('RF F1 Score: ', f1_score(y_test, rf_clf_predict))
    print('RF Confusion Matrix: \n', confusion_matrix(y_test , rf_clf_predict))
    print('RF Classification Report: \n ', classification_report(y_test, rf_clf_predict))
    print("------Time taken for RF: %s minutes" % ((time.time() - start_time_rf)/60))
    time_rf = (time.time() - start_time_rf)/60
    print(time_rf)    
    #########################################################################
    # Gradient Boosting Classifier:
    start_time_gbc = time.time()
    print("-------------Running Gradient Boosting________________")
    
    pipe_gbc = Pipeline([('classifier', GradientBoostingClassifier())])
    
    param_grid_gbc = [
        {'classifier' : [GradientBoostingClassifier()],
        'classifier__learning_rate' : [0.1, 1.0, 2.0],
        'classifier__n_estimators' : [75, 100, 125],
        'classifier__max_depth' : list(range(3,5,1))}
        ]
    
    # define the cross validation and grid search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    gbc_clf = GridSearchCV(pipe_gbc, param_grid=param_grid_gbc, cv=cv, verbose=True, scoring='accuracy', n_jobs=-1)
    
    # Fit the model on training dataset for optm parameters
    gbc_clf.fit(x_train, y_train)
    
    #define best parameter variables to run on GBC algo.
    gbc_learning_rate = gbc_clf.best_params_['classifier__learning_rate']
    gbc_n_estimators = gbc_clf.best_params_['classifier__n_estimators']
    gbc_max_depth = gbc_clf.best_params_['classifier__max_depth']
    
    # Now that we have the best params, we will fit the model to the training data and run
    gbc_clf_best = GradientBoostingClassifier(n_estimators=gbc_n_estimators,
                                              max_depth=gbc_max_depth,
                                              learning_rate=gbc_learning_rate)
    
    # Train the model on optm parameters
    gbc_clf_best.fit(x_train, y_train)
    
    # predict on test set
    gbc_clf_predict = gbc_clf.predict(x_test)
    
    # Run all related reports
    print('GBC Accuracy Score: ', accuracy_score(y_test , gbc_clf_predict))
    print('GBC Precision Score: ', precision_score(y_test, gbc_clf_predict))
    print('GBC Recall Score: ', recall_score(y_test, gbc_clf_predict))
    print('GBC F1 Score: ', f1_score(y_test, gbc_clf_predict))
    print('GBC Confusion Matrix: \n', confusion_matrix(y_test , gbc_clf_predict))
    print('GBC Classification Report: \n', classification_report(y_test, gbc_clf_predict))
    print("------Time taken for GBC: %s minutes" % ((time.time() - start_time_gbc)/60)) 
    time_gbc = (time.time() - start_time_gbc)/60
    print(time_gbc)
    ##################################################################
    # XGB [eXtreme Gradient Boosting]
    # sources: https://cran.r-project.org/web/packages/xgboost/vignettes/xgboost.pdf
    # https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/
    
    start_time_xgb = time.time()
    print("-------------Running eXtreme Gradient Boosting________________")
    from xgboost import XGBClassifier
    
    pipe_xgb = Pipeline([('classifier', XGBClassifier())])
    
    param_grid_xgb = [
    {'classifier' : [XGBClassifier()],
    'classifier__booster' : ['gbtree','dart'],
    'classifier__n_estimators' : [20, 40, 50],
    'classifier__max_depth' : list(range(0,1,1)),
    'classifier__learning_rate': [0.001, 0.01 , 0.05],
    }
    ]
    
    # define the cross validation and grid search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    xgb_clf = GridSearchCV(pipe_xgb, param_grid=param_grid_xgb, cv=cv, verbose=True, scoring='accuracy', n_jobs=-1)
    
    # Fit the model on training dataset for optm parameters
    xgb_clf.fit(x_train, y_train)
    
    #define best parameter variables to run on GBC algo.
    xgb_learning_rate = xgb_clf.best_params_['classifier__learning_rate']
    xgb_booster = xgb_clf.best_params_['classifier__booster']
    xgb_n_estimators = xgb_clf.best_params_['classifier__n_estimators']
    xgb_max_depth = xgb_clf.best_params_['classifier__max_depth']
    
    # Now that we have the best params, we will fit the model to the training data and run
    xgb_clf_best = XGBClassifier(booster=xgb_booster,
                                 n_estimators=xgb_n_estimators,
                                 max_depth=xgb_max_depth,
                                 learning_rate=xgb_learning_rate)
    
    # Train the model on optm parameters
    xgb_clf_best.fit(x_train, y_train)
    
    # predict on test set
    xgb_clf_predict = xgb_clf.predict(x_test)
    
    # Run all related reports
    print('XGB Accuracy Score: ', accuracy_score(y_test , xgb_clf_predict))
    print('XGB Precision Score: ', precision_score(y_test, xgb_clf_predict))
    print('XGB Recall Score: ', recall_score(y_test, xgb_clf_predict))
    print('XGB F1 Score: ', f1_score(y_test, xgb_clf_predict, average='weighted'))
    print('XGB Confusion Matrix: \n', confusion_matrix(y_test , xgb_clf_predict))
    print('XGB Classification Report: \n', classification_report(y_test, xgb_clf_predict))
    print("------Time taken for XGB: %s minutes" % ((time.time() - start_time_xgb)/60))
    time_xgb = (time.time() - start_time_xgb)/60
    print(time_xgb)
    ##################################################################
    # Create an DataFrame with results
    acc_score = []
    prec_score = []
    rcall_score = []
    fone_score = []
    time_model = []
    
    # Append LR to scores list
    acc_score.append(accuracy_score(y_test , lr_clf_predict))
    prec_score.append(precision_score(y_test, lr_clf_predict))
    rcall_score.append(recall_score(y_test, lr_clf_predict))
    fone_score.append(f1_score(y_test, lr_clf_predict))
    time_model.append(time_lr)
    
    # Append RF to scores list
    acc_score.append(accuracy_score(y_test , rf_clf_predict))
    prec_score.append(precision_score(y_test, rf_clf_predict))
    rcall_score.append(recall_score(y_test, rf_clf_predict))
    fone_score.append(f1_score(y_test, rf_clf_predict))
    time_model.append(time_rf)
    
    # Append GBC to scores list
    acc_score.append(accuracy_score(y_test , gbc_clf_predict))
    prec_score.append(precision_score(y_test, gbc_clf_predict))
    rcall_score.append(recall_score(y_test, gbc_clf_predict))
    fone_score.append(f1_score(y_test, gbc_clf_predict))
    time_model.append(time_gbc)
        
    # Append XGB to scores list
    acc_score.append(accuracy_score(y_test , xgb_clf_predict))
    prec_score.append(precision_score(y_test, xgb_clf_predict))
    rcall_score.append(recall_score(y_test, xgb_clf_predict))
    fone_score.append(f1_score(y_test, xgb_clf_predict))
    time_model.append(time_xgb)
    
    report_card = {'Accuracy': acc_score, 'Precision': prec_score, 'Recall':  rcall_score, 'F1': fone_score, 'Time_Model': time_model}
    report_card_df = pd.DataFrame(report_card, index=['LR', 'RF', 'GBC', 'XGB'])
    print(report_card_df)
    return report_card_df
        
##########################################################
