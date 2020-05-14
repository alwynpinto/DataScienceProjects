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


# Using Google Colab:
# Source: https://towardsdatascience.com/3-ways-to-load-csv-files-into-colab-7c14fcbdcb92



# Import data into DF
filenames = sorted(glob('Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\AllData\*.csv'))

df_2019=[]
for filename in filenames:
    df = pd.read_csv(filename, index_col=None, header=0, nrows = 1000)
    df_2019.append(df)

frame = pd.concat(df_2019, axis=0, ignore_index=True)
frame.head() , frame.shape
frame.tail() , frame.shape

frame_5 = frame[['date', 'serial_number', 'model', 'capacity_bytes', 'failure', 'smart_5_normalized', 'smart_5_raw', 'smart_187_normalized', 'smart_187_raw', 'smart_188_normalized', 'smart_188_raw', 'smart_197_normalized', 'smart_197_raw', 'smart_198_normalized', 'smart_198_raw' ]]
frame_5.head() , frame_5.shape
frame_5.tail() , frame_5.shape

frame_5norm = frame_5[['date', 'serial_number', 'model', 'capacity_bytes', 'failure', 'smart_5_normalized', 'smart_187_normalized', 'smart_188_normalized', 'smart_197_normalized', 'smart_198_normalized']]
frame_5norm.head(), frame_5norm.shape
frame_5norm.tail(), frame_5norm.shape

frame_5raw = frame_5[['date', 'serial_number', 'model', 'capacity_bytes', 'failure', 'smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw' ]]
frame_5raw.head(), frame_5raw.shape
frame_5raw.tail(), frame_5raw.shape

'''
Backblaze's analysis of nearly 40,000 drives showed five SMART metrics that correlate strongly with impending disk drive failure:

SMART 5 - Reallocated_Sector_Count.
SMART 187 - Reported_Uncorrectable_Errors.
SMART 188 - Command_Timeout.
SMART 197 - Current_Pending_Sector_Count.
SMART 198 - Offline_Uncorrectable

Not considering others because different manufacturers' SMART values mean very different. If there is data available for what each SMART means and is standard across all the HDD, we can use these for analysis.

http://www.cropel.com/library/smart-attribute-list.aspx
https://www.hdsentinel.com/smart/index.php

'''

# Export data for further consideration
frame.to_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame.csv') # contains all column data
frame_5.to_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame_5.csv') # contains 5 columns mentioned by BackBlaze
frame_5norm.to_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame_5norm.csv') # contains only 5 normalized by BackBlaze
frame_5raw.to_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame_5raw.csv') # contains only 5 normalized by BackBlaze


############# Import data into DataFrame from Exported clean CSV
df = pd.read_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame_5raw.csv', parse_dates=True, index_col='date').fillna(0).drop('Unnamed: 0', axis=1)

# Fill up missing capacity
missing_capacity = (df[df['capacity_bytes'] != -1].drop_duplicates('model').set_index('model')['capacity_bytes'])
df['capacity_bytes'] = df['model'].map(missing_capacity)
df['capacity_tb'] = round(df['capacity_bytes']/1099511627776,2)

# Export cleaned data to CSV for archiving.
df.to_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame_5raw_clean.csv')


######*******************************#####################*********


# Import cleaned data for analysis
df = pd.read_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame_5raw_clean.csv', parse_dates=True, index_col='date')



# Train Test Split
X = df.reset_index().drop(['serial_number', 'model', 'capacity_bytes','capacity_tb','date', 'failure'],axis=1)
y = pd.DataFrame(df.reset_index().failure)

# Preprocessing. Min Max Scaler
columnnames = X.columns
mm_scaler = preprocessing.MinMaxScaler()
x_scaled = mm_scaler.fit_transform(X)
x_scaled = pd.DataFrame(X, columns=columnnames)
x_scaled.shape


# Sources for SMOTE: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
# Source: https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/

# SMOTE for balancing the data

# Method 1: Upsample minority class and Downsample majority class
'''
from imblearn.pipeline import Pipeline
oversample = SMOTE(sampling_strategy = 0.1, random_state=42)
undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
steps = [('o', oversample), ('u', undersample)]
pipeline = Pipeline(steps=steps)
x_train_scaled_s, y_train_s = pipeline.fit_resample(x_train_scaled, y_train)

Counter(y_train_s)[1]/(Counter(y_train_s)[0]+Counter(y_train_s)[1])
'''


# Method 2: Upsample minority class
sm = SMOTE(random_state=42)
x_scaled_sm , y_sm = sm.fit_sample(x_scaled , y)
print('The percentage of failures now in data = ',Counter(y_sm.failure)[1]/(Counter(y_sm.failure)[1]+Counter(y_sm.failure)[0]))
x_scaled_sm.shape
y_sm.shape

#############################################################################################################################################################
# Data Set 1 for Analysis: Choosing the 5 raw SMART data that is chosen by BackBlaze
# Split Train Test data
x_train, x_test, y_train, y_test = train_test_split(x_scaled_sm, y_sm, test_size=0.3, stratify = y_sm, random_state=42)
x_train.shape, x_test.shape




############################################################################################################################################################
# Data Set 2 for Analysis: Choosing all the columns/SMART data
 
# Import all the data
df_all = pd.read_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame.csv').fillna(0)
df_all_L = df_all[['date', 'serial_number', 'model', 'capacity_bytes','failure']]

# Extract only raw data
df_all_R = df_all.filter(regex='raw')


# Marge raw data and relevent columns
df_all_raw = pd.merge(df_all_L , df_all_R, left_index=True, right_index=True)

# Define the X and y for the raw data
x_all_raw = df_all_raw.drop(['date', 'serial_number', 'model', 'capacity_bytes', 'failure'], axis=1)
y_all_raw = pd.DataFrame(df_all_raw.failure)

Counter(y_all_raw['failure'])
x_all_raw.shape

# Min Max Scaler on Data.
mm_scaler = preprocessing.MinMaxScaler()

x_all_raw_col = x_all_raw.columns
x_all_raw_scaled = pd.DataFrame(mm_scaler.fit_transform(x_all_raw), columns=x_all_raw_col)
x_all_raw_scaled

# Running SMOTE for unbalanced data
# Method 2: Upsample minority class
sm = SMOTE(random_state=42)
x_all_raw_scaled_sm , y_all_raw_sm = sm.fit_sample(x_all_raw_scaled , y)
print('The percentage of failures now in data = ',Counter(y_all_raw_sm.failure)[1]/(Counter(y_all_raw_sm.failure)[1]+Counter(y_all_raw_sm.failure)[0]))
x_all_raw_scaled_sm.shape
y_all_raw_sm.shape

# Split Train Test data
x_ar_sm_train, x_ar_sm_test, y_ar_sm_train, y_ar_sm_test = train_test_split(x_all_raw_scaled_sm, y_all_raw_sm, test_size=0.3, stratify = y_sm, random_state=42)
x_ar_sm_train.shape, x_ar_sm_test.shape
x_ar_sm_test.shape, y_ar_sm_test.shape

###############################################################################################
# Data Set 3: Dimensionality Reduction using PCA

# Use Dimensional Reduction with 99% of the variance can be explained
pca99 = PCA(n_components=0.99)
x_pca99_train = pd.DataFrame(pca99.fit_transform(x_ar_sm_train))
x_pca99_train.shape , y_ar_sm_train.shape

x_pca99_test = pd.DataFrame(pca99.fit_transform(x_ar_sm_test))
x_pca99_test.shape, y_ar_sm_test.shape

# Use Dimensional Reduction with 95% of the variance can be explained
pca95 = PCA(n_components=0.95)
x_pca95_train = pd.DataFrame(pca95.fit_transform(x_ar_sm_train))
x_pca95_train.shape , y_ar_sm_train.shape

x_pca95_test = pd.DataFrame(pca95.fit_transform(x_ar_sm_test))
x_pca95_test.shape, y_ar_sm_test.shape



# Testing all classifiers
# Source: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
'''
Ran optimization for over sampling and under sampling percentages and hte results were:
os = [0.1,0.2,0.3]
us = [0.5,0.6,0.7]

and 0.1 and 0.5 gave a score of 0.839
'''


# Create a function to test all classifiers, feeding xtrain, ytrain, xtest, ytest
# Source: https://www.kaggle.com/paultimothymooney/predicting-breast-cancer-from-nuclear-shape
# Source2:  https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
# Source3: https://towardsdatascience.com/grid-search-for-model-tuning-3319b259367e

#########################################################################




def classify_hdd_failure(x_train, x_test, y_train, y_test):
    start_time_lr = time.time()
    ###### Log Regression #####
    print"-------------Running Logistic Regression________________"
    pipe_lr = Pipeline([('classifier', LogisticRegression())])
    #return lr

    param_grid_lr = [{'classifier' : [LogisticRegression()],
                      'classifier__penalty' : ['l1','l2'],
                      'classifier__solver' : ['sag','saga'], 
                      'classifier__max_iter': [1000]}]
    #return param_grid
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    #return cv
    
    lr_clf = GridSearchCV(pipe_lr, param_grid=param_grid_lr, cv=cv, verbose=True, scoring='accuracy', n_jobs=-1)
    #return lr_clf
    
    lr_clf.fit(x_train, y_train)
    
    #define best parameter variables to run on LR algo.
    lr_max_iter = lr_clf.best_params_['classifier__max_iter']
    lr_penalty = lr_clf.best_params_['classifier__penalty']
    lr_solver = lr_clf.best_params_['classifier__solver']
    
    #return lr_max_iter, lr_penalty, lr_solver
    
    # Now that we have the best params, we will fit the model to the training data and run
    lr_clf_best =  LogisticRegression(max_iter = lr_max_iter , 
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
    
    ################################################################################
    # Random Forest Classifier
    start_time_rf = time.time()
    print"-------------Running Random Forest________________"
    pipe_rf = Pipeline([('classifier', RandomForestClassifier())])
    
    # define the parameters to be used for Grid Search
    param_grid_rf = [
        {'classifier' : [RandomForestClassifier()],
        'classifier__n_estimators' : list(range(10,20,10)),
        'classifier__max_depth' : list(range(1,10,1))}
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
        
    #########################################################################
    # Gradient Boosting Classifier:
    start_time_gbc = time.time()
    print"-------------Running Gradient Boosting________________"
    
    pipe_gbc = Pipeline([('classifier', GradientBoostingClassifier())])
    
    param_grid_gbc = [
        {'classifier' : [GradientBoostingClassifier()],
        'classifier__learning_rate' : [0.001 , 0.01 , 0.1, 1.0],
        'classifier__n_estimators' : [50, 100, 150, 200],
        'classifier__max_depth' : list(range(1,10,1))}
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
    rf_clf_best.fit(x_train, y_train)
    
    # predict on test set
    gbc_clf_predict = gbc_clf.predict(x_test)
    
    # Run all related reports
    print('GBC Accuracy Score: ', accuracy_score(y_test , gbc_clf_predict))
    print('GBC Precision Score: ', precision_score(y_test, gbc_clf_predict))
    print('GBC Recall Score: ', recall_score(y_test, gbc_clf_predict))
    print('GBC F1 Score: ', f1_score(y_test, gbc_clf_predict))
    print('GBC Confusion Matrix: ', confusion_matrix(y_test , gbc_clf_predict))
    print('GBC Classification Report: ', classification_report(y_test, gbc_clf_predict))
    print("------Time taken for GBC: %s minutes" % ((time.time() - start_time_gbc)/60))  
    
    ##################################################################
    # XGB [eXtreme Gradient Boosting]
    # sources: https://cran.r-project.org/web/packages/xgboost/vignettes/xgboost.pdf
    # https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/
    
    start_time_xgb = time.time()
    print"-------------Running eXtreme Gradient Boosting________________"
    from xgboost import XGBClassifier
    
    pipe_xgb = Pipeline([('classifier', XGBClassifier())])
    
    param_grid_gbc = [
        {'classifier' : [XGBClassifier()],
        'classifier__booster' : ['gbtree','dart','gblinear'],
        'classifier__n_estimators' : [50, 100, 150, 200],
        'classifier__max_depth' : list(range(1,10,1)),
        'classifier__learning_rate': [0.001 , 0.01 , 0.1, 1.0]}
        ]
    
    # define the cross validation and grid search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    xgb_clf = GridSearchCV(pipe_xgb, param_grid=param_grid_gbc, cv=cv, verbose=True, scoring='accuracy', n_jobs=-1)
    
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
    print('XGB F1 Score: ', f1_score(y_test, xgb_clf_predict))
    print('XGB Confusion Matrix: \n', confusion_matrix(y_test , xgb_clf_predict))
    print('XGB Classification Report: \n', classification_report(y_test, xgb_clf_predict))
    print("------Time taken for XGB: %s minutes" % ((time.time() - start_time_xgb)/60))
    
    ##################################################################
    # Create an DataFrame with results
    acc_score = []
    prec_score = []
    rcall_score = []
    fone_score = []
    
    # Append LR to scores list
    acc_score.append(accuracy_score(y_test , lr_clf_predict))
    prec_score.append(precision_score(y_test, lr_clf_predict))
    rcall_score.append(recall_score(y_test, lr_clf_predict))
    fone_score.append(f1_score(y_test, lr_clf_predict))
    
    # Append RF to scores list
    acc_score.append(accuracy_score(y_test , rf_clf_predict))
    prec_score.append(precision_score(y_test, rf_clf_predict))
    rcall_score.append(recall_score(y_test, rf_clf_predict))
    fone_score.append(f1_score(y_test, rf_clf_predict))
    
    # Append GBC to scores list
    acc_score.append(accuracy_score(y_test , gbc_clf_predict))
    prec_score.append(precision_score(y_test, gbc_clf_predict))
    rcall_score.append(recall_score(y_test, gbc_clf_predict))
    fone_score.append(f1_score(y_test, gbc_clf_predict))
    
    # Append XGB to scores list
    acc_score.append(accuracy_score(y_test , xgb_clf_predict))
    prec_score.append(precision_score(y_test, xgb_clf_predict))
    rcall_score.append(recall_score(y_test, xgb_clf_predict))
    fone_score.append(f1_score(y_test, xgb_clf_predict))
    report_card = {'Accuracy': acc_score, 'Precision': prec_score, 'Recall':  rcall_score, 'F1': fone_score}
    report_card_df = pd.DataFrame(report_card, index=['LR','RF', 'GBC','XGB'])
    print(report_card_df)
    return report_card_df
        



# Run 1
classify_hdd_failure(x_train, x_test, y_train.ravel(), y_test.ravel())

# Run 2
classify_hdd_failure(x_ar_sm_train, x_ar_sm_test, y_ar_sm_train.ravel(), y_ar_sm_test.ravel() )

# Run 3
classify_hdd_failure(x_pca99_train, x_pca99_test, y_ar_sm_train.ravel(), y_ar_sm_test.ravel())

# Run 4
classify_hdd_failure(x_pca95_train, x_pca95_test, y_ar_sm_train.ravel(), y_ar_sm_test.ravel())