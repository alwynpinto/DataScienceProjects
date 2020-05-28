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

# To account for drive age with Back Blaze 5 Smart metrics
frame_8raw = frame[['date', 'serial_number', 'model', 'capacity_bytes', 'failure', 'smart_5_raw',  'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw', 'smart_9_raw', 'smart_12_raw', 'smart_189_raw']]

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
frame_8raw.to_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame_8raw.csv') # contains only 5 raw by BackBlaze + smart_9 + 12 + 189 which is drive age


############# Import data into DataFrame from Exported clean CSV
df = pd.read_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame_5raw.csv', parse_dates=True, index_col='date').fillna(0).drop('Unnamed: 0', axis=1)

# Fill up missing capacity
missing_capacity = (df[df['capacity_bytes'] != -1].drop_duplicates('model').set_index('model')['capacity_bytes'])
df['capacity_bytes'] = df['model'].map(missing_capacity)
df['capacity_tb'] = round(df['capacity_bytes']/1099511627776,2)

# Export cleaned data to CSV for archiving.
df.to_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame_5raw_clean.csv')


# for back blaze 5 smart + smart_9 + 12 + 189
df_8 = pd.read_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame_8raw.csv', parse_dates=True, index_col='date').fillna(0).drop('Unnamed: 0', axis=1)

# Fill up missing capacity
missing_capacity = (df_8[df_8['capacity_bytes'] != -1].drop_duplicates('model').set_index('model')['capacity_bytes'])
df_8['capacity_bytes'] = df_8['model'].map(missing_capacity)
df_8['capacity_tb'] = round(df_8['capacity_bytes']/1099511627776,2)

# Export cleaned data to CSV for archiving.
df_8.to_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame_8raw_clean.csv')


# END OF DATA WRANGLING AND EXPORTING OF DATA #############################
######*******************************#####################*********

df = pd.read_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame_8raw_clean.csv', parse_dates=True, index_col='date')

# EDA and plots
df_plot = df.reset_index()
df_plot.columns[5:13]
to_plot = df_plot.columns[5:13]

for i in to_plot:
    sns.lmplot('date', i, df_plot, hue='failure', fit_reg=False)
    fig = plt.gcf()
    plt.title('SMART metric against Time: '+ i)
    fig.set_size_inches(10, 5)
    plt.show()


# Removal of outliers in dataset
df_allcol = df_plot.columns[5:13]
df_allcol

out_removed = []    
for i in to_plot:
    outlier_df = []
    mean_df = []
    z = []
    mean_value = []
    count = []
    z = np.abs(stats.zscore(df[i]))
    outlier_df = df.iloc[z>3]
    count = Counter(outlier_df.failure)
    print('The Smart',i, count)
    if count[1]==0:
        mean_df = df.iloc[np.where(z<3)]
        mean_value = mean_df[i].mean()
        print(mean_value)
        df[i] = np.where(z>3,mean_value,df[i])
        out_removed.append(i)

out_removed
# Conclusion: Data set has fail data in outliers and hence not imputed.



#plotting box plots for all the 5 SMART metrics. log applied.
sns.boxplot(df.smart_5_raw[np.log(df.smart_5_raw)>0])
sns.boxplot(df.smart_187_raw[np.log(df.smart_187_raw)>0])
sns.boxplot(df.smart_197_raw[np.log(df.smart_197_raw)>0])
sns.boxplot(df.smart_188_raw[np.log(df.smart_188_raw)>0])
sns.boxplot(df.smart_198_raw[np.log(df.smart_198_raw)>0])
sns.boxplot(df.smart_9_raw[np.log(df.smart_9_raw)>0])
sns.boxplot(df.smart_12_raw[np.log(df.smart_12_raw)>0])
sns.boxplot(df.smart_189_raw[np.log(df.smart_189_raw)>0])


# plot Failure by year   
failure_by_year = df_plot.groupby(df_plot['date'].dt.year).failure.sum()
failure_by_year.plot(kind='bar', title='Sum of HDD Failures by Year')

# Number of Unique HDD models
df_plot.model.nunique()

failure_by_model = df_plot.groupby('model').failure.sum().sort_values(ascending=False).head(10)
failure_by_model.plot(kind='bar', title='Failures-by-HDD-Model')

# Since highest failures occures with models ST or SEAGATE 172/206  = 84% failures happens on ST models.
Counter(df[df.model.str.startswith('ST')].failure)


# Since HDD SEAGATE has the worst failure rate, we will only analyse ST data
df_st = df[df.model.str.startswith('ST')]
df_st.head()


# Define the X and y for train/test data
X_5 = df_st.reset_index().drop(['serial_number', 'model', 'capacity_bytes','capacity_tb','date', 'failure',
                                'smart_9_raw', 'smart_12_raw', 'smart_189_raw'],axis=1)
X_8 = df_st.reset_index().drop(['serial_number', 'model', 'capacity_bytes','capacity_tb','date', 'failure'],axis=1)
y = pd.DataFrame(df_st.reset_index().failure)

# Preprocessing. Min Max Scaler
columnnames5 = X_5.columns
columnnames8 = X_8.columns
mm_scaler = preprocessing.MinMaxScaler()
x5_scaled = mm_scaler.fit_transform(X_5)
x8_scaled = mm_scaler.fit_transform(X_8)

x5_scaled = pd.DataFrame(x5_scaled, columns=columnnames5)
x8_scaled = pd.DataFrame(x8_scaled, columns=columnnames8)
x5_scaled.shape, x8_scaled.shape

df_st.columns
# correlation of 8 Smart metrics against failures
corr = df_st[['failure', 'smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw',
       'smart_9_raw', 'smart_12_raw', 'smart_189_raw']].corr()
corr['failure']
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 10))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

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

# Sources for SMOTE: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
# Source: https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/
# SMOTE for balancing the data
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
