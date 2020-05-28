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

# EDA and plots
df_plot = df.reset_index()
df_plot.columns[5:10]

to_plot = df_plot.columns[5:10]

for i in to_plot:
    sns.lmplot('date', i, df_plot, hue='failure', fit_reg=False)
    fig = plt.gcf()
    plt.title('SMART metric against Time: '+ i)
    fig.set_size_inches(10, 5)
    plt.show()


# Removal of outliers in dataset
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

# Conclusion: Data set has fail data in outliers and hence not imputed.



#plotting box plots for all the 5 SMART metrics. log applied.
sns.boxplot(df.smart_5_raw[np.log(df.smart_5_raw)>0])
sns.boxplot(df.smart_187_raw[np.log(df.smart_187_raw)>0])
sns.boxplot(df.smart_197_raw[np.log(df.smart_197_raw)>0])
sns.boxplot(df.smart_188_raw[np.log(df.smart_188_raw)>0])
sns.boxplot(df.smart_198_raw[np.log(df.smart_198_raw)>0])

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
df_st



# Train Test Split
X = df_st.reset_index().drop(['serial_number', 'model', 'capacity_bytes','capacity_tb','date', 'failure'],axis=1)
y = pd.DataFrame(df_st.reset_index().failure)

# Preprocessing. Min Max Scaler
columnnames = X.columns
mm_scaler = preprocessing.MinMaxScaler()
std_scaler = preprocessing.StandardScaler()
x_scaled = std_scaler.fit_transform(X)
x_scaled = pd.DataFrame(X, columns=columnnames)
x_scaled.shape


# correlation of 5 Smart metrics against failures
corr = df.corr()
corr['failure']


#############################################################################################################################################################
# Data Set 1 for Analysis: Choosing the 5 raw SMART data that is chosen by BackBlaze
# Split Train Test data


# Sources for SMOTE: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
# Source: https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/

# SMOTE for balancing the data
# Create Train Test data. So not to oversample the test data.
x_scaled_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, stratify = y, random_state=42)
x_scaled_train.shape, x_test.shape


# Method 1: Upsample minority class and Downsample majority class
from imblearn.pipeline import Pipeline
oversample = SMOTE(sampling_strategy = 0.2, random_state=42)
undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
steps = [('o', oversample), ('u', undersample)]
pipeline = Pipeline(steps=steps)
x_scaled_train_s, y_train_s = pipeline.fit_resample(x_scaled_train, y_train)

# the Dataset has now reduced to 304K rows. This move was mainly to reduce the size of the dataset. Computational resources.
x_scaled_train_s.shape

# After the SMOTE, the failure percentage in the data has now increased to 33%. Could look at how the results in the analysis change with this %
Counter(y_train_s.failure)
print('The percentage of failure in the dataset is now: ', Counter(y_train_s.failure)[1]/(Counter(y_train_s.failure)[0]+ Counter(y_train_s.failure)[1]))


'''
# Method 2: Upsample minority class. This method upsamples the minority class to 50% of the data.
sm = SMOTE(random_state=42)
x_scaled_sm , y_sm = sm.fit_sample(x_scaled , y)
print('The percentage of failures now in data = ',Counter(y_sm.failure)[1]/(Counter(y_sm.failure)[1]+Counter(y_sm.failure)[0]))
x_scaled_sm.shape
y_sm.shape
'''


# Data set 1 for train test 
x_scaled_train_s, x_test, y_train_s, y_test




############################################################################################################################################################
# Data Set 2 for Analysis: Choosing all the columns/SMART data
 
# Import all the data
df_all = pd.read_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame.csv').fillna(0)

df_all_ST = df_all[df_all.model.str.startswith('ST')]

df_all_L = df_all_ST[['date', 'serial_number', 'model', 'capacity_bytes','failure']]

# Extract only raw data
df_all_R = df_all_ST.filter(regex='raw')


# Marge raw data and relevent columns
df_all_raw = pd.merge(df_all_L , df_all_R, left_index=True, right_index=True)
df_all_raw.shape

# to plot all raw values against time and hue by failure. Take a call on 
# if the data need outlier replacement if no failure in outlier data.

to_plot_all_raw = df_all_raw.columns[5:]
for i in to_plot_all_raw:
    sns.lmplot('date', i, df_all_raw, hue='failure', fit_reg=False)
    fig = plt.gcf()
    plt.title('SMART metric against Time: '+ i)
    fig.set_size_inches(10, 5)
    plt.show()
    
# To plot individual SMART plots
smart='smart_18_raw'    
sns.lmplot('date', smart, df_all_raw, hue='failure', fit_reg=False)
fig = plt.gcf()
plt.title('SMART metric against Time: '+ smart)
fig.set_size_inches(10, 5)
plt.show()    

# Columns whose outliers need to be removed.
outlier_remove = df_all_raw.columns[5:]


#Check correlation of raw variable against the rest to decide to eliminate outliers.
corr_all = df_all_raw.corr()



# Calculate mode/mean of all columns in out_remove and store for imputing
df_all_raw = pd.merge(df_all_L , df_all_R, left_index=True, right_index=True)
out_removed = [] # create an empty list to track columsn whose outliers were removed

for i in outlier_remove:
    outlier_df = []
    mean_df = []
    z = []
    mean_value = []
    count = []
    z = np.abs(stats.zscore(df_all_raw[i]))
    outlier_df = df_all_raw.iloc[z>3]
    count = Counter(outlier_df.failure)
    print('The Smart',i, count)
    if count[1]==0:
        mean_df = df_all_raw.iloc[np.where(z<3)]
        mean_value = mean_df[i].mean()
        print(mean_value)
        df_all_raw[i] = np.where(z>3,mean_value,df_all_raw[i])
        out_removed.append(i)

out_removed


# To Check if all outlier removal has been executed
for i in out_removed:
    sns.lmplot('date', i, df_all_raw, hue='failure', fit_reg=False)
    fig = plt.gcf()
    plt.title('SMART metric against Time: '+ i)
    fig.set_size_inches(10, 5)
    plt.show()


# Alternative code to remove outliers using IQR technique
'''
np.random.seed(42)
def remove_outlier(df):
    low = .05
    high = .95
    quant_df = df.quantile([low, high])
    for i in out_remove:
        outlier_df = []
        mode_df = []
        z = []
        mode_value = []
        count = []
        z = np.abs(stats.zscore(df[i]))
        outlier_df = df.iloc[np.where(z>3)]
        count = Counter(outlier_df.failure)
        print(i, count)
        if count[1]==0:
            df = df[(df[i] > quant_df.loc[low, i]) & (df[i] < quant_df.loc[high, i])]
            return df

remove_outlier(df_all_raw)
'''


# Define the X and y for the raw data
x_all_raw = df_all_raw.drop(['date', 'serial_number', 'model', 'capacity_bytes', 'failure'], axis=1)
y_all_raw = pd.DataFrame(df_all_raw.failure)
x_all_raw.shape, y_all_raw.shape

Counter(y_all_raw['failure'])


# Min Max Scaler on Data.
mm_scaler = preprocessing.MinMaxScaler()
std_scaler = preprocessing.StandardScaler()
x_all_raw_col = x_all_raw.columns
x_all_raw_scaled = pd.DataFrame(mm_scaler.fit_transform(x_all_raw), columns=x_all_raw_col)
x_all_raw_scaled


# Split Train Test data
x_ar_scaled_train, x_ar_scaled_test, y_ar_train, y_ar_test = train_test_split(x_all_raw_scaled, y_all_raw, test_size=0.3, stratify = y_all_raw, random_state=42)
x_ar_scaled_train.shape, x_ar_scaled_test.shape, y_ar_train.shape, y_ar_test.shape



# Running SMOTE for unbalanced data
# Method 1: Upsample minority class and Downsample majority class
from imblearn.pipeline import Pipeline
oversample = SMOTE(sampling_strategy = 0.2, random_state=42)
undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
steps = [('o', oversample), ('u', undersample)]
pipeline = Pipeline(steps=steps)
x_all_raw_scaled_train_sm, y_all_raw_train_sm = pipeline.fit_resample(x_ar_scaled_train, y_ar_train)

'''
# Method 2: Upsample minority class
sm = SMOTE(random_state=42)
x_all_raw_scaled_sm , y_all_raw_sm = sm.fit_sample(x_all_raw_scaled , y)
print('The percentage of failures now in data = ',Counter(y_all_raw_sm.failure)[1]/(Counter(y_all_raw_sm.failure)[1]+Counter(y_all_raw_sm.failure)[0]))
x_all_raw_scaled_sm.shape
y_all_raw_sm.shape
'''

# Train Data
x_all_raw_scaled_train_sm.shape, y_all_raw_train_sm.shape
# Test Data
x_ar_scaled_test.shape, y_ar_test.shape

# number of failure in train data after smote
fail_after_sm = Counter(y_all_raw_train_sm.failure)[1]
non_fail_after_sm = Counter(y_all_raw_train_sm.failure)[0]


print('The percentage of failure in SM Train data', (fail_after_sm*100/(fail_after_sm+non_fail_after_sm)))

#number of failures in test data
Counter(y_ar_test.failure)


###############################################################################################
# Data Set 3: Dimensionality Reduction using PCA

# Use Dimensional Reduction with 99% of the variance can be explained
pca99 = PCA(n_components=0.99)
x_pca99_train = pd.DataFrame(pca99.fit_transform(x_all_raw_scaled_train_sm))
x_pca99_train.shape , y_all_raw_train_sm.shape

x_pca99_test = pd.DataFrame(pca99.fit_transform(x_ar_scaled_test))
x_pca99_test.shape, y_ar_test.shape

# Use Dimensional Reduction with 95% of the variance can be explained
pca95 = PCA(n_components=0.95)
x_pca95_train = pd.DataFrame(pca95.fit_transform(x_all_raw_scaled_train_sm))
x_pca95_train.shape , y_all_raw_train_sm.shape

x_pca95_test = pd.DataFrame(pca95.fit_transform(x_ar_scaled_test))
x_pca95_test.shape, y_ar_test.shape



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
    print("-------------Running Logistic Regression________________")
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
    time_lr = (time.time() - start_time_lr)/60
    
    ################################################################################
    # Random Forest Classifier
    start_time_rf = time.time()
    print("-------------Running Random Forest________________")
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
    time_rf = (time.time() - start_time_rf)/60
        
    #########################################################################
    # Gradient Boosting Classifier:
    start_time_gbc = time.time()
    print("-------------Running Gradient Boosting________________")
    
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
    '''
    ##################################################################
    # XGB [eXtreme Gradient Boosting]
    # sources: https://cran.r-project.org/web/packages/xgboost/vignettes/xgboost.pdf
    # https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/
    
    start_time_xgb = time.time()
    print("-------------Running eXtreme Gradient Boosting________________")
    from xgboost import XGBClassifier
    
    pipe_xgb = Pipeline([('classifier', XGBClassifier())])
    
    param_grid_gbc = [
        {'classifier' : [XGBClassifier()],
        'classifier__booster' : ['gbtree','dart','gblinear'],
        'classifier__n_estimators' : [50, 100, 150, 200],
        'classifier__max_depth' : list(range(1,10,2)),
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
    time_xgb = (time.time() - start_time_xgb)/60
    '''
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
    '''
    # Append XGB to scores list
    acc_score.append(accuracy_score(y_test , xgb_clf_predict))
    prec_score.append(precision_score(y_test, xgb_clf_predict))
    rcall_score.append(recall_score(y_test, xgb_clf_predict))
    fone_score.append(f1_score(y_test, xgb_clf_predict))
    '''
    # Append time for processing to list
    time_model.append(time_lr)
    time_model.append(time_rf)
    time_model.append(time_gbc)
    #time_model.append(time_xgb)
    
    report_card = {'Accuracy': acc_score, 'Precision': prec_score, 'Recall':  rcall_score, 'F1': fone_score, 'Time_Model': time_model}
    report_card_df = pd.DataFrame(report_card, index=['LR','RF', 'GBC','Time_Model'])
    print(report_card_df)
    return report_card_df
        



# Run 1: BackBlaze 5 features raw measurements
# train data
x_scaled_train_s.shape, y_train_s.shape 
# test data
x_test.shape, y_test.shape
Counter(y_test.failure)

classify_hdd_failure(x_scaled_train_s, x_test, y_train_s.values.ravel(), y_test.values.ravel())

# Run 2: With all raw column as features
# Train data
x_all_raw_scaled_train_sm.shape, y_all_raw_train_sm.shape
# Test Data
x_ar_scaled_test.shape, y_ar_test.shape

classify_hdd_failure(x_all_raw_scaled_train_sm, x_ar_scaled_test, y_all_raw_train_sm.values.ravel(), y_ar_test.values.ravel())


# Run 3
# train data:
x_pca99_train.shape , y_all_raw_train_sm.shape
# test data:
x_pca99_test.shape, y_ar_test.shape

classify_hdd_failure(x_pca99_train, x_pca99_test, y_all_raw_train_sm.values.ravel(), y_ar_test.values.ravel())




# Run 4
# train data
x_pca95_train.shape , y_all_raw_train_sm.shape
# test data
x_pca95_test.shape, y_ar_test.shape


classify_hdd_failure(x_pca95_train, x_pca95_test, y_all_raw_train_sm.ravel().values.ravel(), y_ar_test.ravel().values.ravel())

sorted(sklearn.metrics.SCORERS.keys())

#########################################################################
# Logistic Regression
start_time_lr = time.time()
pipe_lr = Pipeline([('classifier', LogisticRegression())])

param_grid_lr = [
    {'classifier' : [LogisticRegression()],
     'classifier__penalty' : ['l1','l2'],
     'classifier__solver' : ['liblinear','saga','saga'],
     'classifier__max_iter': [1000]}
]
param_grid_lr

from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_clf = GridSearchCV(pipe_lr, param_grid=param_grid_lr, cv=cv, verbose=True, scoring='f1_score', n_jobs=-1)
x5_scaled_train_s, x5_test, y5_train_s, y5_test
lr_clf.fit(x5_scaled_train_s, y5_train_s)

#define best parameter variables to run on LR algo.
lr_max_iter = lr_clf.best_params_['classifier__max_iter']
lr_penalty = lr_clf.best_params_['classifier__penalty']
lr_solver = lr_clf.best_params_['classifier__solver']

# Now that we have the best params, we will fit the model to the training data and run
lr_clf_best =  LogisticRegression(max_iter = lr_max_iter , 
                                  penalty = lr_penalty,
                                  solver = lr_solver)
lr_clf_best.fit(x5_scaled_train_s, y5_train_s)
lr_clf_predict = lr_clf_best.predict(x5_test)
lr_clf_predict

# Run all related reports
print('LR Accuracy Score: ', accuracy_score(y5_test.values.ravel(), lr_clf_predict))
print('LR Precision Score: ', precision_score(y5_test.values.ravel(), lr_clf_predict))
print('LR Recall Score: ', recall_score(y5_test.values.ravel(), lr_clf_predict))
print('LR F1 Score: ', f1_score(y5_test.values.ravel(), lr_clf_predict))
print('LR Confusion Matrix: \n', confusion_matrix(y5_test.values.ravel(), lr_clf_predict))
print('LR Classification Report: \n', classification_report(y5_test.values.ravel(), lr_clf_predict))
print("------Time taken for LR: %s minutes" % ((time.time() - start_time_lr)/60))


#########################################################################
# Random Forest Classifier

# define pipeline
pipe_rf = Pipeline([('classifier', RandomForestClassifier())])

# define the parameters to be used for Grid Search
param_grid_rf = [
    {'classifier' : [RandomForestClassifier()],
    'classifier__n_estimators' : [10, 11,12,13,14,15, 16,17,18,19,20],
    'classifier__max_depth' : [26,27,18,29,30,31,32,33,34,35]}]

# define the cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_clf = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=cv, verbose=True, scoring='accuracy', n_jobs=-1)

# Fit the model to the Training Dataset
rf_clf.fit(x_all_raw_scaled_train_sm, y_all_raw_train_sm.values.ravel())


# Train Data
x_all_raw_scaled_train_sm.shape, y_all_raw_train_sm.shape
# Test Data
x_ar_scaled_test.shape, y_ar_test.shape


#define best parameter variables to run on RF algo.
rf_n_estimators = rf_clf.best_params_['classifier__n_estimators']
rf_max_depth = rf_clf.best_params_['classifier__max_depth']

rf_n_estimators, rf_max_depth

# Now that we have the best params, we will fit the model to the training data and run
rf_clf_best = RandomForestClassifier(n_estimators=rf_n_estimators,
                                     max_depth=rf_max_depth)
rf_clf_best.fit(x_all_raw_scaled_train_sm, y_all_raw_train_sm.values.ravel())

# Run Test data to predict
rf_clf_predict = rf_clf_best.predict(x_ar_scaled_test)

# Run all related reports.
print('RF Accuracy Score: ', accuracy_score(y_ar_test.values.ravel() , rf_clf_predict))
print('RF Precision Score: ', precision_score(y_ar_test.values.ravel(), rf_clf_predict))
print('RF Recall Score: ', recall_score(y_ar_test.values.ravel(), rf_clf_predict))
print('RF F1 Score: ', f1_score(y_ar_test.values.ravel(), rf_clf_predict))
print('RF Confusion Matrix: \n', confusion_matrix(y_ar_test.values.ravel() , rf_clf_predict))
print('RF Classification Report: \n ', classification_report(y_ar_test.values.ravel(), rf_clf_predict))



#########################################################################
# Gradient Boosting Classifier:
    
x_all_raw_scaled_train_sm, x_ar_scaled_test, y_all_raw_train_sm.values.ravel(), y_ar_test.values.ravel()
    
    
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
gbc_clf.fit(x_ar_scaled_pca99_train_sm, y_all_raw_train_sm.values.ravel())

x_ar_scaled_pca99_train_sm, x_ar_scaled_pca99_test, y_all_raw_train_sm.values.ravel(), y_all_raw_test.values.ravel()

#define best parameter variables to run on GBC algo.
gbc_learning_rate = gbc_clf.best_params_['classifier__learning_rate']
gbc_n_estimators = gbc_clf.best_params_['classifier__n_estimators']
gbc_max_depth = gbc_clf.best_params_['classifier__max_depth']

gbc_learning_rate, gbc_n_estimators, gbc_max_depth

# Now that we have the best params, we will fit the model to the training data and run
gbc_clf_best = GradientBoostingClassifier(n_estimators=gbc_n_estimators,
                                          max_depth=gbc_max_depth,
                                          learning_rate=gbc_learning_rate)

# Train the model on optm parameters
gbc_clf_best.fit(x_ar_scaled_pca99_train_sm, y_all_raw_train_sm.values.ravel())

# predict on test set
gbc_clf_predict = gbc_clf.predict(x_ar_scaled_pca99_test)

# Run all related reports
print('GBC Accuracy Score: ', accuracy_score(y_all_raw_test.values.ravel() , gbc_clf_predict))
print('GBC Precision Score: ', precision_score(y_all_raw_test.values.ravel(), gbc_clf_predict))
print('GBC Recall Score: ', recall_score(y_all_raw_test.values.ravel(), gbc_clf_predict))
print('GBC F1 Score: ', f1_score(y_all_raw_test.values.ravel(), gbc_clf_predict))
print('GBC Confusion Matrix: \n', confusion_matrix(y_all_raw_test.values.ravel() , gbc_clf_predict))
print('GBC Classification Report: \n ', classification_report(y_all_raw_test.values.ravel(), gbc_clf_predict))
    

##################################################################
# XGB [eXtreme Gradient Boosting]
# sources: https://cran.r-project.org/web/packages/xgboost/vignettes/xgboost.pdf
# https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/

from xgboost import XGBClassifier

xgb = XGBClassifier()

param_grid_xgb = [
    {'classifier' : [XGBClassifier()],
    'classifier__booster' : ['gbtree'],
    'classifier__n_estimators' : [20,30,40],
    'classifier__max_depth' : list(range(0,1,1)),
    'classifier__learning_rate': [0.001, 0.005, 0.01],
    }
    ]

# define the cross validation and grid search
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_clf = GridSearchCV(xgb, param_grid=param_grid_xgb, cv=cv, verbose=True, scoring='accuracy', n_jobs=-1)

# Data set 1 for train test 
x_scaled_train_s, x_test, y_train_s, y_test
x_scaled_train_s
x_test

#Dataset 2
# Train Data
x_all_raw_scaled_train_sm.shape, y_all_raw_train_sm.shape
# Test Data
x_ar_scaled_test.shape, y_ar_test.shape


# Fit the model on training dataset for optm parameters
xgb_clf.fit(x_ar_scaled_pca99_train_sm, y_all_raw_train_sm.values.ravel())

x_ar_scaled_pca99_train_sm, x_ar_scaled_pca99_test, y_all_raw_train_sm.values.ravel(), y_all_raw_test.values.ravel()

#define best parameter variables to run on GBC algo.
xgb_learning_rate = xgb_clf.best_params_['classifier__learning_rate']
xgb_booster = xgb_clf.best_params_['classifier__booster']
xgb_n_estimators = xgb_clf.best_params_['classifier__n_estimators']
xgb_max_depth = xgb_clf.best_params_['classifier__max_depth']

xgb_learning_rate, xgb_booster, xgb_n_estimators, xgb_max_depth


# Now that we have the best params, we will fit the model to the training data and run
xgb_clf_best = XGBClassifier(booster=xgb_booster,
                             n_estimators=xgb_n_estimators,
                             max_depth=xgb_max_depth,
                             learning_rate=xgb_learning_rate)

# Train the model on optm parameters
xgb_clf_best.fit(x_ar_scaled_pca99_train_sm, y_all_raw_train_sm.values.ravel())

# predict on test set
xgb_clf_predict = xgb_clf.predict(x_ar_scaled_pca99_test)



# Run all related reports
print('XGB Accuracy Score: ', accuracy_score(y_all_raw_test.values.ravel() , xgb_clf_predict))
print('XGB Precision Score: ', precision_score(y_all_raw_test.values.ravel(), xgb_clf_predict))
print('XGB Recall Score: ', recall_score(y_all_raw_test.values.ravel(), xgb_clf_predict))
print('XGB F1 Score: ', f1_score(y_all_raw_test.values.ravel(), xgb_clf_predict))
print('XGB Confusion Matrix: \n', confusion_matrix(y_all_raw_test.values.ravel() , xgb_clf_predict))
print('XGB Classification Report: \n', classification_report(y_all_raw_test.values.ravel(), xgb_clf_predict))

##################################################################


from sklearn.svm import SVC

svc = SVC() 

'''
SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=True)
'''
#to print all params in the estimator
for param in svc.get_params().keys():
    print(param)
    
param_grid_svc = {'C': [0.1, 1, 10],  
              'gamma': [1, 0.1, 0.01], 
              'kernel': ['linear', 'rbf']} 

# define the cross validation and grid search
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(svc, param_grid_svc, cv=cv, refit=True, verbose=3, n_jobs=-1)

# Fit the model on training dataset for optm parameters
grid.fit(x_scaled_train_s, y_train_s.values.ravel())
x_scaled_train_s, x_test, y_train_s, y_test

#classify_hdd_failure(x_ar_scaled_pca99_train_sm, x_ar_scaled_pca99_test, y_all_raw_train_sm.values.ravel(), y_all_raw_test.values.ravel())

#define best parameter variables to run on GBC algo.
svc_C = grid.best_params_['C']
svc_gamma = grid.best_params_['gamma']
svc_kernel = grid.best_params_['kernel']


svc_C, svc_gamma, svc_kernel


# Now that we have the best params, we will fit the model to the training data and run
svc_best = SVC(C = svc_C, gamma = svc_gamma, kernel = svc_kernel)

# Train the model on optm parameters
svc_best.fit(x_ar_scaled_pca99_train_sm, y_all_raw_train_sm.values.ravel())

# predict on test set
svc_clf_predict = svc.predict(x_ar_scaled_pca99_test)

# Run all related reports
print('SVC Accuracy Score: ', accuracy_score(y_all_raw_test.values.ravel() , svc_clf_predict))
print('SVC Precision Score: ', precision_score(y_all_raw_test.values.ravel(), svc_clf_predict))
print('SVC Recall Score: ', recall_score(y_all_raw_test.values.ravel(), svc_clf_predict))
print('SVC F1 Score: ', f1_score(y_all_raw_test.values.ravel(), svc_clf_predict))
print('SVC Confusion Matrix: \n', confusion_matrix(y_all_raw_test.values.ravel() , svc_clf_predict))
print('SVC Classification Report: \n', classification_report(y_all_raw_test.values.ravel(), svc_clf_predict))

########################################################

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
pd.DataFrame(report_card, index=['LR','RF', 'GBC','XGB'])


################################################################################
################################################################################





def classify_hdd_failure(x_train, x_test, y_train, y_test):
    ###### Log Regression #####
    pipe_lr = Pipeline([('classifier', LogisticRegression())])
    #return lr

    param_grid_lr = [{'classifier' : [LogisticRegression()],'classifier__penalty' : ['l1','l2'],'classifier__solver' : ['sag','saga'], 'classifier__max_iter': [1000]}]
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
    
    ################################################################################
    # Random Forest Classifier
    
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
    
    
    
    #########################################################################
    # Gradient Boosting Classifier:
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
        
    
    ##################################################################
    # XGB [eXtreme Gradient Boosting]
    # sources: https://cran.r-project.org/web/packages/xgboost/vignettes/xgboost.pdf
    # https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/
    
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
    return report_card
        

classify_hdd_failure(x_frac_train, x_test, y_frac_train, y_test)
