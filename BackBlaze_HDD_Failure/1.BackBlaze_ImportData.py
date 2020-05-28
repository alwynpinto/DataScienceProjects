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

