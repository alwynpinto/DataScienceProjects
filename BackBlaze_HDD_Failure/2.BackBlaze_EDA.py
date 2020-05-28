
BackBlaze HDD Failure Dataset
Alwyn Pinto
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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score, fbeta_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
random.seed(42)



df = pd.read_csv(r'Z:\PersonalFolders\_\Springboard\Projects\BackBlazeHDDfailure\frame_8raw_clean.csv', parse_dates=True, index_col='date')

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
# Only smart_9_raw has been imputed

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
