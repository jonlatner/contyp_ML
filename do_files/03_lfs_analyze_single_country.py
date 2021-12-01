# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
https://jbhender.github.io/Stats506/F17/Projects/G15.html

"""

def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
if __name__ == "__main__":
    clear_all()

# top codes 
import os
import pandas as pd
import numpy as np
import pyarrow.feather as feather
from numpy import argmax

# file paths
main_path = "/Users/jonathanlatner/Google Drive/SECCOPA/projects/"
data_path = "/Users/jonathanlatner/Desktop/python_stuff/data_files/"
dst_path = "distribution_contyp/data_files/eu_lfs/"
graphs_path = "/Users/jonathanlatner/Desktop/python_stuff/graphs/"
tables_path = "/Users/jonathanlatner/Desktop/python_stuff/tables/"

from statsmodels.formula.api import logit
from sklearn.linear_model import LogisticRegression

# from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import recall_score, accuracy_score, roc_curve

# import matplotlib.pyplot as plt
# import seaborn as sns

 # options
pd.set_option("display.max_columns", 40)
pd.set_option('display.float_format', lambda x: '%.5f' % x) # remove scientific notation

# load data
df0 = feather.read_feather(os.path.join(data_path,"df_eu_lfs_sample_10_clean.feather"))


# examine data
print(df0)
df0.head()
df0.tail()
df0.describe()
df0.columns
df0.index

#Count total NaN at each column in DataFrame
df0.isnull().sum()

# regression
df_reg=df0[(df0['country_name'] == "Germany")]

# LOGISTIC REGRESSION
glm = logit('temp ~ age_cat_Y + age_cat_O + edu_cat_L + edu_cat_H + female*married', data=df_reg).fit()
glm.summary()
# print(glm.params)
yhat=glm.predict()

array = df_reg.values
X = array[:,4:]
Y = array[:,3]
Y=Y.astype('int')

# Calculate Youden’s J statistic to get the best threshold
fpr, tpr, thresholds = roc_curve(Y, yhat)
J = tpr - fpr
ix = argmax(J)
best_thresh = thresholds[ix] # best_thresh
best_thresh 


# Logistic Regression        
clf = LogisticRegression().fit(X, Y)
# clf.intercept_
# clf.coef_
yhat=clf.predict(X)
y_pred = np.where(yhat > .5, 1, 0)
# print(classification_report(Y, y_pred))
recall = recall_score(Y, y_pred, average=None)
recall_LR_TN = recall[0]
recall_LR_TP = recall[1]
accuracy_LR = accuracy_score(Y, y_pred) 
        
# Decision Tree
clf = DecisionTreeClassifier().fit(X, Y)
yhat=clf.predict(X)
y_pred = np.where(yhat > .5, 1, 0)
# print(classification_report(Y, y_pred))
recall = recall_score(Y, y_pred, average=None)
recall_DT_TN = recall[0]
recall_DT_TP = recall[1]
accuracy_DT = accuracy_score(Y, y_pred) 
        
# Naive Bayes
clf = GaussianNB().fit(X, Y)
yhat=clf.predict(X)
y_pred = np.where(yhat > .5, 1, 0)
# print(classification_report(Y, y_pred))
recall = recall_score(Y, y_pred, average=None)
recall_NB_TN = recall[0]
recall_NB_TP = recall[1]
accuracy_NB = accuracy_score(Y, y_pred) 
        
data = []
data.append([recall_LR_TN, recall_LR_TP, accuracy_LR, 
             recall_DT_TN, recall_DT_TP, accuracy_DT, 
             recall_NB_TN, recall_NB_TP, accuracy_NB,
             ])

df_fit = pd.DataFrame(data)
        
df_fit = pd.DataFrame(data,columns=("recall_LR_TN", "recall_LR_TP", "accuracy_LR", 
                                    "recall_DT_TN", "recall_DT_TP", "accuracy_DT", 
                                    "recall_NB_TN", "recall_NB_TP", "accuracy_NB", 
                                    ))
        
df_fit.to_csv(os.path.join(tables_path,"df_fit.csv"))
