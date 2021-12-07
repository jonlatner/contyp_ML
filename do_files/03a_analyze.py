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

# file paths
main_dir = "/Users/jonathanlatner/Documents/GitHub/contyp_ML/"
data_dir = "data_files/"
graphs_dir = "graphs/"
results_dir = "results/"

# top codes 
import os
import pandas as pd
import numpy as np
from numpy import argmax

# import statsmodels.api as sm
from statsmodels.formula.api import logit
from sklearn.linear_model import LogisticRegression

# from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, accuracy_score, roc_curve, classification_report

# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC

 # options
pd.set_option("display.max_columns", 40)
pd.set_option('display.float_format', lambda x: '%.5f' % x) # remove scientific notation

# load data
# df0 = pd.read_csv(os.path.join(main_dir,data_dir,"df_eu_lfs_sample_10_clean.csv"))
df0 = pd.read_pickle(os.path.join(main_dir,data_dir,"df_eu_lfs_sample_10_clean.pkl"))

# examine data
# print(df0)
# df0.head()
# df0.tail()
# df0.columns
# df0.index
# df0.describe()

# regression
df1=df0
df1.sort_values(by=['country',"year"], inplace=True)
df1=df1[(df1['country'] != "MT")] # drop Malta
df1=df1[(df1['country'] != "EE") | (df1['year'] != 2003)] # drop Malta
# df1['country'].value_counts()

pd.crosstab(df1["year"], df1["country"])

data = []
country = list(sorted(set(df1["country"])))

for c in sorted(country):
    df_country = df1[(df1['country'] == c)]
    print(c)
    year = list(set(df_country["year"]))
    for y in sorted(year):
        print(y)
        df_country_year = df_country[(df_country['year'] == y)]

        # LOGISTIC REGRESSION
        glm = logit('temp ~ age_cat_Y + age_cat_O + edu_cat_L + edu_cat_H + female', data=df_country_year).fit()
        yhat=glm.predict()
        # glm.summary()
        # print(glm.params)

        array = df_country_year.values
        X = array[:,4:]
        Y = array[:,3]
        Y=Y.astype('int')
        y_true = df_country_year['temp']
        
        # Calculate Youden’s J statistic to get the best threshold
        fpr, tpr, thresholds = roc_curve(Y, yhat)
        J = tpr - fpr
        ix = argmax(J)
        best_thresh = thresholds[ix] # best_thresh
        # best_thresh 
        
        # Logistic Regression - Youden’s J      
        clf = LogisticRegression().fit(X, Y)
        yhat=clf.predict(X)
        y_pred = np.where(yhat > best_thresh, 1, 0)
        # print(classification_report(y_true, y_pred))
        recall = recall_score(Y, y_pred, average=None)
        TN = recall[0]
        TP = recall[1]
        accuracy = accuracy_score(y_true, y_pred) 
        type = "LR - Youden's J"
        
        data.append([c, y, type, TN, TP, accuracy])

        # Logistic Regression        
        clf = LogisticRegression().fit(X, Y)
        yhat=clf.predict(X)
        y_pred = np.where(yhat > .5, 1, 0)
        # print(classification_report(Y, y_pred))
        recall = recall_score(y_true, y_pred, average=None)
        TN = recall[0]
        TP = recall[1]
        accuracy = accuracy_score(y_true, y_pred) 
        type = "LR"
        
        data.append([c, y, type, TN, TP, accuracy])
        
        # Decision Tree
        clf = DecisionTreeClassifier().fit(X, Y)
        yhat=clf.predict(X)
        y_pred = np.where(yhat > .5, 1, 0)
        # print(classification_report(Y, y_pred))
        recall = recall_score(y_true, y_pred, average=None)
        TN = recall[0]
        TP = recall[1]
        accuracy = accuracy_score(y_true, y_pred) 
        type = "DT"
        
        data.append([c, y, type, TN, TP, accuracy])
        
        # Linear Discrimination Analysis
        clf = LinearDiscriminantAnalysis().fit(X, Y)
        yhat=clf.predict(X)
        y_pred = np.where(yhat > .5, 1, 0)
        # print(classification_report(Y, y_pred))
        recall = recall_score(y_true, y_pred, average=None)
        TN = recall[0]
        TP = recall[1]
        accuracy = accuracy_score(y_true, y_pred) 
        type = "LDA"

        data.append([c, y, type, TN, TP, accuracy])
        
        # Naive Bayes
        clf = GaussianNB().fit(X, Y)
        yhat=clf.predict(X)
        y_pred = np.where(yhat > .5, 1, 0)
        # print(classification_report(Y, y_pred))
        TN = recall[0]
        TP = recall[1]
        accuracy = accuracy_score(y_true, y_pred) 
        type = "NB"

        data.append([c, y, type, TN, TP, accuracy])
        
        # LOGISTIC REGRESSION
        glm = logit('temp ~ age_cat_Y + age_cat_O + edu_cat_L + edu_cat_H + female', data=df_country_year).fit()
        yhat=glm.predict()
        # glm.summary()
        # print(glm.params)

        array = df_country_year.values
        X = array[:,4:]
        Y = array[:,3]
        Y=Y.astype('int')
        y_true = df_country_year['temp']
        
        # Calculate Youden’s J statistic to get the best threshold
        fpr, tpr, thresholds = roc_curve(Y, yhat)
        J = tpr - fpr
        ix = argmax(J)
        best_thresh = thresholds[ix] # best_thresh
        # best_thresh 
        
        # Logistic Regression - Youden’s J      
        clf = LogisticRegression().fit(X, Y)
        yhat=clf.predict(X)
        y_pred = np.where(yhat > best_thresh, 1, 0)
        # print(classification_report(y_true, y_pred))
        recall = recall_score(Y, y_pred, average=None)
        TN = recall[0]
        TP = recall[1]
        accuracy = accuracy_score(y_true, y_pred) 
        type = "LR - Youden's J"
        
        data.append([c, y, type, TN, TP, accuracy])

        # Logistic Regression        
        clf = LogisticRegression().fit(X, Y)
        yhat=clf.predict(X)
        y_pred = np.where(yhat > .5, 1, 0)
        # print(classification_report(Y, y_pred))
        recall = recall_score(y_true, y_pred, average=None)
        TN = recall[0]
        TP = recall[1]
        accuracy = accuracy_score(y_true, y_pred) 
        type = "LR"
        
        data.append([c, y, type, TN, TP, accuracy])
        
        # Decision Tree
        clf = DecisionTreeClassifier().fit(X, Y)
        yhat=clf.predict(X)
        y_pred = np.where(yhat > .5, 1, 0)
        # print(classification_report(Y, y_pred))
        recall = recall_score(y_true, y_pred, average=None)
        TN = recall[0]
        TP = recall[1]
        accuracy = accuracy_score(y_true, y_pred) 
        type = "DT"
        
        data.append([c, y, type, TN, TP, accuracy])
        
        # Linear Discrimination Analysis
        clf = LinearDiscriminantAnalysis().fit(X, Y)
        yhat=clf.predict(X)
        y_pred = np.where(yhat > .5, 1, 0)
        # print(classification_report(Y, y_pred))
        recall = recall_score(y_true, y_pred, average=None)
        TN = recall[0]
        TP = recall[1]
        accuracy = accuracy_score(y_true, y_pred) 
        type = "LDA"

        data.append([c, y, type, TN, TP, accuracy])
        
        # Naive Bayes
        clf = GaussianNB().fit(X, Y)
        yhat=clf.predict(X)
        y_pred = np.where(yhat > .5, 1, 0)
        # print(classification_report(Y, y_pred))
        TN = recall[0]
        TP = recall[1]
        accuracy = accuracy_score(y_true, y_pred) 
        type = "NB"

        data.append([c, y, type, TN, TP, accuracy])
        
df_fit = pd.DataFrame(data)

df_fit = pd.DataFrame(data,columns=('country', 'year', "type", "TN", "TP", "accuracy"))
        
df_fit.sort_values(by=['country',"year","type"], inplace=True)
print(df_fit)

df_fit.to_csv(os.path.join(main_dir,results_dir,"df_fit_10.csv"))


