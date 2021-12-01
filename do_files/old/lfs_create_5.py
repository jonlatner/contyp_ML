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
from patsy import dmatrices
import pyreadr
import os
import pandas as pd
import numpy as np
from numpy import argmax

import statsmodels.api as sm
from statsmodels.formula.api import logit
from sklearn.linear_model import LogisticRegression

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score, classification_report, roc_curve

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

 
# options
pd.set_option("display.max_columns", 40)
pd.set_option('display.float_format', lambda x: '%.5f' % x) # remove scientific notation

# file paths
main_path = "/Users/jonathanlatner/Google Drive/SECCOPA/projects/"
dst_path = "distribution_contyp/data_files/eu_lfs/"
graphs_path = "/Users/jonathanlatner/Desktop/python_stuff/graphs/"
tables_path = "/Users/jonathanlatner/Desktop/python_stuff/tables/"

# load data
result = pyreadr.read_r(os.path.join(main_path,dst_path,"df_eu_lfs_sample_10.rds"))
print(result.keys()) # let's check what objects we got
df0 = result[None] # extract the pandas data frame for object None

# examine data
print(df0)
df0.head()
df0.tail()
df0.describe()
df0.columns
df0.index

# random sample
# df1=df0.groupby(["country","year"]).sample(frac=.1)
df1=df0

# clean data
df2=df1
df2.columns= df0.columns.str.lower()
df2=df2.rename(columns={"hatlev1d": "edu_cat", "coeff": "weight"})
df2=df2[(df2['temp'] == 1) | (df2['temp'] == 2)] # drop missing temp
df2=df2[(df2['year'] >= 1996)]
df2=df2[(df2['weight'] > 0)]
df2=df2[(df2['ilostat'] == 1)] # keep if employed

df2=df2[(df2['age'] >= 25) | (df2['age'] <= 54)]
df2["age_sq"]=df2["age"]**2
def age_groups(series):
    if series < 35:
        return "Y"
    elif 35 <= series < 45:
        return "M"
    elif 45 <= series:
        return "O"

df2['age_cat'] = df2['age'].apply(age_groups)

df_age_dummy = pd.get_dummies(df2['age_cat'], prefix='age_cat')
df_age_dummy
df2 = pd.concat([df2, df_age_dummy ], axis = 1)
del df_age_dummy

# education
df2=df2[(df2['edu_cat'] != "9") & (df2['edu_cat'] != "") & (~df2['edu_cat'].isnull())] # drop missing edu_cat
df2=df2[(df2['edu_cat'] != "")] # drop missing edu_cat
df2=df2[df2['edu_cat'].notnull()] # drop missing edu_cat equivalent
df2=df2[~df2['edu_cat'].isnull()] # drop missing edu_cat equivalent


df2["edu_cat"]=df2["edu_cat"].cat.remove_unused_categories()
df_edu_dummy = pd.get_dummies(df2['edu_cat'], prefix='edu_cat')
df_edu_dummy
df2 = pd.concat([df2, df_edu_dummy ], axis = 1)
del df_edu_dummy
df2["edu_cat"].value_counts(dropna=False)

df2["female"]=df2["sex"]-1 

df2=df2[(~df2['marstat'].isnull())] # drop missing marstat
df2['married']=df2['marstat'].replace([0,1,2],[0,0,1])
pd.crosstab(df2["married"], df2["marstat"])

df2["femstat"]=df2["female"]*df2["married"]


df2["temp"]=df2["temp"]-1 
df2["temp"]=df2["temp"].astype(int)
df2["year"]=df2["year"].astype(int)


# df2=df2.drop(['sex',"ilostat","marstat","countryb","ftpt"], 1) # drop columns
df2=df2[["country_name",'country',"year",
         "temp",
         "age_cat_Y","age_cat_O",
         "edu_cat_L","edu_cat_H",
         "female", "married", "femstat"]]
df2.head
df2.columns
df2.describe

#Count total NaN at each column in DataFrame
df2.isnull().sum()

# regression
df_reg=df2
df_reg.sort_values(by=['country',"year"], inplace=True)

df_reg = df2[(df2['country_name'] != "Estonia") | (df2['year'] != 2002)]
df_reg = df_reg[(df_reg['country_name'] != "Estonia") | (df_reg['year'] != 1997)]
df_reg = df_reg[(df_reg['country_name'] != "Lithuania") | (df_reg['year'] != 2000)]
df_reg = df_reg[(df_reg['country_name'] != "Luxembourg") | (df_reg['year'] != 1996)]

# df_reg=df2[(df2['country_name'] == "Luxembourg")]
# df_reg = df_reg[(df_reg['country_name'] != "Luxembourg") | (df_reg['year'] != 1996)]
df_reg=df2[(df2['country_name'] == "Germany")]

data = []
country = list(set(df_reg["country_name"]))
for c in country:
    df_country = df_reg[(df_reg['country_name'] == c)]
    print(c)
    year = list(set(df_country["year"]))
    for y in year:
        print(y)

        # LOGISTIC REGRESSION
        # glm = logit('temp ~ age_cat_Y + age_cat_O + edu_cat_L + edu_cat_H + female*married', data=df_reg).fit()
        # glm.summary()
        # print(glm.params)

        #DECISION TREE        
        array = df_reg.values
        X = array[:,4:]
        Y = array[:,3]
        Y=Y.astype('int')

        # Logistic Regression        
        clf = LogisticRegression().fit(X, Y)
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
        
        # Linear Discrimination Analysis
        clf = LinearDiscriminantAnalysis().fit(X, Y)
        yhat=clf.predict(X)
        y_pred = np.where(yhat > .5, 1, 0)
        # print(classification_report(Y, y_pred))
        recall = recall_score(Y, y_pred, average=None)
        recall_LDA_TN = recall[0]
        recall_LDA_TP = recall[1]
        accuracy_LDA = accuracy_score(Y, y_pred) 
        
        # Naive Bayes
        clf = GaussianNB().fit(X, Y)
        yhat=clf.predict(X)
        y_pred = np.where(yhat > .5, 1, 0)
        # print(classification_report(Y, y_pred))
        recall = recall_score(Y, y_pred, average=None)
        recall_NB_TN = recall[0]
        recall_NB_TP = recall[1]
        accuracy_NB = accuracy_score(Y, y_pred) 
        
        data.append([c, y, 
                     recall_LR_TN, recall_LR_TP, accuracy_LR, 
                     recall_DT_TN, recall_DT_TP, accuracy_DT, 
                     recall_LDA_TN, recall_LDA_TP, accuracy_LDA, 
                     recall_NB_TN, recall_NB_TP, accuracy_NB, 
                     ])
        df_fit = pd.DataFrame(data)
        
        df_fit = pd.DataFrame(data,columns=('country', 'year', 
                     "recall_LR_TN", "recall_LR_TP", "accuracy_LR", 
                     "recall_DT_TN", "recall_DT_TP", "accuracy_DT", 
                     "recall_LDA_TN", "recall_LDA_TP", "accuracy_LDA", 
                     "recall_NB_TN", "recall_NB_TP", "accuracy_NB", 
                     ))
        
        df_fit.sort_values(by=['country',"year"], inplace=True)
        df_fit.to_csv(os.path.join(tables_path,"df_fit.csv"))

df_fit_long = pd.melt(df_fit, id_vars=["country","year"])
df_graph = df_fit_long[df_fit_long['variable'].str.contains("accuracy")]
g = sns.FacetGrid(df_graph, 
                  col="country", 
                  # col_wrap=6, 
                  hue = "variable", legend_out=True)
g.map_dataframe(sns.lineplot, x="year", y="value")
g.set_titles("{col_name}")  # use this argument literally
g.add_legend(loc="lower center")
