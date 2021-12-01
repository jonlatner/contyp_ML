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
import pyreadr
import os
import pandas as pd
import numpy as np
from numpy import argmax

import statsmodels.api as sm
from statsmodels.formula.api import logit

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score, classification_report, roc_curve

import matplotlib.pyplot as plt
import seaborn as sns
 
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
# df2=df2[(df2['edu_cat'] != "")] # drop missing edu_cat
# df2=df2[df2['edu_cat'].notnull()] # drop missing edu_cat equivalent
# df2=df2[~df2['edu_cat'].isnull()] # drop missing edu_cat equivalent


df2["edu_cat"]=df2["edu_cat"].cat.remove_unused_categories()
df_edu_dummy = pd.get_dummies(df2['edu_cat'], prefix='edu_cat')
df_edu_dummy
df2 = pd.concat([df2, df_edu_dummy ], axis = 1)
del df_edu_dummy

df2["female"]=df2["sex"]-1 

df2=df2[(~df2['marstat'].isnull())] # drop missing marstat
df2['married']=df2['marstat'].replace([0,1,2],[0,0,1])
pd.crosstab(df2["married"], df2["marstat"])

df2["femstat"]=df2["female"]*df2["married"]


df2["temp"]=df2["temp"]-1 
df2["temp"]=df2["temp"].astype(int)
df2["year"]=df2["year"].astype(int)


df2=df2.drop(['sex',"ilostat","marstat","countryb","ftpt"], 1) # drop columns
df2.head
df2.columns
df2.describe

#Count total NaN at each column in DataFrame
df2.isnull().sum()
df2["edu_cat"].value_counts(dropna=False)
df2.isnull().sum()

# frequency counts
# =============================================================================
# df2["edu_cat"].value_counts()
# df2["married"].describe()
# df2["sex"].value_counts()
# pd.crosstab(df2["edu_cat"], df2["sex"])
# pd.crosstab(df2["edu_cat"], df2["sex"], normalize=True)
# pd.crosstab(df2["edu_cat"], df2["sex"], normalize="columns")
# pd.crosstab(df2["edu_cat"], df2["sex"], normalize="index")
# =============================================================================


# regression
df_reg=df2
df_reg.sort_values(by=['country',"year"], inplace=True)

df_reg = df2[(df2['country_name'] != "Estonia") | (df2['year'] != 2002)]
df_reg = df_reg[(df_reg['country_name'] != "Estonia") | (df_reg['year'] != 1997)]
df_reg = df_reg[(df_reg['country_name'] != "Lithuania") | (df_reg['year'] != 2000)]
df_reg = df_reg[(df_reg['country_name'] != "Luxembourg") | (df_reg['year'] != 1996)]

# df_reg=df2[(df2['country_name'] == "Luxembourg")]
# df_reg = df_reg[(df_reg['country_name'] != "Luxembourg") | (df_reg['year'] != 1996)]
# df_reg=df2[(df2['country_name'] == "Germany")]

data = []
country = list(set(df_reg["country_name"]))
for c in country:
    df_country = df_reg[(df_reg['country_name'] == c)]
    print(c)
    year = list(set(df_country["year"]))
    for y in year:
        print(y)

        df_year = df_country[(df_reg['year'] == y)]
        y_true = df_year['temp']
        
        # LOGISTIC REGRESSION
        glm = logit('temp ~ age_cat_Y + age_cat_O + edu_cat_L + edu_cat_H + female*married', 
                    data=df_year).fit()
        glm.summary()
        yhat=glm.predict()
        pd.DataFrame(yhat).describe()

        # Calculate Youden’s J statistic to get the best threshold
        fpr, tpr, thresholds = roc_curve(y_true, yhat)
        J = tpr - fpr
        ix = argmax(J)
        best_thresh = thresholds[ix]
        # best_thresh
                
        # Describe model fit
        y_pred = np.where(yhat > .5, 1, 0)
        # print(classification_report(y_true, y_pred))
        recall = recall_score(y_true, y_pred, average=None)
        recall_0 = recall[0]
        recall_1 = recall[1]
        accuracy = accuracy_score(y_true, y_pred) 

        # Describe model fit with Youden's J statistic
        y_pred = np.where(yhat > best_thresh, 1, 0)
        # print(classification_report(y_true, y_pred))
        recall = recall_score(y_true, y_pred, average=None)
        recall_0_best = recall[0]
        recall_1_best = recall[1]
        accuracy_best = accuracy_score(y_true, y_pred) 

        #DECISION TREE
        X = df_year[["age_cat_Y", "age_cat_M", "age_cat_O", "edu_cat_L", "edu_cat_M", "edu_cat_H", "female", "married", "femstat"]]
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X, y_true)
        yhat_DT = clf.predict(X)
        yhat_DT = clf.predict_proba(X)
        pd.DataFrame(yhat_DT).describe()

        # Describe model fit
        y_pred = np.where(yhat_DT > .5, 1, 0)
        # y_pred = (clf.predict_proba(X)[:,1] >= best_thresh).astype(bool) # set threshold as 0.3
        
        print(classification_report(y_true, y_pred))
        recall = recall_score(y_true, y_pred, average=None)
        recall_0_DT = recall[0]
        recall_1_DT = recall[1]
        accuracy_DT = accuracy_score(y_true, y_pred) 
        
        data.append([c, y, 
                     recall_0, recall_0_best, recall_0_DT, 
                     recall_1, recall_1_best, recall_1_DT, 
                     accuracy, accuracy_best, accuracy_DT 
                     ])
        df_fit = pd.DataFrame(data)
        df_fit = pd.DataFrame(data,columns=('country', 'year', 
                                            "recall_0", "recall_0_best", "recall_0_DT", 
                                            "recall_1", "recall_1_best", "recall_1_DT", 
                                            "accuracy_1", "accuracy_1_best", "accuracy_1_DT"
                                            ))

        df_fit.sort_values(by=['country',"year"], inplace=True)
        df_fit.to_csv(os.path.join(tables_path,"df_fit.csv"))
        
        # https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/
    
# GRAPH
df_fit_long = pd.melt(df_fit, id_vars=["country","year"])
df_graph = df_fit_long[df_fit_long['variable'].str.contains("recall_0")]
df_graph = df_fit_long[df_fit_long['variable'].str.contains("recall_1")]

df_graph = df_fit_long[df_fit_long['variable'].str.contains("accuracy")]
g = sns.FacetGrid(df_graph, 
                  col="country", 
                  col_wrap=6, 
                  hue = "variable", legend_out=True)
g.map_dataframe(sns.lineplot, x="year", y="value")
g.set_titles("{col_name}")  # use this argument literally
g.add_legend(loc="lower center", bbox_to_anchor=[.85, .1], ncol=1)

# g.savefig(os.path.join(graphs_path,"graph_50.pdf"), bbox_inches='tight')
g.savefig(os.path.join(graphs_path,"graph_accuracy.pdf"), bbox_inches='tight')



df_graph = df_fit_long[df_fit_long['variable'].str.contains("recall_1")]
g = sns.FacetGrid(df_graph, 
                  col="country", 
                  col_wrap=6, 
                  hue = "variable", legend_out=True)
g.map_dataframe(sns.lineplot, x="year", y="value")
g.set_titles("{col_name}")  # use this argument literally
g.add_legend(loc="lower center", bbox_to_anchor=[.85, .1], ncol=1)

# g.savefig(os.path.join(graphs_path,"graph_50.pdf"), bbox_inches='tight')
g.savefig(os.path.join(graphs_path,"graph_recall_TP.pdf"), bbox_inches='tight')



df_graph = df_fit_long[df_fit_long['variable'].str.contains("recall_0")]
g = sns.FacetGrid(df_graph, 
                  col="country", 
                  col_wrap=6, 
                  hue = "variable", legend_out=True)
g.map_dataframe(sns.lineplot, x="year", y="value")
g.set_titles("{col_name}")  # use this argument literally
g.add_legend(loc="lower center", bbox_to_anchor=[.85, .1], ncol=1)

# g.savefig(os.path.join(graphs_path,"graph_50.pdf"), bbox_inches='tight')
g.savefig(os.path.join(graphs_path,"graph_recall_TN.pdf"), bbox_inches='tight')

