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
from numpy import sqrt
from numpy import argmax
import statsmodels.api as sm
from statsmodels.formula.api import logit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score, classification_report, roc_curve
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import seaborn as sns
 
# options
pd.set_option("display.max_columns", 40)
pd.set_option('display.float_format', lambda x: '%.5f' % x) # remove scientific notation

# file paths
main_path = "/Users/jonathanlatner/Google Drive/SECCOPA/projects/"
dst_path = "distribution_contyp/data_files/eu_lfs/"

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
df_reg=df2[(df2['country_name'] == "Germany")]

Y = pd.DataFrame(df_reg['temp'])
Y = np.ravel(Y)
X = df_reg[["age_cat_Y", "age_cat_M", "age_cat_O", "edu_cat_L", "edu_cat_M", "edu_cat_H", "female", "married"]]

Y_df = pd.DataFrame(Y)
Y_df.describe()

# LOGISTIC REGRESSION
# glm = logit('temp ~ age_cat_Y + age_cat_O + edu_cat_L + edu_cat_H + female*married', 
#             data=df_reg).fit()
# glm.summary()

model = LogisticRegression()
model.fit(X, Y)    

# =============================================================================
# # predict probabilities
# yhat = model.predict_proba(X)
# # keep probabilities for the positive outcome only
# yhat = yhat[:, 1]
# # calculate roc curves
# fpr, tpr, thresholds = roc_curve(Y, yhat)
# # plot the roc curve for the model
# pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
# pyplot.plot(fpr, tpr, marker='.', label='Logistic')
# # axis labels
# pyplot.xlabel('False Positive Rate')
# pyplot.ylabel('True Positive Rate')
# pyplot.legend()
# # show the plot
# pyplot.show()
# =============================================================================

# predict probabilities
yhat = model.predict_proba(X)
# keep probabilities for the positive outcome only
yhat = yhat[:, 1]
# calculate roc curves
fpr, tpr, thresholds = roc_curve(Y, yhat)

# Calculate Youden’s J statistic to get the best threshold
J = tpr - fpr
ix = argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold=%f' % (best_thresh))
best_thresh

# =============================================================================
# # plot the roc curve for the model
# pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
# pyplot.plot(fpr, tpr, marker='.', label='Logistic')
# pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
# # axis labels
# pyplot.xlabel('False Positive Rate')
# pyplot.ylabel('True Positive Rate')
# pyplot.legend()
# # show the plot
# pyplot.show()
# =============================================================================

y_pred_50 = np.where(yhat > .5, 1, 0)
y_pred_best = np.where(yhat > best_thresh, 1, 0)

print(classification_report(Y, y_pred_50))
print(classification_report(Y, y_pred_best))
