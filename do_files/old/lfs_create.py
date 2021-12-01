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
    # insert here your code


# top codes 
import pyreadr
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import logit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score, classification_report
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

# df_reg=df2[(df2['country_name'] == "Germany")]
# df_reg.columns

df_reg = df2[(df2['country_name'] != "Estonia") | (df2['year'] != 2002)]

data = []
country = list(set(df_reg["country_name"]))
for c in country:
    df_country = df_reg[(df_reg['country_name'] == c)]
    print(c)
    year = list(set(df_country["year"]))
    for y in year:
        print(y)
        df_year = df_country[(df_reg['year'] == y)]
        
        y_true = pd.DataFrame(df_year['temp'])
        y_true = np.ravel(y_true)
        
        #DECISION TREE
        X = df_year[["age_cat_Y", "age_cat_M", "age_cat_O", "edu_cat_L", "edu_cat_M", "edu_cat_H", "female", "married"]]
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X, y_true)
        YPred = clf.predict(X) 
        print(classification_report(y_true, YPred))
        auc = roc_auc_score(y_true, YPred)
        auc

        # LOGISTIC REGRESSION
        model = LogisticRegression()
        model.fit(X, y_true)    
        # y_pred_glm = model.predict(X) 
        y_pred_glm = model.predict_proba(X) 
        # keep probabilities for the positive outcome only
        y_pred_glm = y_pred_glm[:, 1]
        
        confusion_matrix(y_true, y_pred_glm).ravel()

        pd.DataFrame(y_pred_glm).describe()
        pd.DataFrame(y_true).describe()


        # predict probabilities
        lr_probs = model.predict_proba(X)
        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]

        # calculate scores
        lr_auc = roc_auc_score(y_true, y_pred_glm)
        lr_auc 

        glm = logit('temp ~ age_cat_Y + age_cat_O + edu_cat_L + edu_cat_H + female + married', 
                    data=df_year).fit()
        glm.summary()

        # True outcome, predicted outcome
        # https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
        
        y_true = df_year['temp']
        yhat = glm.predict()
        pd.DataFrame(yhat).describe()
        y_pred = np.where(yhat > .10, 1, 0)
        print(classification_report(y_true, y_pred))
        auc = roc_auc_score(y_true, y_pred)
        auc

        # Confusion Matrix
        # confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Specificity - True negative rate - The ratio of correct negative predictions to the total negative examples.
        specificity = tn / (tn+fp) 
        
        # Recall - True positive rate - The ratio of correct positive predictions to the total positives examples.
        # recall = recall_score(y_true, y_pred, average=None)
        # recall[0]
        recall = tp / (tp+fn) 
        
        # Precision — Also called Positive predictive value
        # The ratio of correct positive predictions to the total predicted positives.
        precision_score(y_true, y_pred, average=None)
        precision = tp / (tp + fp)

        # Accuracy score - What proportion the proportion of predictions classified correctly?
        # accuracy_score(y_true, y_pred) 
        accuracy = (tp+tn) / (tp+tn+fp+fn)
        
        auc = roc_auc_score(y_true, y_pred)
        r2 = glm.prsquared

        data.append([c, y, specificity, recall, accuracy, auc, r2])
        df_fit = pd.DataFrame(data,columns=('country', 'year', "specificity", "recall", "accuracy", 'prsquare', "auc"))
        df_fit.sort_values(by=['country',"year"], inplace=True)
        
        

# GRAPH
df_fit_long = pd.melt(df_fit, id_vars=["country","year"], value_vars=["specificity", "recall", "accuracy", 'prsquare', "auc"])
df_graph = df_fit_long[(df_fit_long["variable"] == "specificity") | (df_fit_long["variable"] == "recall") | (df_fit_long["variable"] == "accuracy")]
g = sns.FacetGrid(df_graph, 
                  col="country", 
                  col_wrap=6, 
                  hue = "variable", legend_out=True)
g.map_dataframe(sns.lineplot, x="year", y="value")
g.set_titles("{col_name}")  # use this argument literally
g.add_legend(loc="lower center", bbox_to_anchor=[.85, .1], ncol=1)



