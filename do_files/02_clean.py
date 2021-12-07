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
 
# options
pd.set_option("display.max_columns", 40)
pd.set_option('display.float_format', lambda x: '%.5f' % x) # remove scientific notation

# file paths
main_dir = "/Users/jonathanlatner/Documents/GitHub/contyp_ML/"
data_files = "data_files/"
graphs_files = "graphs/"
tables_files = "tables/"

# load data
df0 = pd.read_pickle(os.path.join(main_dir,data_files,"df_eu_lfs_sample_10.pkl"))

# examine data
df0.head()
df0.tail()
df0.describe()
df0.columns
df0.index

# select data
df2=df0
df2.columns= df2.columns.str.lower()
df2=df2.rename(columns={"hatlev1d": "edu_cat", "coeff": "weight"})
df2=df2[(df2['year'] >= 1998)]
df2=df2[(df2['ilostat'] == 1)] # keep if employed
df2=df2[(df2['age'] >= 25) & (df2['age'] <= 54)]
df2=df2[(df2['weight'] > 0)]
df2=df2[(df2['temp'] == 1) | (df2['temp'] == 2)] # keep temporary or permanent contract missing temp
df2=df2[(df2['edu_cat'] == "L") | (df2['edu_cat'] == "M") | (df2['edu_cat'] == "H") ] # drop missing edu_cat
df2.describe()

print(len(df2.index))

df2['ilostat'].value_counts()

# df2=df2[(~df2['marstat'].isnull())] # drop missing marstat
# df2.describe()

# clean data

# age
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
df_edu_dummy = pd.get_dummies(df2['edu_cat'], prefix='edu_cat')
df_edu_dummy
df2 = pd.concat([df2, df_edu_dummy ], axis = 1)
del df_edu_dummy
df2["edu_cat"].value_counts(dropna=False)

df2["female"]=df2["sex"]-1 

df2["temp"]=df2["temp"]-1 
df2["temp"]=df2["temp"].astype(int)
df2["year"]=df2["year"].astype(int)

# Married
# df2['married']=df2['marstat'].replace([0,1,2],[0,0,1])
# pd.crosstab(df2["married"], df2["marstat"])
# df2["femstat"]=df2["female"]*df2["married"]

df2.describe()


# select columns
# df2=df2.drop(['sex',"ilostat","marstat","countryb","ftpt"], 1) # drop columns
df3=df2[['country',"year",
         "temp",
         "age_cat_Y","age_cat_O",
         "edu_cat_L","edu_cat_H",
         "female"]]

df3.head
df3.columns
df3.describe()

#Count total NaN at each column in DataFrame
df2.isnull().sum()

# save

df3.to_csv(os.path.join(main_dir,data_files,"df_eu_lfs_sample_10_clean.csv"))
df3.to_pickle(os.path.join(main_dir,data_files,"df_eu_lfs_sample_10_clean.pkl"))

# df3 = pd.read_pickle(os.path.join(main_dir,data_files,"df_eu_lfs_sample_1_clean.pkl"))



