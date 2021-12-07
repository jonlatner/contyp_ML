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

# df_Italy_1998 = pd.read_csv("/Users/jonathanlatner/Desktop/EU_LFS_2019_raw_data/YearlyFiles_83_2019/IT_YEAR_1998_onwards/IT1998_y.csv")
# df_Italy_1998 = df_Italy_1998[["COUNTRY", "YEAR", "TEMP", "AGE", "SEX", "MARSTAT", "COUNTRYB", "ILOSTAT", "FTPT", "HATLEV1D", "COEFF"]]
 
# file paths
raw_data_folder = "/Users/jonathanlatner/Desktop/EU_LFS_2019_raw_data/YearlyFiles_83_2019/"
data_files = "/Users/jonathanlatner/Documents/GitHub/contyp_ML/data_files"

df_country = []
for sub_folder in sorted(os.listdir(raw_data_folder)):
    # if sub_folder.startswith("IT_YEAR_1998_onwards"): # to do one specific country
    if sub_folder.endswith("1998_onwards"): # to do one specific period for all countries
    # if sub_folder.endswith("_stuff"): # to do one specific period for all countries
        dir_path = os.path.join(raw_data_folder, sub_folder)
        c = sub_folder[0:2]
        print(c)
        for file_name in sorted(os.listdir(dir_path)):
            if file_name.endswith(".csv"): # to do one specific period for all countries
                print(file_name)
                file_path = os.path.join(dir_path, file_name)
                df_country_year = pd.read_csv(file_path)
                df_country_year = df_country_year[["COUNTRY", "YEAR", "TEMP", "AGE", "SEX", "MARSTAT", "COUNTRYB", "ILOSTAT", "FTPT", "HATLEV1D", "COEFF"]]
                df_country_year = df_country_year.drop('COUNTRYB', 1)
                df_country_year = df_country_year.sample(frac=0.01, replace=True, random_state=1234)
                df_country.append(df_country_year)
                
df_output = pd.concat(df_country)
df_output.to_pickle(os.path.join(data_files,"df_eu_lfs_sample_1.pkl"))
# df_output = pd.read_pickle(os.path.join(data_files,"df_eu_lfs_sample_1.pkl"))

df_country = []
for sub_folder in sorted(os.listdir(raw_data_folder)):
    # if sub_folder.startswith("IT_stuff"): # to do one specific country
    # if sub_folder.startswith("IT_YEAR_1998_onwards"): # to do one specific country
    if sub_folder.endswith("1998_onwards"): # to do one specific period for all countries
    # if sub_folder.endswith("_stuff"): # to do one specific period for all countries
        dir_path = os.path.join(raw_data_folder, sub_folder)
        c = sub_folder[0:2]
        print(c)
        for file_name in sorted(os.listdir(dir_path)):
            if file_name.endswith(".csv"): # to do one specific period for all countries
                print(file_name)
                file_path = os.path.join(dir_path, file_name)
                df_country_year = pd.read_csv(file_path)
                df_country_year = df_country_year[["COUNTRY", "YEAR", "TEMP", "AGE", "SEX", "MARSTAT", "COUNTRYB", "ILOSTAT", "FTPT", "HATLEV1D", "COEFF"]]
                df_country_year = df_country_year.drop('COUNTRYB', 1)
                df_country_year = df_country_year.sample(frac=0.1, replace=True, random_state=1234)
                df_country.append(df_country_year)

df_output = pd.concat(df_country)
df_output.to_pickle(os.path.join(data_files,"df_eu_lfs_sample_10.pkl"))
# df_output = pd.read_pickle(os.path.join(data_files,"df_eu_lfs_sample_10.pkl"))
