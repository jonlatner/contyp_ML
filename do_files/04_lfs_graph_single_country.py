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
main_path = "/Users/jonathanlatner/Google Drive/SECCOPA/projects/"
data_path = "/Users/jonathanlatner/Desktop/python_stuff/data_files/"
dst_path = "distribution_contyp/data_files/eu_lfs/"
graphs_path = "/Users/jonathanlatner/Desktop/python_stuff/graphs/"
tables_path = "/Users/jonathanlatner/Desktop/python_stuff/tables/"

# top codes 
import os
import pandas as pd
import matplotlib.pyplot as plt

 # options
pd.set_option("display.max_columns", 40)
pd.set_option('display.float_format', lambda x: '%.5f' % x) # remove scientific notation

# load data
df0 = pd.read_csv(os.path.join(tables_path,"df_fit.csv"))
# df0 = pd.read_csv(os.path.join(tables_path,"df_fit_youdenJ.csv"))

df0['country'] = "Germany"

df0_long = pd.melt(df0, id_vars=["country"])
df0_long 

# GRAPH TRUE NEGATIVE
df_graph = df0_long[df0_long['variable'].str.contains("recall")]
df_graph = df0_long[df0_long['variable'].str.contains("_TN")]
df_graph["variable"] = df_graph["variable"].str.replace("recall_", "")
df_graph["variable"] = df_graph["variable"].str.replace("_TN", "")
# ax = df_graph.plot.bar(x='variable', y='value', rot=0, legend=False)

# Declaring the figure or the plot (y, x) or (width, height)
plt.figure(figsize = (12,7))# Categorical data: Country names
x = df_graph["variable"]
y = df_graph["value"]
plt.bar(x, y)

barplot = plt.bar(x, y)
for bar in barplot:
    yval = bar.get_height()
    # plt.text(position of x axis where you want to show text, position of y-axis, text, other_options)
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, str(round(float(yval), 2)), va='bottom') #va: vertical alignment y positional argument

# GRAPH TRUE POSITIVE
df_graph = df0_long[df0_long['variable'].str.contains("recall")]
df_graph = df0_long[df0_long['variable'].str.contains("_TP")]
df_graph["variable"] = df_graph["variable"].str.replace("recall_", "")
df_graph["variable"] = df_graph["variable"].str.replace("_TP", "")
# ax = df_graph.plot.bar(x='variable', y='value', rot=0, legend=False)

# Declaring the figure or the plot (y, x) or (width, height)
plt.figure(figsize = (12,7))# Categorical data: Country names
x = df_graph["variable"]
y = df_graph["value"]
plt.bar(x, y)

barplot = plt.bar(x, y)
for bar in barplot:
    yval = bar.get_height()
    # plt.text(position of x axis where you want to show text, position of y-axis, text, other_options)
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, str(round(float(yval), 2)), va='bottom') #va: vertical alignment y positional argument

# GRAPH ACCURACY
df_graph = df0_long[df0_long['variable'].str.contains("accuracy")]
df_graph["variable"] = df_graph["variable"].str.replace("accuracy_", "")
# ax = df_graph.plot.bar(x='variable', y='value', rot=0)

plt.figure(figsize = (12,7))# Categorical data: Country names
x = df_graph["variable"]
y = df_graph["value"]
plt.bar(x, y)

barplot = plt.bar(x, y)
for bar in barplot:
    yval = bar.get_height()
    # plt.text(position of x axis where you want to show text, position of y-axis, text, other_options)
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, str(round(float(yval), 2)), va='bottom') #va: vertical alignment y positional argument
