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
import seaborn as sns

# file paths
main_dir = "/Users/jonathanlatner/Documents/GitHub/contyp_ML/"
data_files = "data_files/"
graphs_files = "graphs/"
results_files = "results/"

# load data

df_fit = pd.read_csv(os.path.join(main_dir,results_files,"df_fit_10_single_country.csv"), index_col=0)
# df_fit.head()
df_fit_long = pd.melt(df_fit, id_vars=["country","year","type"])
# df_fit_long.head()
df_fit_long['variable'] = df_fit_long['variable'].astype("category")
df_fit_long['variable'].cat.reorder_categories(['accuracy', 'TN', "TP"])
df_fit_long['variable'].value_counts()
df_fit_long

g = sns.FacetGrid(df_fit_long, 
                  col="variable", 
                  # col_wrap=6, 
                  hue = "type", col_order = ["accuracy","TN", "TP"],
                  legend_out=True)
g.map_dataframe(sns.lineplot, x="year", y="value")
g.set_titles("{col_name}")  # use this argument literally
g.add_legend(bbox_to_anchor =(0.4,-0.1), loc='lower center', ncol=4, title="")
g.savefig(os.path.join(main_dir,graphs_files,"graph_single_country.pdf"), bbox_inches='tight')
